import os.path as osp
import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from mmcv.cnn import normal_init


from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu():
    url = clip._MODELS["ViT-B/32"]
    model_path = clip._download(url, os.path.expanduser("~/.cache/clip"))
    
    try:
        # loading JIT archive
        print("jit version")
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None
    
    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')
    
    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
    
    def forward(self, prompts, tokenized_prompts):

        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):

    def __init__(self, classnames, clip_model, ctx=16):
        super().__init__()
        n_cls = len(classnames)    # number of classes
        n_ctx_di = ctx             # number of context words in domain invariant part
        self.n_ctx_di = n_ctx_di
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        domain_names = ['cityscapes', 'foggycityscapes']
        domain_templates = ['in a {} image'.format(domain_name) for domain_name in domain_names]
        n_dms = len(domain_names)  # number of domains
        n_ctx_ds = ctx             # number of context words in domain specific part
        self.n_dms = n_dms
        self.n_ctx_ds = n_ctx_ds
        n = n_ctx_di + n_ctx_ds    # number of context words in total

        #if random_init:
        prompt_prefix = ' '.join(['X'] * n)
        print(f'Initial context: "{prompt_prefix}"')

        print('Initializing a domain-invariant context')
        di_vectors = torch.empty(n_ctx_di, ctx_dim, dtype=dtype).to(torch.device("cuda"))
        nn.init.normal_(di_vectors, std=0.02)
        print(f'Number of domain-invariant context words (tokens): {n_ctx_di}')       
        self.ctx_di = nn.Parameter(di_vectors)

        print('Initializing a domain-specific context')
        ds_vectors = torch.empty(n_dms, n_ctx_ds, ctx_dim, dtype=dtype).to(torch.device("cuda"))
        nn.init.normal_(ds_vectors, std=0.02)
        print(f'Number of domain-specific context words (tokens): {n_ctx_ds}')
        self.ctx_ds = nn.Parameter(ds_vectors)

        classnames = [name.replace('_', ' ') for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + ' ' + name + ' ' + domain + '.' for domain in domain_templates for name in classnames]
        print(f'Prompts: {prompts}')

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(torch.device("cuda"))
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer('token_prefix', embedding[:, :1, :]) # SOS
        self.register_buffer('token_suffix', embedding[:, 1 + n:, :]) # CLS, EOS

        self.n_cls = n_cls
        self.tokenized_prompts = tokenized_prompts # torch.Tensor
        self.name_lens = name_lens

        self.clip_model = clip_model

        #
        temp = "A photo of a {} in a {} image."
        domain_discription = ['sunny', 'foggy']
        prompts_ = [temp.format(classname, domainname) for domainname in domain_discription for classname in classnames]
        print(f"Naive prompts: {prompts_}")
        prompts_ = torch.cat([clip.tokenize(p) for p in prompts_]).to(torch.device("cuda"))
        with torch.no_grad():
            text_features = clip_model.encode_text(prompts_)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.naive_text_features = text_features


    
    def forward(self):
        prefix = self.token_prefix
        suffix = self.token_suffix

        ctx_di = self.ctx_di        # [16, 512]
        ctx_dim = ctx_di.size(-1)
        ctx_ds = self.ctx_ds        # [n_dms, 16, 512]
        if ctx_di.dim() == 2:
            ctx_di = ctx_di.unsqueeze(0).expand(self.n_dms, -1, -1) # [n_dms, 16, 512]
            ctx_di = ctx_di.unsqueeze(1).expand(-1, self.n_cls, -1, -1) # [n_dms, n_cls, 16, 512]
        else: #ctx_di is class-wise
            ctx_di = ctx_di.unsqueeze(0).expand(self.n_dms, -1, -1,-1)  # [n_dms, n_cls, 16, 512]

        ctx_ds = ctx_ds.unsqueeze(1).expand(-1, self.n_cls, -1, -1) # [n_dms, n_cls, 16, 512]

        ctx = torch.cat([ctx_di, ctx_ds], dim=2).reshape(self.n_dms * self.n_cls, self.n_ctx_di + self.n_ctx_ds, ctx_dim) # [n_dms, n_cls, 32, 512]-> [n_dms * n_cls, 32, 512]
        prompts = torch.cat([
            prefix, # [n_cls, 1, 512]
            ctx,    # [n_dms * n_cls, 32, 512]
            suffix  # [n_cls, *, 512]
        ], dim=1)
        
        return prompts, self.tokenized_prompts


class DAPromptHead(nn.Module):

    def __init__(self, classnames, clip_model, ctx=16):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model,ctx)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        #self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        
        #self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.text_mapping = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),)
        '''
        for m in self.text_mapping.modules():
            if isinstance(m, nn.Linear):
                normal_init(m, std=0.01)
        '''
    
    def get_embedding(self):
        prompts, tokenized_prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)
        naive_text_features = self.prompt_learner.naive_text_features

        cos = nn.CosineSimilarity(dim=1, eps=1e-7)
        cos_score = cos(text_features, naive_text_features)
        cos_score = 1.0 - torch.mean(cos_score)

        #############
        #text_features = self.text_mapping(text_features.float())
        #text_features = text_features.unsqueeze(-1).unsqueeze(-1)
        #############
        return text_features.float(), cos_score.float()

    def forward(self, image_features):
        text_features, cos_score = self.get_embedding()

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = image_features @ text_features.t()
        return logits

