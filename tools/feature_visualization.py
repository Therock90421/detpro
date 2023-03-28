import cv2
import mmcv
import numpy as np
import os
import torch
import matplotlib.pyplot as plt


def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = feature_map[:,0,:,:]*0
    heatmaps = []
    for c in range(feature_map.shape[1]):
        heatmap+=feature_map[:,c,:,:]
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)

    heatmap /= feature_map.shape[1]
    max = np.max(heatmap)
    min = np.min(heatmap)
    print(max)
    print(min)
    heatmap -= min
    heatmap /= (max - min)
    #heatmap /= np.max(heatmap)
    heatmaps.append(heatmap)

    return heatmaps

def draw_heat_map(features,save_dir = '/GOODDATA/lihaochen/DSS/feature_map/',name = None):
    i = 0
    if isinstance(features,torch.Tensor):
        for heat_maps in features:
            print(heat_maps.shape)
            heat_maps=heat_maps.unsqueeze(0)
            print(heat_maps.shape)
            heatmaps = featuremap_2_heatmap(heat_maps)
            print(heat_maps.shape)
            # 这里的h,w指的是你想要把特征图resize成多大的尺寸
            # heatmap = cv2.resize(heatmap, (h, w))  
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.resize(heatmap, (2048, 1024))  # 将热力图的大小调整为与原始图像相同
                # 下面这行将热力图转换为RGB格式 ，如果注释掉就是灰度图
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap
                #plt.imshow(superimposed_img,cmap='gray')
                #plt.show()
                print(save_dir,name +str(i)+'.png')
                cv2.imwrite(os.path.join(save_dir,name + str(i)+'.png'), superimposed_img)
                i += 1
    else:
        for featuremap in features:
            heatmaps = featuremap_2_heatmap(featuremap)
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
                heatmap = cv2.resize(heatmap, (2048, 1024))  # 将热力图的大小调整为与原始图像相同
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_RAINBOW)
                # superimposed_img = heatmap * 0.5 + img*0.3
                superimposed_img = heatmap
                #plt.imshow(superimposed_img,cmap='gray')
                #plt.show()
                # 下面这些是对特征图进行保存，使用时取消注释
                #cv2.imshow("1",superimposed_img)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                print(save_dir,str(i)+'.png')
                cv2.imwrite(os.path.join(save_dir,name + str(i)+'.png'), superimposed_img)
                i += 1

def featuremap_to_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = feature_map[:,0,:,:]*0
    heatmaps = []
    for c in range(feature_map.shape[1]):
        heatmap+=feature_map[:,c,:,:]
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)

    heatmap /= np.max(heatmap)
    heatmaps.append(heatmap)

    return heatmaps

def draw_feature_map(features,save_dir = '/GOODDATA/lihaochen/DSS/feature_map/',name = None):
    i = 0
    if isinstance(features,torch.Tensor):
        for heat_maps in features:
            print(heat_maps.shape)
            heat_maps=heat_maps.unsqueeze(0)
            print(heat_maps.shape)
            heatmaps = featuremap_to_heatmap(heat_maps)
            print(heat_maps.shape)
            # 这里的h,w指的是你想要把特征图resize成多大的尺寸
            # heatmap = cv2.resize(heatmap, (h, w))  
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.resize(heatmap, (2048, 1024))  # 将热力图的大小调整为与原始图像相同
                # 下面这行将热力图转换为RGB格式 ，如果注释掉就是灰度图
                #heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap
                #plt.imshow(superimposed_img,cmap='gray')
                #plt.show()
                print(save_dir,name +str(i)+'.png')
                cv2.imwrite(os.path.join(save_dir,name + str(i)+'.png'), superimposed_img)
                i += 1
    else:
        for featuremap in features:
            heatmaps = featuremap_to_heatmap(featuremap)
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
                heatmap = cv2.resize(heatmap, (2048, 1024))  # 将热力图的大小调整为与原始图像相同
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_RAINBOW)
                # superimposed_img = heatmap * 0.5 + img*0.3
                superimposed_img = heatmap
                #plt.imshow(superimposed_img,cmap='gray')
                #plt.show()
                # 下面这些是对特征图进行保存，使用时取消注释
                #cv2.imshow("1",superimposed_img)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                print(save_dir,str(i)+'.png')
                cv2.imwrite(os.path.join(save_dir,name + str(i)+'.png'), superimposed_img)
                i += 1