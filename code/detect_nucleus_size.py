# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 22:24:41 2023

@author: sunnf
"""
import os
import cv2
import numpy as np
import segment_tools
import time
import pickle

img_dir = '..\\data\\images'
regions_dir = '..\\data\\regions'
img_bw_dir = '..\\data\\images\\img_BW'
img_list = os.listdir(img_dir)
img_list = [i for i in img_list if '.down-4.png' in i]
img_bk_list = os.listdir(regions_dir)
img_bk_list = [i for i in img_bk_list if 'output_BW.png' in i]

# print(img_bk_list)

def liverCellBlobs(image):
    kernel = np.ones((2,2),np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    # saveImage('liver_morph.png',image)
    # image_blue = image[:,:,0]
    # saveImage('liver_blue.png',image_blue)
    # ret,image = cv2.threshold(image,200,255,cv2.THRESH_BINARY)
    # saveImage('threshold.png',image)
    
    params=cv2.SimpleBlobDetector_Params()
    params.filterByConvexity=True
    params.minConvexity=0.35
    params.filterByInertia=False
    #params.minInertiaRatio = 0.2

    #params.maxThreshold = 20
    # params.filterByArea=True
    params.minArea=8
    params.maxArea=50
    

    detector=cv2.SimpleBlobDetector_create(params)
    
    keypoints=detector.detect(image)
    return keypoints
    # blank=np.zeros((1,1))
    # blobs=cv2.drawKeypoints(image,keypoints,blank,(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # number_of_blobs=len(keypoints)
    # text="total no of blobs"+str(len(keypoints))
    # cv2.putText(blobs,text,(20,550),cv2.FONT_HERSHEY_SIMPLEX,1,(100,0,255),2)
    # cv2.imshow('blob using default parameters',blobs)
    # cv2.imwrite(filename+'.jpg',blobs)
    # cv2.waitKey(0)
    #print('blob#',number_of_blobs)

def get_keypoints_mean_median(keypoints):
    size_list =[]
    for k in keypoints:
        size_list.append(k.size)
    
    size_list.sort()
    length = len(size_list)
    sz_mean = sum(size_list)/length
    sz_median = size_list[int(length/2)]
    return sz_mean,sz_median
        
     ############     Calculate average image brightness of imgray     #####################
def get_img_mean(imgray, img_bk, mean_dict, img_name):   
    w = imgray.shape[0]
    h = imgray.shape[1]
    mean = np.average(imgray)
    no_black = np.sum(img_bk == 0)
    imgray_mean = mean * ((w*h)/(w*h-no_black))
    mean_dict[img_name] = imgray_mean
    
    
def process_img(img_name, nucleus_dict, mean_dict):
    nucleus_flag = False
    mean_flag = False
    #if dict already has this image info, do not process
    if img_name+'_mean' in nucleus_dict:
        print('skip nucleus:', img_name)
        nucleus_flag = True
        
    if img_name in mean_dict:
        print('skip mean:', img_name)
        mean_flag = True
    
    if nucleus_flag==True and mean_flag == True:
        return
    
    # load images
    img = cv2.imread(img_dir+'\\'+img_name) 
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_bk_name = segment_tools.find_img_bk_name(img_name, img_bk_list)
    img_bk = ''
    mask = ''
    is_old_img_bk = True
    if img_bk_name=='':
        im_bk,mask = segment_tools.get_BW_BKGD(imgray)    
        segment_tools.saveImage(regions_dir+'\\'+img_name+'output_BW.png', im_bk)
        print(img_name, ' is processed with new bkground img!')
        is_old_img_bk = False
    else:
        print('process with old bkgd:', img_bk_name)
        img_bk = cv2.imread(regions_dir +'\\'+ img_bk_name, cv2.IMREAD_GRAYSCALE)

    if is_old_img_bk:
        imgray = cv2.bitwise_and(imgray,imgray, mask=img_bk)
    else:
        #img_bk = cv2.cvtColor(img_bk, cv2.COLOR_BGR2GRAY)
        imgray = cv2.bitwise_and(imgray,imgray, mask=mask)
        # imgray = cv2.bitwise_and(imgray,img_bk)

    # if nucleus_dict does not contain this image nucleus size
    if nucleus_flag == False:
        ############     Process liver points         #####################
        liver_points = liverCellBlobs(imgray)
        mean, median = get_keypoints_mean_median(liver_points)
        nucleus_dict[img_name+'_mean'] = mean
        nucleus_dict[img_name+'_median'] = median
        #draw liver points on grayImage
        blank=np.zeros((1,1))
        img_BW_name = img_name.replace('.png', '_BW'+'{:4.3f}'.format(mean)+'_'+'{:4.3f}'.format(median)+'.png')
        imgray=cv2.drawKeypoints(imgray,liver_points,blank,(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        segment_tools.saveImage(img_bw_dir +'\\'+ img_BW_name, imgray)
    
    if mean_flag == False:
        get_img_mean(imgray, img_bk, mean_dict, img_name)
    
    
    
    
    
    
since = time.time()
nucleus_dict={}
mean_dict = {}

    
try:
    #load nucleus_dict from pkfile
    with open('nucleus.pkfile', 'rb') as nucleus_file: 
        nucleus_dict = pickle.load(nucleus_file)
except (OSError, IOError) as e:
    print('nucleus.pkfile does not exist, we will create a new one')
    
    
try:
    #load nucleus_dict from pkfile
    with open('img_mean.pkfile', 'rb') as mean_file: 
        mean_dict = pickle.load(mean_file)
except (OSError, IOError) as e:
    print('mean_file.pkfile does not exist, we will create a new one')

for img_name in img_list:
    process_img(img_name, nucleus_dict, mean_dict)
    segment_tools.save_masks_pickle(nucleus_dict, 'nucleus.pkfile')
    segment_tools.save_masks_pickle(mean_dict, 'img_mean.pkfile')

print(mean_dict)
    
# img_name = 'S15-25211-2-F.down-4.png'
# process_img(img_name, nucleus_dict)
# img_name = 'S15-09829-1.down-4.png'
# process_img(img_name)

time_elapsed = time.time() - since
print('end_time complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    

