# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 14:28:10 2023

@author: sunnf
"""
import os
import cv2
import numpy as np
import segment_tools
from skimage import morphology
import time
# import sys
# from pathlib import Path
# from helpers_cv2 import *
from matplotlib import pyplot as plt

region_dir = 'D:\\deep_learning_work\\meta_segment_all\\regions\\'
region_outputs_dir = 'D:\\deep_learning_work\\meta_segment_all\\regions\\outputs\\'
result_dir='D:\\deep_learning_work\\meta_segment_all\\regions\\results\\'

img_bk_list = os.listdir(region_dir)
img_bk_list = [i for i in img_bk_list if 'output_BW.png' in i]

since = time.time()
start = time.localtime()
t = time.asctime(start)
print(t)


def get_contour(imgray):    
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def process_img(img_name, cell_size, kernel_size_adj):
    cell_size = cell_size
    img_label_name = 'D:\\deep_learning_work\\meta_segment_all\\processed\\' + img_name + '-label.png'
    org_image_name = 'D:\\deep_learning_work\\meta_segment_all\\processed\\'+ img_name + '.png'
    img_name = img_name + '-label.'+str(cell_size)+'output' #S13-01039-2-F.down-4-label.18output
    org_img = cv2.imread(org_image_name)
    print('processing:',img_name)
    org_img_gray = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)    
    
    
    cell_size = cell_size +kernel_size_adj
    if cell_size%2 == 0:
        cell_size += 1
    
    print('cell_size', cell_size)
    # Normalize the image
    #img_normalized = cv2.normalize(org_img_gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_16UC1)
    #(thresh, im_bw) = cv2.threshold(org_img_gray, 230, 255, cv2.THRESH_BINARY) #235, 240
    #segment_tools.saveImage(region_dir+img_name+'_BW.png', im_bw)
    
    img_bk_name = segment_tools.find_img_bk_name(img_name, img_bk_list)
    if img_bk_name == '':
        im_bw, mask = segment_tools.get_BW_BKGD(org_img_gray)    
        segment_tools.saveImage(region_dir+img_name+'_BW.png', im_bw)
        im_bw = mask
    else:
        im_bw = cv2.imread(region_dir +'\\'+ img_bk_name, cv2.IMREAD_GRAYSCALE)
    

    # hist = cv2.calcHist([org_img_gray], [0], None, [256], [0, 256])
    # print(hist[:240])
    # # normalize the histogram
    # #hist /= hist.sum()
    # # plot the normalized histogram
    # plt.figure()
    # plt.title("Grayscale Histogram (Normalized)")
    # plt.xlabel("Bins")
    # plt.ylabel("% of Pixels")
    # x_max = 240
    # x_min=200
    # plt.plot(hist[0:x_max])
    # plt.xlim([0, x_max+1])
    # plt.show()
    
    img_dir_name = region_outputs_dir + img_name + '.png'
    img_name_opened = region_dir + img_name + '_opened.png'
    # img_name_CONTOUR1 = region_dir + img_name + '_cont1.png'
    # img_name_CONTOUR2 = region_dir + img_name + '_cont2.png'
    # img_name_hull = region_dir + img_name + '_hull.png'
    print(img_dir_name)
    img = cv2.imread(img_dir_name)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_label = cv2.imread(img_label_name)

    # contours = get_contour(imgray)
    # mask = np.zeros(imgray.shape, np.uint8)
    # mask_contour1 = cv2.drawContours(mask, contours, -1, (255),1)
    # segment_tools.saveImage(img_name_CONTOUR1, mask_contour1)
    #kernel = np.ones((51,51),np.uint8)
    r = cell_size
    kernel = np.fromfunction(lambda x, y: ((x-r)**2 + (y-r)**2 <= r**2)*1, (2*r+1, 2*r+1), dtype=int).astype(np.uint8)
    mask_dilate = cv2.dilate(imgray,kernel,iterations = 2)
    mask_close = cv2.morphologyEx(mask_dilate, cv2.MORPH_CLOSE, kernel)
    
    kernel_size = (cell_size,cell_size) # should roughly have the size of the elements you want to remove
    kernel_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    #mask_eroded = mask_close
    mask_eroded =   cv2.erode(mask_close, kernel_el, (-1, -1))
    mask_cleaned = morphology.remove_small_objects(mask_eroded, min_size=5000, connectivity=1)
    # print(type(mask_cleaned))
    # print(type(im_bw))
    mask_cleaned = cv2.bitwise_and(mask_cleaned,im_bw) #im_bw is the background
    segment_tools.saveImageNoPrint(img_name_opened, mask_cleaned)
    
    # contours = get_contour(mask_dilate)
    # mask_contour2 = cv2.drawContours(mask, contours, -1, (255),1)
    # segment_tools.saveImage(img_name_CONTOUR2, mask_contour2)
    
    alpha = 0.5
    # print(mask_cleaned.shape)
    # print(img_label.shape)
    # Method 3: Convert mask to BGR
    mask_cleaned = cv2.cvtColor(mask_cleaned, cv2.COLOR_GRAY2BGR)
    img_overlap = cv2.addWeighted(mask_cleaned, alpha , img_label, 1-alpha, 0)
    result_img = cv2.addWeighted(img_overlap, alpha , org_img, 1-alpha, 0)
    
    #overlap brown BGR: 202,207,248
    BGR_MIN = np.array([202, 207, 248], np.uint8)
    BGR_MAX = np.array([202, 207, 248], np.uint8)
    dst = cv2.inRange(img_overlap, BGR_MIN, BGR_MAX)
    iou_pix = cv2.countNonZero(dst)
    #overlap green BGR: 128,255,128
    BGR_MIN = np.array([128, 255, 128], np.uint8)
    BGR_MAX = np.array([128, 255, 128], np.uint8)
    dst = cv2.inRange(img_overlap, BGR_MIN, BGR_MAX)
    iou_pix += cv2.countNonZero(dst)
    
    #ground brown Truth: 150, 159, 242
    BGR_MIN = np.array([150, 159, 242], np.uint8)
    BGR_MAX = np.array([150, 159, 242], np.uint8)
    dst = cv2.inRange(img_label, BGR_MIN, BGR_MAX)
    true_pix = cv2.countNonZero(dst)
    #ground green Truth: 0, 255, 0
    BGR_MIN = np.array([0, 255, 0], np.uint8)
    BGR_MAX = np.array([0, 255, 0], np.uint8)
    dst = cv2.inRange(img_label, BGR_MIN, BGR_MAX)
    true_pix += cv2.countNonZero(dst)
    
    #labeled, 255,255,255
    label_pix = np.sum (mask_cleaned == 255)/3    
    iou = iou_pix/(true_pix+label_pix-iou_pix)
    iou = '{:4.3f}'.format(iou)
    print('iou is:', iou,'iou_pix', iou_pix,'true_pix',true_pix,'label_pix',label_pix)
    
    img_name_overlap = region_dir + img_name +'_iou_'+str(iou)+ '_overlap.png'
    result_img_name = result_dir + img_name +'_iou_'+str(iou)+ '_overlap.png'
    segment_tools.saveImageNoPrint(img_name_overlap, img_overlap)
    segment_tools.saveImage(result_img_name, result_img)
    
since = time.time()
img_list = os.listdir(region_outputs_dir)
img_list = [i for i in img_list if 'output.png' in i]
img_list = [i for i in img_list if '.down-4-label.' in i]

test_flag = False

if test_flag == False:
    print('total images to be processes:', len(img_list))
    for img_name in img_list:
        cell_size=0
        if img_name.find('.0')<0:        
            index = img_name.find('-label.')
            #print(img_name)
            print(img_name[index+7:index+9])
            cell_size = int(img_name[index+7:index+9])
        img_name = img_name[:index]
        print('Processing img:', img_name)
        if cell_size!=0:
            process_img(img_name, cell_size,4)
            time_elapsed = time.time() - since
        print('end_time complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

if test_flag == True:
    img_name = 'S15-25211-1-F.down-4'
    img_name = 'S10-3238.down-4'
    img_name='S19-13814-1-F.down-4'
    img_name = 'S13-01039-2-F.down-4'
    img_name = 'S10-1999-1.down-4'
    img_name = 'S10-1999-1-F.down-4'
    img_name = 'S10-33024-1-F.down-4'
    img_name = 'S10-33024-2-F.down-4'
    img_name = 'S11-6409-1.down-4'   #21, 8
    img_name = 'S11-6409-1-F.down-4' #20, 20
    img_name = 'S11-8518-1-F.down-4'
    img_name = 'S12-09804-1.down-4' #4
    
    #############################################
    #single image
    process_img(img_name,13, 4)#4#20#6#-2#-8#16

time_elapsed = time.time() - since
print('end_time complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))