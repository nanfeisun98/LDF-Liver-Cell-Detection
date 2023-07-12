# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 19:47:58 2023

@author: sunnf
"""
import os
import csv
import cv2
import numpy as np
import time

since = time.time()
start = time.localtime()
t = time.asctime(start)
print(t)

fields = ['subject', 'liver size']#,'cell size','cell no']
rows = []

img_BW_dir = 'D:\\deep_learning_work\\meta_segment_all\\regions'
img_list = os.listdir(img_BW_dir)
img_list = [i for i in img_list if '.png' in i]
img_list = [i for i in img_list if 'BW' in i]


pre_img_short_name = ''
for img_name in img_list:
    row = []
    img_short_name = img_name[0:img_name.find('.')]
    img_full_name = img_BW_dir + '\\' + img_name
    img = cv2.imread(img_full_name, cv2.IMREAD_GRAYSCALE)
    n_white_pix = np.sum(img == 255)
    cell_sz = 0
    cell_no = 0
    if pre_img_short_name != img_short_name:
        print(img_short_name, 'liver size:', n_white_pix)#, 'cell size:',cell_sz,'cell no:', cell_no )
        pre_img_short_name = img_short_name
        row.append(img_short_name)
        row.append(n_white_pix)
        rows.append(row)

with open('liversize', 'w', newline='') as f:
     
    # using csv.writer method from CSV package
    write = csv.writer(f)     
    write.writerow(fields)
    write.writerows(rows)
    
time_elapsed = time.time() - since
print('end_time complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))