# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 14:53:21 2023

@author: sunnf
"""
import os
import csv
import numpy as np
import cv2
import time
import pickle
from read_roi import read_roi_file
from read_roi import read_roi_zip
import segment_tools

since = time.time()
start = time.localtime()
t = time.asctime(start)
print(t)

patch_size = 512

def getPoints(rois):
    key_points = []
    
    for r in rois:
        if rois[r]['type']!='point':
            continue
        x_list = rois[r]['x']
        y_list = rois[r]['y']
        for (x,y) in zip(x_list,y_list):
            #print(x, y, 0)
            point = cv2.KeyPoint(x, y, 10)
            key_points.append(point)        
    return key_points
    
def writeDict_2_csv(results, results_columns,csv_filename):
    try:
        with open(csv_filename, 'w', newline='') as csvfile: #must have newline='' to avoid  extra blank row
            writer = csv.DictWriter(csvfile, fieldnames=results_columns)
            writer.writeheader()
            for data in results:
                #print('data',data)
                writer.writerow(data)
    except IOError:
        print("I/O error")
    

def find(j_points, filtered_masks, masks, img_result):
    #predictive_p_and_actual_p
    tp_masks = []
    tp_points = []
    #predictive_n_and_actual_p 
    fn_masks = []
    #predictive_p_and_actual_n 
    fp_masks = []
    #all masks in filtered_masks are pp
    for mask in filtered_masks:
       if segment_tools.has_j_point(mask, j_points, tp_points):
           tp_masks.append(mask)
       else:
           fp_masks.append(mask)
           
    # if len(tp_masks) != len(tp_points):
    #     print('warning, tp_mask is:', len(tp_masks))
    #     print('tp_points is:', len(tp_points))
    
    AP = len(j_points)
    FN = len(j_points) - len(tp_points)
    PP = len(filtered_masks)
    PN = len(masks) + FN
    TP = len(tp_points)
    FP = len(fp_masks)    
    TN = len(masks) - PP 
    AN = len(masks) - TP 
    total_points = len(masks) + FN
    img_result['TP']='{:4d}'.format(TP)
    img_result['FP']='{:4d}'.format(FP)
    img_result['FN']='{:4d}'.format(FN)
    img_result['TN']='{:5d}'.format(TN)
    
    sensitivity = 1
    specificity = 1
    accuracy = 1
    precision = 1

    if total_points > 0:
        accuracy = (TP + TN)/total_points
    if AP != 0:
        sensitivity = TP/AP
    if AN !=0:
        specificity = TN/AN
    if AP==0:
        precision = 1
    elif PP != 0:
        precision = TP/PP
    
    IOU=0
    if TP+FP+FN > 0:
        IOU=TP/(TP+FP+FN)
    print('TP:', TP, 'FN', FN)
    print('TN:', TN, 'FP', FP)
    return IOU,sensitivity, specificity, precision, tp_points

def process_mask(maskfolder_flag, mask_name, img_result, mask_blobs_all, nucleus_dict, mean_dict):
    mask_filename = ''
    mask_foldername = ''
    img_name = ''
    if maskfolder_flag == True:
        mask_foldername = mask_folder_dir+'\\'+maskname
        img_name = maskname + '.png'
    else:
        mask_filename = mask_dir+'\\'+maskname
        img_name = maskname.replace('_masks.pkfile','')
    liver_mean = 5.0
    liver_median = 5.0
    
    if img_name+'_mean' in nucleus_dict:        
        liver_mean = nucleus_dict[img_name+'_mean']   
        liver_median = nucleus_dict[img_name+'_median']
    print('liver_mean',liver_mean, 'liver_median', liver_median)
    img_mean = 220
    if img_mean in mean_dict:
        img_mean = mean_dict[img_name]
    img_result['image_name']='{:25s}'.format(img_name)
    roi_name = img_name.replace('.down-4.png','.rois.zip')
    img_dir_name = img_dir + '\\' + img_name
    roi_dir_name = img_dir + '\\' + roi_name
    image = cv2.imread(img_dir_name)
    rois = read_roi_zip(roi_dir_name)
    j_points = getPoints(rois)
    
    if mask_filename != '':
        print(mask_filename)
        with open(mask_filename, 'rb') as masks_file: 
            masks = pickle.load(masks_file)
    elif mask_foldername != '':
        print(mask_foldername)
        masks = segment_tools.load_maskfolder(mask_foldername)
        
    median_area, min_area, max_area = segment_tools.get_median_area(masks)
    print('process mask:', mask_filename)
    print('median_area', median_area, min_area, max_area)
    filter_flag = {}
    filter_flag['check_iou'] = True
    filter_flag['check_mean'] = True
    filter_flag['check_cnt'] = True
    filter_flag['check_blob'] = True
    
    # filter_flag['check_iou'] = True
    # filter_flag['check_cnt'] = True
    # # filter_flag['check_mean'] = True
    # filter_flag['check_blob'] = True
    
    save_cell = False
    # save_cell = True
    cell_mean = img_mean + (255 - img_mean) * 3/4
    inner_mean = 241 #S10-1999-1, only counts large and bright cells
    blob_filter = {}
    blob_filter['extent'] = 0.7
    blob_filter['circularity'] = 0.8 #no circularity requirements, unless the no. of cells are too less
    blob_filter['extent_cir_sum'] = 1.5
    blob_filter['solidity'] = 0.75
    blob_filter['cnt_diff'] = 1
    useRegression = -1    
    mean_diff_flag = True
    predicted_iou = 0.95
    min_scale = (liver_mean /5.5)
    
    if filter_flag['check_iou'] == True:
        print('iou',predicted_iou )
    image_copy,filtered_masks, mask_blobs = segment_tools.draw_bbox(patch_size, masks, j_points,image, img_name, 0, 0, filter_flag,median_area, liver_mean, min_scale,mean_diff_flag,blob_filter,   predicted_iou,          inner_mean, save_cell, useRegression)
    filtered_masks = segment_tools.delete_lonely_mask(filtered_masks,5)
    
    # print('filtered_masks',len(filtered_masks))
    if len(filtered_masks)<300 and len(filtered_masks) > 100:
        print('run 2nd attempt')
        #if liver is large and we only have fewer fat droplets, relax the predicted_iou, and increase circularity
        if liver_mean > 5.4:
            predicted_iou = 0.95
            circularity_lmt = 0.8
         
        #S10-1999-1, only counts large and bright cells
        # inner_mean = min(230,  inner_mean)  
        inner_mean = 241
        # min_scale = 0.7
        if filter_flag['check_iou'] == True:
            print('iou',predicted_iou )
        image_copy,filtered_masks,mask_blobs = segment_tools.draw_bbox(patch_size, masks, j_points,image, img_name, 0, 0, filter_flag,median_area,liver_mean,min_scale,mean_diff_flag,blob_filter,predicted_iou, inner_mean, save_cell, useRegression)
        filtered_masks = segment_tools.delete_lonely_mask(filtered_masks,5)
    
    # if len(filtered_masks)<50 and len(filtered_masks) > 0:
    #     print('run allows circularity changes')       
    #     blob_filter['cnt_diff'] = 1 #default is 1.0
    #     inner_mean = cell_mean
    #     print(cell_mean)
    #     # predicted_iou = 0.9
    #     # inner_mean = min(235,  inner_mean)  
    #     image_copy,filtered_masks,mask_blobs = segment_tools.draw_bbox(masks, j_points,image, img_name, 0, 0, filter_flag,median_area,liver_mean,min_scale,mean_diff_flag,blob_filter,predicted_iou, inner_mean, save_cell, useRegression)
    #     filtered_masks = segment_tools.delete_lonely_mask(filtered_masks,5)
        
    # if len(filtered_masks)< 50:
    #     print('run 4th attempt')
    #     inner_mean = 240 #S12-09804-2-F
    #     inner_mean = 228 #S10-33024-2-F
    #     circularity_lmt = 0.7 #S10-33024-2-F
    #     predicted_iou = 0.9
    #     image_copy,filtered_masks,mask_blobs = segment_tools.draw_bbox(masks, j_points,image, img_name, 0, 0, filter_flag,median_area,liver_mean,min_scale,mean_diff_flag,blob_filter,predicted_iou, inner_mean, save_cell, useRegression)
    #     filtered_masks = segment_tools.delete_lonely_mask(filtered_masks,5)
   
    
    if len(filtered_masks)>600: #get more small droplets
        print('run 5th attempt')
        mean_diff_flag = True
        # S10-1999-1-F, relax all blob feature, large cell
        # blob_filter['extent'] = -1 #0.65
        blob_filter['solidity'] = 0.75
        blob_filter['extent_cir_sum'] = 1.5
        # blob_filter['circularity'] = -1# 0.7 #no circularity requirements, unless the no. of cells are too less
       
        blob_filter['cnt_diff'] = 1 #default is 1.0
        min_scale = 0.7 #median * meidan * meidan / (5.8*5.8)
        inner_mean = 240 # S10-1999-1-F
        predicted_iou = 0.9 #S12-09804-1
        if filter_flag['check_iou'] == True:
            print('iou',predicted_iou )
        image_copy,filtered_masks, mask_blobs = segment_tools.draw_bbox(patch_size, masks, j_points,image, img_name,0, 0, filter_flag,median_area,liver_mean, min_scale,mean_diff_flag,blob_filter, predicted_iou, inner_mean, save_cell, useRegression)
    
    mask_blobs_all.extend(mask_blobs)
    print('final filtered masks:',len(filtered_masks))
    #save filtered_masks, in new folder structurem do not save filteres mask
    # filtered_mask_filename = mask_filename.replace('png_masks','png_filtered_masks')
    # segment_tools.save_masks_pickle(filtered_masks, filtered_mask_filename)
    blank=np.zeros((1,1))
    #draw all points red, later draw TP points black. (so red will be only FP points)
    image_copy=cv2.drawKeypoints(image_copy,j_points,blank,(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)    
    IOU, sensitivity, specificity, precision, tp_points = find(j_points, filtered_masks, masks, img_result)
    image_copy=cv2.drawKeypoints(image_copy,tp_points,blank,(0,0,0),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    segment_tools.saveImage(img_dir+'\\'+img_name+'_result.png',image_copy)
    print(img_name, 'sensitivity=', sensitivity,' specificity = ', specificity )
    img_result['IOU'] = '{:4.3f}'.format(IOU)
    img_result['sensitivity'] = '{:4.3f}'.format(sensitivity)
    img_result['specificity'] = '{:4.3f}'.format(specificity)
    img_result['precision'] = '{:4.3f}'.format(precision)
    


mask_dir='..\\data\\masks_'+str(patch_size)
mask_folder_dir = '..\\data\\masks_'+str(patch_size)
img_dir = '..\\data\\images'
mask_list = os.listdir(mask_dir)
mask_folder_list = os.listdir(mask_folder_dir)
print(mask_folder_dir)
print(mask_folder_list)

mask_list = [i for i in mask_list if '.pkfile' in i]
results = []
mask_blobs_all=[]
counter = 0
results_columns = ['image_name','TP','FP','TN','FN','IOU','sensitivity','specificity','precision']
mask_blobs_columns = ['image_name','id','area','perimeter','convexity','circularity','aspect_ratio','rect_area','extent','hull_area','solidity','mean','std','range','bd_mean','bd_std','bd_range','mean_diff','predicted_iou', 'is_j_point']
#load liver size
nucleus_dict = {}
mean_dict = {}
#load nucleus_dict from pkfile
with open('nucleus.pkfile', 'rb') as nucleus_file: 
    nucleus_dict = pickle.load(nucleus_file)
    
#load mean_dict from pkfile
with open('img_mean.pkfile', 'rb') as mean_file: 
    mean_dict = pickle.load(mean_file)
    
test_flag = False
maskfolder_flag = True

if test_flag == False:
    if maskfolder_flag == True:
        mask_list = mask_folder_list
        print(mask_list)
    for maskname in mask_list:
        img_result={}   
        #load mask list
        
        process_mask(maskfolder_flag, maskname, img_result, mask_blobs_all, nucleus_dict, mean_dict)
        results.append(img_result)
        print(results) 
    
        writeDict_2_csv(results, results_columns,'results.csv')
        writeDict_2_csv(mask_blobs_all, mask_blobs_columns,'mask_blobs.csv')
        counter+=1
        
        time_elapsed = time.time() - since
        print('end_time complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

##################################################
#process single mask

maskname_lst=[]
# maskname='S19-7528-2-F.down-4.png_masks.pkfile'
# maskname='S19-13814-1-F.down-4.png_masks.pkfile'

# maskname='S13-01039-2-F.down-4.png_masks.pkfile'
# maskname='S10-1999-1-F.down-4.png_masks.pkfile'
#maskname='S11-28445-1-F.down-4.png_masks.pkfile' #jj mark too small
#maskname='S11-6409-1-F.down-4.png_masks.pkfile'   #large irregular size marked, small irregular marked
# maskname='S12-09804-1.down-4.png_masks.pkfile' #many super small cells, min size /2 to /4 best iou, /6 best FN
# maskname='S12-09804-1-F.down-4.png_masks.pkfile'
# maskname='S13-23325-1-F.down-4.png_masks.pkfile'
maskname = 'S16-8578-1-F.down-4.png_masks.pkfile' # less than 100 and super small blobs
maskname = 'S14-1646-3-F.down-4.png_masks.pkfile' #Too many small FP droplets
#maskname = 'S11-8518-1-F.down-4.png_masks.pkfile' # 50% FP too small, 50% FP look similiar to fat droplet, except their borders are less smoothy
maskname = 'S18-25172-1-F.down-4.png_masks.pkfile'


# maskname='S10-33024-1-F.down-4.png_masks.pkfile'
# maskname='S10-33024-2-F.down-4.png_masks.pkfile'
# maskname='S11-28445-1-F.down-4.png_masks.pkfile'
# maskname ='S12-02907.down-4.png_masks.pkfile'
# maskname ='S12-09804-1-F.down-4.png_masks.pkfile'
# maskname ='S12-09804-1.down-4.png_masks.pkfile'
maskname ='S12-17257-2-F.down-4.png_masks.pkfile'
maskname ='S12-19059.down-4.png_masks.pkfile'
maskname ='S13-1039-1-F.down-4.png_masks.pkfile'
maskname ='S13-23325-1-F.down-4.png_masks.pkfile'
maskname = 'S15-19472.down-4.png_masks.pkfile'
maskname = 'S16-8578-1-F.down-4.png_masks.pkfile' # less than 100 and super small blobs
maskname = 'S19-19001-1-F.down-4.png_masks.pkfile' # less than 100 and super small blobs
maskname='S10-3238.down-4.png_masks.pkfile'
maskname='S10-1999-1-F.down-4.png_masks.pkfile'

# maskname_lst.append('S10-33024-2-F.down-4.png_masks.pkfile') # reduce inner mean to get less FN
# maskname_lst.append('S10-33024-1-F.down-4.png_masks.pkfile')
# maskname_lst.append('S12-09804-1-F.down-4.png_masks.pkfile')
# maskname_lst.append('S12-09804-2-F.down-4.png_masks.pkfile')
# maskname_lst.append('S13-01039-2-F.down-4.png_masks.pkfile')
# maskname_lst.append('S16-17637-1.down-4.png_masks.pkfile')
# maskname_lst.append('S16-8578-1-F.down-4.png_masks.pkfile')
# maskname_lst.append('S10-1999-1-F.down-4.png_masks.pkfile')
# maskname_lst.append('S10-3238.down-4.png_masks.pkfile')
# maskname_lst.append('S14-1953-3-f.down-4.png_masks.pkfile')
# maskname_lst.append('S11-28445-1.down-4.png_masks.pkfile')


# maskname_lst.append('S10-1999-1.down-4.png_masks.pkfile')
# maskname_lst.append('S10-1999-1-F.down-4.png_masks.pkfile') #trouble img, too many  FP is not marked
# maskname_lst.append('S10-2361-1-F.down-4.png_masks.pkfile')
# maskname_lst.append('S10-33024-1-F.down-4.png_masks.pkfile') #trouble img, cell streched
# maskname_lst.append('S10-33024-2-F.down-4.png_masks.pkfile')
# maskname_lst.append('S11-6409-1.down-4')
# maskname_lst.append('S11-6409-1-F.down-4')
# maskname_lst.append('S11-8518-1-F.down-4.png_masks.pkfile')
# maskname_lst.append('S12-09804-1.down-4.png_masks.pkfile')   #trouble img, super small droplets, cause huge FP
# maskname_lst.append('S12-09804-1-F.down-4.png_masks.pkfile') 
# maskname_lst.append('S12-02907.down-4.png_masks.pkfile')
# maskname_lst.append('SB10-1478-3-F.down-4.png_masks.pkfile')
# maskname_lst.append('S12-02907.down-4.png_masks.pkfile')
# maskname_lst.append('S10-1999-1.down-4')
maskname_lst.append('S10-14139-1.down-4')

print(test_flag)
if test_flag == True:
    img_result={}
    for maskname in maskname_lst: 
        process_mask(maskfolder_flag, maskname, img_result,mask_blobs_all, nucleus_dict, mean_dict)
        # writeDict_2_csv(mask_blobs_all, mask_blobs_columns,'mask_blobs.csv')
        print(img_result)

time_elapsed = time.time() - since
print('end_time complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))