"""
Created on Wed Apr 19 09:52:27 2023

@author: sunnf
"""
import cv2
import time
import os
import pickle
import segment_tools
import numpy as np
import math

patch_size = 512

since = time.time()
start = time.localtime()
t = time.asctime(start)
print(t)

def get_avg_mask_size(filtered_mask):
    total_size = 0
    for mask in filtered_mask:
        bbox = mask['bbox']
        h = int(bbox[2])
        w = int(bbox[3])
        total_size+=(h+w)/2
    
    avg_size = 0
    if len(filtered_mask)!=0:
        avg_size = int(total_size/len(filtered_mask))
    return avg_size


                
def merge_masks(masks):
    or_output = list(map(any, zip(*masks)))
    return or_output
        

def create_lp_by_mask1(img_new, masks, draw_pixel_by_pixel):    
    img_row = img_new.shape[0]
    img_col = img_new.shape[1]
    print('img_row', img_row,'img_col',img_col)
    count = 0
    for my_mask in masks:
        x = int(my_mask['bbox'][0])
        y = int(my_mask['bbox'][1])
        w = int(my_mask['bbox'][2])
        h = int(my_mask['bbox'][3])
        mask_base_x = x - x%patch_size
        mask_base_y = y - y%patch_size
        #print('mask_base_x', mask_base_x, 'mask_base_y',mask_base_y)
        if draw_pixel_by_pixel==True:
            segmentations = my_mask['segmentation']
            row_size = len(segmentations)
            col_size = len(segmentations[0])
            #print('seg_row', row_size, 'seg_col',col_size)
            for row in range(row_size):
                for col in range(col_size):
                    if segmentations[row][col] == True:
                        #if img_row > mask_base_y+row  and img_col > mask_base_x+col:                  
                        img_new[mask_base_y+row, mask_base_x+col] = (255, 255, 255)
        else:
            #draw bbox
            img_new[y:y+h, x:x+w] = (255, 255, 255)
        count+=1
        #print('finished mask ', count)
        # if count>200:
        #     break
        #patch = cv2.bitwise_and(img_new,img_new,mask = mask)
        
        #img_new[mask_base_x:mask_base_x+w, mask_base_y:mask_base_y+h] = img_new[mask_base_x:mask_base_x+w, mask_base_y:mask_base_y+h]
        #break
    
    return img_new

def process_image(img_name):    
    img_label_name=img_name+'-label'
    image_dir = 'D:\\deep_learning_work\\meta_segment_all\\processed'
    
    region_dir = 'D:\\deep_learning_work\\meta_segment_all\\regions\\'
    filtered_mask_dir=mask_dir+'\\'+img_name+'.png_filtered_masks.pkfile'
    

    img_label = cv2.imread(image_dir+'\\'+img_label_name+'.png')
    #create a new black imagw with same size
    w = img_label.shape[0]
    h = img_label.shape[1]
    # print('image size=', w,'x',h)
    img_new = np.zeros([w,h,3],dtype=np.uint8)
    #img_new = np.full((w, h, 3), 255, dtype=np.uint8)
    


    filtered_masks = ''
    if os.path.exists(filtered_mask_dir):
        with open(filtered_mask_dir, 'rb') as filtered_mask_file: 
            filtered_masks = pickle.load(filtered_mask_file)
    
    #image = cv2.imread(image_dir+'\\'+img_name+'.png')
    #if no pickle file
    if filtered_masks =='':
        print('no mask file', filtered_mask_dir)    
    else:
        #get median size of filtered_masks
        if len(filtered_masks) > 200:
            cleaned_mask = segment_tools.delete_lonely_mask(filtered_masks,3.5)
        else:
            cleaned_mask = segment_tools.delete_lonely_mask(filtered_masks,9)
        cell_size = get_avg_mask_size(cleaned_mask)
        time_elapsed = time.time() - since
        # print('finished deleting mask {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        img_new = create_lp_by_mask1(img_new, cleaned_mask, False)
        # adding outputs folder
        img_label_output = region_dir+'outputs\\'+img_label_name+'.'+str(cell_size)+ 'output.png'
        segment_tools.saveImage(img_label_output, img_new)

    time_elapsed = time.time() - since
    print('end_time complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
since = time.time()

mask_dir = 'D:\\deep_learning_work\\meta_segment_all\\masks'

mask_list = os.listdir(mask_dir)
mask_list = [i for i in mask_list if '.png_filtered_masks.pkfile' in i]

test_flag = True

if test_flag == False:
    print('total filteres_masks to be processes:', len(mask_list))
    
    for filtered_mask in mask_list:
        img_name = filtered_mask.replace('.png_filtered_masks.pkfile','')
        print('Processing img:', img_name)
        process_image(img_name)
            
####################################################
#process single image
if test_flag == True:
    img_name='S15-25211-1-F.down-4'
    img_name = 'S10-3238.down-4'
    img_name='S19-13814-1-F.down-4'
    img_name = 'S13-01039-2-F.down-4'
    img_name = 'S10-1999-1.down-4'
    img_name = 'S10-1999-1-F.down-4'
    img_name = 'S10-33024-1-F.down-4'
    img_name = 'S10-33024-2-F.down-4'
    img_name = 'S11-6409-1.down-4'
    img_name = 'S11-6409-1-F.down-4'
    img_name = 'S11-8518-1-F.down-4'
    img_name = 'S12-09804-1.down-4'
    img_list = ['S10-1999-1-F.down-4', 'S10-27614-1-F.down-4']
    counter = 1
    for img_name in img_list:
        process_image(img_name)
        time_elapsed = time.time() - since
        print('image', counter,' complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        counter += 1
        
time_elapsed = time.time() - since
print('end_time complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))