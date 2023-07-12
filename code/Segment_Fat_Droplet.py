# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 16:04:13 2023

@author: sunnf
"""
import cv2
import time
import os
import torch
import numpy as np
#import image_slicer
from segment_anything import build_sam, SamAutomaticMaskGenerator
# import segment_tools
from amg import write_masks_to_folder
import json

def getMeanWithoutBlack(im):
    total = 0
    counter = 0
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i,j,0] > 0:
                total += im[i,j,0]
                counter+=1
    return total/counter
        
def getEllipseMask(w,h):
    ellipse_mask = np.zeros([w,h],dtype=np.uint8)
    center_coordinates = (int(h/2),int(w/2))
    axesLength = (int(h/2),int(w/2))
    angle = 0
    startAngle = 0
    endAngle = 360
    # white color mask
    color = (255)
    # Line thickness of -1 px
    thickness = -1
    # Using cv2.ellipse() method
    # Draw a ellipse with blue line borders of thickness of -1 px
    ellipse_mask = cv2.ellipse(ellipse_mask, center_coordinates, axesLength, angle,
                              startAngle, endAngle, color, thickness)
    return ellipse_mask

def update_xy_global(masks, r, c):
    for mask in masks:
        x = int(mask['bbox'][0])
        y = int(mask['bbox'][1])
        #Update mask with patch info
        mask['bbox'][0] = int(c+x)
        mask['bbox'][1] = int(r+y)
        #remove too small and too large fat droplet
        # if w*h > 5000:
        #     continue
        
        
        # # #remove low quality iou
        # if mask['predicted_iou'] <= 0.97:
        #     continue
        # #get mask_img
        # mask_img = image[y:y+h,x:x+w]
        # mean = -1
        
        # ellipse_mask = getEllipseMask(h,w)
        # ellipse_img = cv2.bitwise_and(mask_img,mask_img,mask = ellipse_mask)
        # mean = getMeanWithoutBlack(ellipse_img)
        
        # #remove cells that are not round w:h 1:2
        # if w/h>2 or h/w>2:
        #     #masks.remove(mask)
        #     continue
        # #remove mask with mean brightness > 250
        # if np.mean(mask_img) >= 250: 
        #     #masks.remove(mask)
        #     continue
        # #for single cell, use ellipse mask
        
        # #remove mask with mean brightness <= 150
        # if mean <= 239:#241, 
        #     continue
        
        # cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),1)
        # # font
        # font = cv2.FONT_HERSHEY_SIMPLEX          
        # # org
        # org = (x, y)          
        # # fontScale
        # fontScale = 0.3         
        # # Blue color in BGR
        # color = (255, 0, 0)          
        # # Line thickness of 1-x
        # thickness = 1           
        # # Using cv2.putText() method
        # image = cv2.putText(image, str(counter)+'-'+str(mask['predicted_iou'])[:4]+'_'+str(mean)[:3], org, font, fontScale, color, thickness, cv2.LINE_AA)
        # filtered_mask.append(mask)


since = time.time()
now = time.localtime()
current_time = time.strftime("%H:%M:%S", now)
print(current_time)
# img_dir = 'C:\\Users\\sunnf\\Downloads\\test.png'
# image = cv2.imread(img_dir)
patch_size = 512
target_dir = 'tiles'
img_dir = '..\\data\\images'
mask_dir = '..\\data\\masks_'+str(patch_size)
imagenames = os.listdir(img_dir)
png_imgnames = [i for i in imagenames if '-4.png' in i]
#build sam model
sam_model = build_sam(checkpoint="..\\sam_vit_h_4b8939.pth") #make sure here you refer to the meta SAM model path
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#sam_model.to(device)
mask_generator = SamAutomaticMaskGenerator(sam_model)
# mask_generator = SamAutomaticMaskGenerator(sam_model, output_mode=output_mode, **amg_kwargs)

counter = 0
padding = 30
for imagename in png_imgnames:
    masks = []
    print('processing image:',counter, img_dir+'\\'+imagename)
    image = cv2.imread(img_dir+'\\'+imagename)
    r_ct=0
    img_mask = []
    img_filtered_mask=[]    
    for r in range(0,image.shape[0],patch_size - padding):
        c_ct = 0
        for c in range(0,image.shape[1],patch_size - padding):#step = 512 - 30 = 482         
            r_patch = min(r+patch_size,image.shape[0]-1)
            c_patch = min(c+patch_size,image.shape[1]-1)
            patch = image[r:r_patch, c:c_patch,:]
            brightness = np.mean(patch)
            if np.mean(patch) >= 240:#241
                continue
            #print(np.mean(patch), end='')
            #saveImage(target_dir+'\\'+imagename +'_'+ str(r_ct)+'_'+str(c_ct)+'_'+str(brightness)+'.png',patch)
            masks = mask_generator.generate(patch)            
            #remove large masks
            update_xy_global(masks, r, c)
            img_mask.extend(masks)
            
            # img_filtered_mask.extend(filtered_masks)
            # segment_tools.saveImage(target_dir+'\\'+imagename +'_'+ str(r_ct)+'_'+str(c_ct)+'.png',patch)
            # segment_tools.save_masks_pickle(img_mask, mask_dir+'\\'+imagename+'_masks.pkfile')
            # segment_tools.save_masks_pickle(img_filtered_mask, mask_dir+'\\'+imagename+'_filtered_masks.pkfile')  
            time_elapsed = time.time() - since
            print('patch_'+str(r_ct)+'_'+str(c_ct)+' complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            c_ct+=1
            #torch.cuda.empty_cache()
        r_ct+=1
    counter+=1
    time_elapsed = time.time() - since
    print('image '+str(counter)+' complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    #save masks to image folder
    mask_dir_folder = mask_dir + '\\' + imagename[:-4] +'\\'
    print(mask_dir_folder)
    os.makedirs(mask_dir_folder, exist_ok=True) 
    write_masks_to_folder(img_mask, mask_dir_folder)

time_elapsed = time.time() - since
print('end_time complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

# print(image.shape)

# masks = ''
# with open('masks.file', 'rb') as masks_file: 
#     masks = pickle.load(masks_file)

# if masks =='':
#     mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint="sam_vit_h_4b8939.pth"))
#     masks = mask_generator.generate(image)

# # Step 2
# with open('masks.file', 'wb') as masks_file:
#   pickle.dump(masks, masks_file)

# targetImageName = 'D:\\deep_learning_work\\meta_segment_all\\' + 'testmask.png'

# print(len(masks))
# image = draw_bbox(masks, image)
# saveImage(targetImageName, image)
