# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 10:24:46 2023

@author: sunnf
"""
import cv2
import numpy as np
import pickle
from skimage import morphology
import math
import pandas as pd

patch_size = 512
patch_size_margin = patch_size - 30
cell_img_folder='D:\\Images\\cells'
mask_img_folder='D:\\Images\\maskImgs'

def get_median_area(masks):
    area_list = []
    for mask in masks:
        area_list.append(mask['area'])
        
    area_list.sort()
    min_area = area_list[0]
    max_area = area_list[-1]
    median_area = area_list[int(len(area_list)/2)]
    
    return median_area, min_area, max_area
    

def get_xy_size(bbox):
    y = int(bbox[0])
    x = int(bbox[1])
    h = int(bbox[2])
    w = int(bbox[3])
    cen_x = x+int(w/2)
    cen_y = y+int(h/2)
    size = max(w,h)
    return cen_x, cen_y,size

def find_img_bk_name(img_name, img_bk_list):
    for img_bk_name in img_bk_list:
        # print(img_bk_name,img_name[:-15] )
        if img_bk_name.find(img_name[:-15]) > -1:
            # print(img_bk_name, img_name[:-4])
            return img_bk_name
    return ''

#dist_scale default = 3, small cell size larger dist
def delete_lonely_mask(masks, dist_scale):
    mask_new = []
    neighbor_required = 1
    # print('before delete lonely mask', len(masks))
    for mask in masks:
        cen_x, cen_y,size=get_xy_size(mask['bbox'])
        size_scale = 15/size #(2*min_size)
        if size_scale<1:
            size_scale = 1
            
        neighbor_count = -1
        for mask1 in masks:
            size_scale1 = 15/size #(2*min_size)
            if size_scale1<1:
                size_scale1 = 1
                
            cen_x1, cen_y1, size1=get_xy_size(mask1['bbox'])                       
            #check if these two masks overlap
            dist_x = (cen_x-cen_x1)
            dist_y = (cen_y-cen_y1)
            dist = math.sqrt(dist_x*dist_x + dist_y*dist_y)
            if dist < dist_scale * size * size_scale or dist < dist_scale * size1 * size_scale1:
                neighbor_count+=1
                if neighbor_count>=neighbor_required:
                    #we should keep this mask
                    mask_new.append(mask)
                    break
    #end for loop
    # print('after delete lonely mask', len(mask_new))
    return mask_new

def remove_small_mask(mask):
    binarized = np.where(mask>1, 1, 0)
    processed = morphology.remove_small_objects(binarized.astype(bool), min_size=200000, connectivity=2).astype(int)
    # black out pixels
    mask_x, mask_y = np.where(processed == 0)
    mask[mask_x, mask_y] = 0

def get_BW_BKGD(org_img_gray):
    
    (thresh, im_bw) = cv2.threshold(org_img_gray, 230, 255, cv2.THRESH_BINARY) #235, 240    
    remove_small_mask(im_bw)
    im_bw = cv2.bitwise_not(im_bw)
    remove_small_mask(im_bw)
    mask = cv2.inRange(im_bw, 0, 1)
    return im_bw, mask

def blob_properties(cnt):
    result={}
    result['area']= cv2.contourArea(cnt)
    if result['area'] == 0:
        return result
    result['perimeter'] = cv2.arcLength(cnt,True)
    result['convexity'] = cv2.isContourConvex(cnt)   
    result['circularity'] =4*3.1416*result['area']/(result['perimeter']*result['perimeter'])
    # result['bbox'] = cv2.boundingRect(cnt) #x,y,w,h
    # w=result['bbox'][2]
    # h=result['bbox'][3]
    bbox = cv2.boundingRect(cnt)
    w=bbox[2]
    h=bbox[3]
    result['aspect_ratio'] = float(w)/h
    result['rect_area'] = w*h
    result['extent'] = float(result['area'])/result['rect_area']
    #result['hull'] = cv2.convexHull(cnt)
    #result['hull_area'] = cv2.contourArea(result['hull'] )
    hull = cv2.convexHull(cnt)
    result['hull_area'] = cv2.contourArea(hull)
    
    if result['hull_area'] == 0:
        result['solidity'] = 0
    else:
        result['solidity'] = float(result['area'])/result['hull_area']
    # (xa,ya),(MA,ma),angle = cv2.fitEllipse(cnt)
    # rect = cv2.minAreaRect(cnt)
    # (xc,yc),radius = cv2.minEnclosingCircle(cnt)
    # ellipse = cv2.fitEllipse(cnt)
    # [vx,vy,xf,yf] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
    # lefty = int((-xf*vy/vx) + yf)
    # righty = int(((cols-xf)*vy/vx)+yf)
    # # Add parameters to list
    # add = i+1, area, round(perimeter, 1), convexity, round(aspect_ratio, 3), round(extent, 3), w, h, round(hull_area, 1), round(angle, 1), x1, y1, x2, y2,round(radius, 6), xa, ya, xc, yc, xf[0], yf[0], rect, ellipse, vx[0], vy[0], lefty, righty
    # cont_props.append(add)
    return result

def load_maskfolder(mask_foldername):
    masks = []
    #load meta data
    df = pd.read_csv(mask_foldername + '\\metadata.csv')
    for row in df.itertuples():
        mask = {}
        mask['id'] = row[1]
        mask['img_name'] = mask_foldername + '\\' + str(row[1]) + '.png'
        #print('*****', mask['img_name'])
        mask['area'] = row[2]
        mask['bbox'] = [row[3],row[4],row[5],row[6]]
        # if '\\3.png' in mask['img_name']:
        #     print(row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10])
        #     print('bbox', mask['bbox'])
        mask['point_coords'] = [row[7], row[8]]
        mask['predicted_iou'] = row[9]
        mask['stability_score'] = row[10]
        masks.append(mask)
        
    return masks
    
def save_masks_pickle(img_mask,img_pkfile_name):
    print('save mask pkfile',img_pkfile_name)
    with open(img_pkfile_name, 'wb') as masks_file: 
        pickle.dump(img_mask, masks_file)

def saveImage(imgName, image):
    try:
        writeStatus = cv2.imwrite(imgName,image)
    except:
        print('image writting error!',writeStatus)
    
    print('write image to', imgName)

def saveImageNoPrint(imgName, image):
    try:
        writeStatus = cv2.imwrite(imgName,image)
    except:
        print('image writting error!', writeStatus)

#return the mean, std, range of the pixels
def getMeanWithoutBlack(im):
    number_list=[]
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i,j,0] > 0:
                #number_list.append((im[i,j,0]+im[i,j,1]+im[i,j,2])/3)
                number_list.append(im[i,j,0])
    if len(number_list) == 0:
        return 0,0,0
    return np.average(number_list), np.std(number_list),np.max(number_list) - np.min(number_list)

#return the mean, std, range of the pixels
def getMeanWithoutBlack_gray_img(im):    
    total = np.sum(im[0:im.shape[0], 0:im.shape[1]])
    number_of_black_pix = np.sum(im == 0)
    print('total', total, 'black_px', number_of_black_pix,im.shape[0] * im.shape[1], im.shape[0] * im.shape[1] - number_of_black_pix )
    return total/(im.shape[0] * im.shape[1] - number_of_black_pix )
    
        
def getEllipseMask(w,h,margin):
    ellipse_mask = np.zeros([w,h],dtype=np.uint8)
    center_coordinates = (int(h/2),int(w/2))
    axesLength = (int(h*margin/2),int(w*margin/2))
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

def has_j_point(mask, j_points, pp_ap_points):
    #all points in j_points are ap
    bbox_x = mask['bbox'][0]
    bbox_y = mask['bbox'][1]
    w = mask['bbox'][2]
    h = mask['bbox'][3]
    for p in j_points:
        p_x = p.pt[0]
        p_y = p.pt[1] 
        #if we found a j_point inside the bbox of the mask, return True           
        if p_x >= bbox_x and p_x <= bbox_x+w and p_y>=bbox_y and p_y<=bbox_y+h:
            #check if the point is already in pp_ap_points
            if p not in pp_ap_points:
                pp_ap_points.append(p)
            # else:
            #     print('find same j_point in different masks')
            return True
    #if iterate through all j_points and cannot find any
    return False

def regression_score(blob_dict):
    # score = -5.231238455 + 0.006251244 * blob_dict['area'] + 0.02629545*blob_dict['perimeter'] +2.249189741*blob_dict['circularity'] -0.136671419 * blob_dict['aspect_ratio']
    # score +=(1.149287912*blob_dict['extent']-0.008010684*blob_dict['hull_area']-1.842632614*blob_dict['solidity']+0.008744711*blob_dict['bd_std']-0.002181447*blob_dict['bd_range'])
    # score +=(0.015226709*blob_dict['mean_diff'] + 3.850838312*blob_dict['predicted_iou'])    
    # return score
    
    # score = -1.040751781 + 0.001476008 * blob_dict['area'] + 0.002025411*blob_dict['perimeter'] +0.710659651*blob_dict['circularity'] - 2.1617E-05 * blob_dict['rect_area']
    # score +=(-0.001127508*blob_dict['hull_area']-0.655296838*blob_dict['solidity']-0.000883284*blob_dict['mean']-0.003994095*blob_dict['std'])
    # score +=(-0.000258793*blob_dict['range']+ 0.001254453*blob_dict['bd_range']+ 0.00084624*blob_dict['mean_diff'] + 1.448438877*blob_dict['predicted_iou'])    
    # return score

    score = 0.38422314 + 0.000981401 * blob_dict['area'] + 0.002120084*blob_dict['perimeter'] +0.804879626*blob_dict['circularity'] - 7.30671E-05 * blob_dict['rect_area']
    score += 0.24782374*blob_dict['extent']
    score +=(-0.000943833*blob_dict['hull_area']-0.787434677*blob_dict['solidity']-0.000780783*blob_dict['mean']-0.007596211*blob_dict['std'])
    score +=(0.000167532*blob_dict['range']-0.000565159*blob_dict['bd_mean'] + 0.001347155*blob_dict['bd_range'])
    
    return score

def is_blob(solidity, extent, circularity, blob_filter, w, h, min_size, is_small_blob):
    
    if blob_filter['extent_cir_sum']>0 and extent + circularity < blob_filter['extent_cir_sum'] :
        return False
   
    # if  blob_filter['extent'] > 0 and float(extent) < blob_filter['extent']:# or float(blob_dict['solidity']) > 0.97:
    #     return False
    
    if blob_filter['solidity']>0 and float(solidity) < blob_filter['solidity']:
        return False
    # elif float(solidity) < 0.75 and (w > min_size*3 or h > min_size*3):
    #     return False
    
    # if blob_filter['circularity'] > 0 and circularity < blob_filter['circularity']:# or float(blob_dict['solidity']) > 0.97:
    #     return False
    # # large blob needs higher circularity, remove FP
    # if is_small_blob != True and circularity_lmt < circularity_lmt:
    #     return False
    
    return True

    # if float(blob_dict['circularity']) < 0.33:# or float(blob_dict['solidity']) > 0.97:
    #     return False
    # if thresh < 200:
    #     return False

def check_mean(mean, bd_mean,inner_mean,min_size, mean_diff_flag, w, h):
    #if border and center image has similiar mean values, that means it is white background
    #print('mean',mean,'bd_mean',bd_mean)
    if mean_diff_flag == True: #this flag will be turned false when the droplets are tightly together
        if (mean - bd_mean) < 6: # all mean_diff < 5 mustbe skipped
            return False
        elif (mean - bd_mean) < 8 and (w * h > 600): #for larger blob increase the mean_diff requirement to 7
            return False
        # elif (mean - bd_mean) < 12 and (w * h > 900 ): #for larger blob increase the mean_diff requirement to 7
        #     return False
    #remove cells that are not round w:h 1:2
    if w/h>2 or h/w>2:
        return False
    # elif w*h>650 and (w/h>2 or h/w>2):
    #     return False
    # elif w*h>900 and (w/h>1.5 or h/w>1.5):
    #     return False
    #remove mask with mean brightness > 250
    # if np.mean(mask_img) >= 250: 
    #     return False

    #remove mask with mean brightness <= inner mean
    if inner_mean> 0 and mean<=inner_mean:
        return False
    
    return True



def get_mask_cnt(mask,x,y,w,h, patch_size):
    #convert the mask to a binary image, using original mask
    # mask_bw = np.array(mask['segmentation'] * 255, dtype = np.uint8)
    #load the mask from png file, using new mask folder
    mask_name = mask['img_name']
    #print('*****',mask_name)
    mask_gray = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
    thresh = 127
    mask_bw = cv2.threshold(mask_gray, thresh, 255, cv2.THRESH_BINARY)[1]
    
    #print('****',y,y+h,x,x+w)
    if mask['point_coords'][0] < 400 or x % (patch_size - 30) + patch_size_margin + w > patch_size:
        x = x % (patch_size - 30)  #patch size, padding = 30
    else:
        x = x % (patch_size - 30) + patch_size_margin
        
    if mask['point_coords'][1] < 400 or y % (patch_size - 30) + patch_size_margin + h > patch_size:
        y = y % (patch_size - 30)
    else:
        y = y % (patch_size - 30) + patch_size_margin
        
    small_mask_bw = mask_bw[y:y+h, x:x+w]
    # print(y,y+h,x,x+w)
    # saveImage('small_mask_bw_'+str(mask['id'])+'.png', small_mask_bw)
    
    #find the contours of the mask
    contours, hierarchy = cv2.findContours(small_mask_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(contours)
    if contours is None or len(contours) == 0:
        return None, None, None
    
    return max(contours, key = cv2.contourArea),contours, small_mask_bw

def get_blob_cnt(mask_img):
    #do blob attributes
    mask_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    (thresh,mask_bw) = cv2.threshold(mask_gray,128,255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    # (thresh,mask_bw) = cv2.threshold(mask_gray,230,255, cv2.THRESH_BINARY)
    contours = cv2.findContours(mask_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    if contours is None:
        return None, None, None
    
    return max(contours, key = cv2.contourArea),contours, mask_bw

def get_contour_img(mask_img, contour_mask):
    border_mask = cv2.bitwise_not(contour_mask)
        
    cnt_img = cv2.bitwise_and(mask_img,mask_img,mask = contour_mask)
    border_img = cv2.bitwise_and(mask_img,mask_img,mask = border_mask)
    mean, std, rge = getMeanWithoutBlack(cnt_img)
    bd_mean,bd_std, bd_rge = getMeanWithoutBlack(border_img)
    return mean, std, rge, bd_mean,bd_std, bd_rge
   
def get_ellipse_img(h,w,mask_img):
    ellipse_mask = getEllipseMask(h,w,1)
    border_mask = cv2.bitwise_not(ellipse_mask)
    #if the cell size > 30, we only take the middle 90%
    # if h>30 or w > 30:
    ellipse_mask = getEllipseMask(h,w,0.9)
        
    ellipse_img = cv2.bitwise_and(mask_img,mask_img,mask = ellipse_mask)
    border_img = cv2.bitwise_and(mask_img,mask_img,mask = border_mask)
    mean, std, rge = getMeanWithoutBlack(ellipse_img)
    bd_mean,bd_std, bd_rge = getMeanWithoutBlack(border_img)
    return mean, std, rge, bd_mean,bd_std, bd_rge

#special treatment for small droplet that is < 1.2 times min_size
def is_small_droplet(w, h, mask, blob_dict, min_size):
    return False
    # if w > 3*min_size or h > 3*min_size:
    #     return False
    # if w/h > 1.5 or h/w > 1.5:
    #     return False
    # if blob_dict['circularity'] <0.8:
    #     return False
    
    # return True
    
#special treatment for medium to large droplet that contains dark dots inside
def is_sinoid():
    
    return True
    

def draw_bbox(patch_size, masks,j_points, image, img_name, r, c, filter_flag,median_area, liver_size,min_scale,mean_diff_flag,blob_filter, predicted_iou, inner_mean, save_cell, useRegression):     
    print('mask size:',len(masks))
    print(img_name)
    image_copy = image.copy()
    counter = 0
    filtered_mask = []
    mask_blobs=[]
    scale = (4.5/liver_size) * (4.5/liver_size)
    # min_size = liver_size * 3 * scale * min_scale#4.6 - 5.8 * 2
    # min_size = liver_size * 3.4  * min_scale #s1999-1.png
    min_size = math.sqrt(median_area)* min_scale
    max_size = (liver_size * 12)   #55.2 - 69.6 = 3047.04 - 4844.16
    print('min_size',min_size,'max_size',max_size)
    
    for mask in masks:        
        x = int(mask['bbox'][0])
        y = int(mask['bbox'][1])
        w = int(mask['bbox'][2])
        h = int(mask['bbox'][3])
        
        #Update mask with patch info
        mask['bbox'][0] = int(c+x)
        mask['bbox'][1] = int(r+y)
        
        # if counter ==0:
        #     print('mask coordinate:',x,y,w,h,int(c+x),int(r+y))
        
        is_j_point = False
        if has_j_point(mask, j_points, []):
            is_j_point = True
        
        #remove small area by w and h
        if w<min_size or h<min_size:
            continue
        # remove too small and too large fat droplet
        if w>max_size or h>max_size:
            continue
            
        #get mask_img
        if y%patch_size_margin+h > patch_size or x%patch_size_margin+w > patch_size:
            # print('border > ', patch_size)
            continue
        mask_img = image[y:y+h,x:x+w]
        mean = -1
        bd_mean = -1
        blob_max_cnt, blob_contours, mask_bw = get_blob_cnt(mask_img)
        max_cnt, contours, cnt_mask = get_mask_cnt(mask,x,y,w,h, patch_size)
        if max_cnt is None:
            print('max_cnt is none')
            continue
        
        #mean, std, rge, bd_mean,bd_std, bd_rge = get_ellipse_img(h,w,mask_img)
        mean, std, rge, bd_mean,bd_std, bd_rge = get_contour_img(mask_img, cnt_mask)
        
        
        if mean == 0 or std ==0 or rge ==0 or bd_mean==0 or bd_std==0 or bd_rge ==0:
            continue
        
        #print(x,y,w,h)
        
        blob_dict = blob_properties(max_cnt)
        blob_dict['mean'] = mean
        blob_dict['std'] = std
        blob_dict['range'] = rge
        blob_dict['bd_mean'] = bd_mean
        blob_dict['bd_std'] = bd_std
        blob_dict['bd_range'] = bd_rge
        blob_dict['mean_diff'] = mean - bd_mean
        if is_j_point:
            blob_dict['is_j_point'] = 1
        else:
            blob_dict['is_j_point'] = 0
        
        blob_dict['predicted_iou'] = mask['predicted_iou']
        blob_dict['id'] = counter
        blob_dict['image_name'] = img_name
        
        #if std ==0, that is blank field
        if blob_dict['area']==0 or blob_dict['std'] == 0 or blob_dict['range'] == 0 or blob_dict['bd_std'] == 0 or blob_dict['bd_range'] == 0:
            continue
        
        #remove low quality iou
        #small droplet only need iou > 0.9
        is_small_blob = is_small_droplet(w, h, mask, blob_dict, min_size)
        #check mean values
        if filter_flag['check_mean']==True and check_mean(mean, bd_mean, inner_mean, min_size, mean_diff_flag, w, h) == False:
            continue
        #check iou
        if filter_flag['check_iou']==True and mask['predicted_iou'] < predicted_iou:
            continue
        #check contour diff
        cnt_diff = cv2.matchShapes(max_cnt,blob_max_cnt,1,0.0)
        if filter_flag['check_cnt']==True and cnt_diff > blob_filter['cnt_diff']:
            continue
        
        # if use_filter==True and is_small_blob == True and mask['predicted_iou'] <  0.9:
        #     continue
        # #large droplet got iou from external
        # elif use_filter==True and  is_small_blob == False and mask['predicted_iou'] < predicted_iou:
        #     continue
        # elif mask['predicted_iou'] > 0.90 and mask['predicted_iou'] <= 0.96:
        #     if is_j_point == True:
        #         print('min_size * 2',min_size * 2 ,w, h, w/h, h/w,'circularity', blob_dict['circularity'])
           
        
        # r_score = regression_score(blob_dict)
        # if useRegression > 0 and r_score<useRegression:
        #     continue
        
        #check blob features
        if filter_flag['check_blob']==True and is_blob(blob_dict['solidity'], blob_dict['extent'], blob_dict['circularity'],blob_filter, w, h, min_size, is_small_blob) == False:
            continue

        if save_cell== True:
            # saveImageNoPrint(cell_img_folder+'\\'+ img_name +'_elpimg_'+str(counter)+'_'+str(int(mean))+'_'+str(int(std))+'_'+str(int(rge))+'.png', ellipse_img)
            # saveImageNoPrint(cell_img_folder+'\\'+ img_name +'_elpimg_'+str(counter)+'a_'+str(int(bd_mean))+'_'+str(int(bd_std))+'_'+str(int(bd_rge))+'.png', border_img)
            mask_img_name = mask_img_folder
            mask_img_copy = mask_img.copy()
            cv2.drawContours(mask_img_copy, contours, -1, (128,0,0),1) 
            cv2.drawContours(mask_img_copy, blob_contours, -1, (0,0,255),1)
            # cv2.drawContours(mask_bw, contours, -1, (0,0,255),1)
            if is_j_point:
                # cv2.circle(mask_img,(int(x+w/2), int(y+h/2)), 5, (0,0,0), -1) #not working, x,y are whole image coordinator
                mask_img_name+='\\true'
            else:
                mask_img_name+='\\false'
            
            mask_img_name+= '\\'+ img_name+'.'
            mask_img_name+= str(counter)
            mask_img_name+='_'+ '{:4.3f}'.format(blob_dict['predicted_iou'])
            mask_img_name+='_'+ '{:4.3f}'.format(blob_dict['extent'])
            mask_img_name+='_'+ '{:4.3f}'.format(blob_dict['solidity'])
            mask_img_name+='_'+ '{:4.3f}'.format(blob_dict['circularity'])
            mask_img_name+='_'+ '{:4.0f}'.format(mean)
            mask_img_name+='_'+ '{:4.0f}'.format(bd_mean)
            mask_img_name+='_'+ '{:4.3f}'.format(cnt_diff)            
            mask_img_name+='_'+ str(w) + 'x' +str(h)
            # mask_img_name+='_'+ '{:4.0f}'.format(thresh)      
          
            # if blob_dict['extent'] < 0.7:
            saveImage( mask_img_name +'.png', mask_img_copy)
            # saveImageNoPrint( mask_img_name +'_bw.png', mask_bw)

        cv2.rectangle(image_copy,(x,y),(x+w,y+h),(0,255,0),1)
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX          
        # org
        org = (x, y)          
        # fontScale
        fontScale = 0.3         
        # Blue color in BGR
        color = (255, 0, 0)          
        # Line thickness of 1-x
        thickness = 1           
        # Using cv2.putText() method
        #image_copy = cv2.putText(image_copy, str(counter)+'-'+str(mask['predicted_iou'])[:4]+'_'+str(mean)[:3], org, font, fontScale, color, thickness, cv2.LINE_AA)
        # image_copy = cv2.putText(image_copy, str(counter)+'.'+str(bd_mean)[:3]+'.'+str(mean)[:3], org, font, fontScale, color, thickness, cv2.LINE_AA)
        image_copy = cv2.putText(image_copy,  str(mask['predicted_iou'])[:4]+'.'+ str(bd_mean)[:3]+'.'+str(mean)[:3]+'.'+str(w), org, font, fontScale, color, thickness, cv2.LINE_AA)
        # image_copy = cv2.putText(image_copy, str(blob_dict['extent'])[:4]+'.'+str(blob_dict['solidity'])[:4]+'.'+str(blob_dict['circularity'])[:4]+'.'+str(w), org, font, fontScale, color, thickness, cv2.LINE_AA)
        # image_copy = cv2.putText(image_copy, str(counter), org, font, fontScale, color, thickness, cv2.LINE_AA)
        filtered_mask.append(mask)
        counter+=1
        mask_blobs.append(blob_dict)
    return image_copy, filtered_mask, mask_blobs

#when one box is inside of another box, delete the smaller one and keep the bigger one
# def merge_bbox(masks):
#     counter = 0
#     for mask in masks[:]:
#         x = int(mask['bbox'][0])
#         y = int(mask['bbox'][1])
#         w = int(mask['bbox'][2])
#         h = int(mask['bbox'][3])
#         print('mask',counter)
#         print('segmentation',mask['segmentation'])
#         print('bbox',mask['bbox'])
#         print('predicted_iou',mask['predicted_iou'])
#         print('area',mask['area'])
#         print('stability_score',mask['stability_score'])
#         counter+=1