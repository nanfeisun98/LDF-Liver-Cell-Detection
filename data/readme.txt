Data Folder structure
1. images folder stores original images, ground truth , and results
   Original images include two files, the liver image and the liver label image
   it also contains a zip file which stores manually selected ROI (fat droplets)
   The validated results are also stored here.
2. The mask_256, 512, 1024, are mask folders that are used to save segment all masks. 256 is for patch size 256
   mask_512 is for patch size 512. Each image contains thousands of small masks and were stored in the image_*.down-4 folder accordingly.
3. the region folder stores all black and white image mask for liver itself
