Segment_tools.py file contains all tool functions, and it is needed by all other functions.

Step 1. Run detect_nucleus_size.py to calculate all image's nucleus mean size
Step 2. Run Segment_Fat_Droplet.py file to do meta segamentation process. All masks will be generated and stored under path 'data\mask_512\'
Step 3. Run ValidateMasks.py file to select the fat droplets and validate them against the manual selected droplets stored in 'image_*.rois.zip' file
        The validated results of the image will be saved in image folder, the data report will be saved in results.csv file.
