'''
File used to remove broken images in directory

Authors: Isak Bengtson, Mattias Wedin


'''

import os
import cv2
import pathlib
from PIL import Image

def check_images( s_dir):
    bad_images=[]
    bad_ext=[]
    s_list= os.listdir(s_dir)
    for klass in s_list:
        klass_path=os.path.join (s_dir, klass)
        print ('processing class directory ', klass)
        if os.path.isdir(klass_path):
            file_list=os.listdir(klass_path)
            counter = 0
            for f in file_list:
                counter+=1;               
                f_path=os.path.join (klass_path,f)
                if os.path.isfile(f_path):
                    # img = Image.open(f_path)
                    # img.save(f_path)
                    try:
                        img=cv2.imread(f_path)
                        shape=img.shape
                    except:
                        # print('file ', f_path, 'is not a valid image file')
                        os.remove(f_path)
                        print('removed ', f_path)
                        bad_images.append(f_path)
                    # If dataset needs to be smaller
                    # if(counter > 999):
                    #     os.remove(f_path)


source_dir = pathlib.Path('../data/images/mouse')

good_exts=['jpg', 'png', 'jpeg', 'gif', 'bmp' ] # list of acceptable extensions
check_images(source_dir, good_exts)
