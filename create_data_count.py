# MT: modifications by Mengting
import sys
import numpy as np
import random
from model_settings import img_height, img_width, glimpses, min_edge, max_edge, min_blobs_train, max_blobs_train, min_blobs_test, max_blobs_test # MT 

def set_size(num_blobs, testing=False):
    """Get amount of images for each number"""
    if testing is False:
        return int(10000/(num_blobs**2)) # training: number distribution 1/(n^2)
    else:
        return 1000 # testing: make 1000 images for each number

def get_total(testing, min_blobs, max_blobs):
    """Get total amount of images"""
    c = min_blobs
    total = 0
    while (c < max_blobs + 1):
       total = total + set_size(c, even)
       c = c + 1
    return total

def get_dims(testing, min_edge, max_edge):
     """Get the dimensions of each blob"""
 
     if testing:
         width = 5
     else:
        width = random.randint(min_edge, max_edge)
 
     height = width # square blobs
 
     return width, height

def get_coordinate(testing):
     """Get the coordinates of the top left corner for each blob"""
 
     if testing:
         cX_ = random.randint(8, 9)
         cY_ = 3
     else:
         cX_ = random.randint(7, 10)
         cY_ = 10
 
     return cX_, cY_

def get_mask(num_blobs):
    mask = np.zeros(glimpses)
    for i in range (num_blobs+1):
        mask[i] = 1
    return mask

def generate_data(testing, min_blobs, max_blobs): # MT     
    n_labels = max_blobs_train - min_blobs_train + 1 # MT
    total = get_total(testing, min_blobs, max_blobs)
    imgs = np.zeros([total, img_height*img_width]) # input img size
    labels = np.zeros([total, n_labels])
    blob_list = np.zeros([total, glimpses, 2])
    size_list = np.zeros([total, glimpses])
    mask_list = np.zeros([total, glimpses])
    num_list = np.zeros(total)
    count_word = np.zeros([total, glimpses, n_labels+1])
    num_blobs = min_blobs
    img_count = 0

    while (num_blobs < max_blobs + 1):

        nOfItem = set_size(num_blobs, testing) # testing, 1000; training, 10000/(num_blobs**2)

        for i in range(nOfItem):
            img = np.ones(img_height*img_width) # input img size # MT: change zeros to ones
            num_count = 0
            glimpse_count = 0
            cX_prev = 0
            cY_prev = 0

            while num_count < num_blobs:
                width, height = get_dims(testing, min_edge, max_edge)
                cX_, cY_ = get_coordinate(testing)
                if num_count == 0:
                    cX = cX_
                else:
                    cX = cX_prev + cX_
                cY = random.randint(img_height/2 - cY_, img_height/2 + cY_ - height) # -10~10 / -3~3
                blob_list[img_count, glimpse_count, 0] = cX + width/2 # lower left corner
                blob_list[img_count, glimpse_count, 1] = cY + height/2
                size_list[img_count, glimpse_count] = height
                count_word[img_count, glimpse_count, num_count + 1] = 1 # "I'm done!":[1,0,0,0,0], "one":[0,1,0,0,0], "two":[0,0,1,0,0] 
                cX_prev = cX
                cY_prev = cY

                for p in range(cY, cY+height):
                    for q in range(cX, cX+width):
                        img[p*img_width+q] = 255

                num_count += 1
                glimpse_count += 1

            # I'm done!
            blob_list[img_count, glimpse_count:glimpses, 0] = cX_prev+width/2
            blob_list[img_count, glimpse_count:glimpses, 1] = cY_prev+height/2
            count_word[img_count, glimpse_count:glimpses, 0] = 1

            imgs[img_count] = img
            labels[img_count, num_blobs - 1] = 1
            mask_list[img_count] = get_mask(num_blobs)
            num_list[img_count] = num_blobs + 1
            img_count = img_count + 1
        num_blobs = num_blobs + 1

    np.set_printoptions(threshold=np.nan)

#    if empty_img:
#        train = np.vstack((train, np.zeros((1000, 10000))))
#        for empty_idx in range(1000):
#            empty_label = np.zeros(max_blobs + 1)
#            empty_label[0] = 1
#            label = np.vstack((label, empty_label))

    return imgs, labels, blob_list, size_list, mask_list, num_list, count_word

def generate_blank_img(testing, min_blobs, max_blobs): # MT     
    n_labels = max_blobs_train - min_blobs_train + 1 # MT
    total = get_total(testing, min_blobs, max_blobs)
    imgs = np.zeros([total, img_height*img_width]) # input img size
    labels = np.zeros([total, n_labels])
    blob_list = np.zeros([total, glimpses, 2])
    size_list = np.zeros([total, glimpses])
    mask_list = np.zeros([total, glimpses])
    num_list = np.zeros(total)
    count_word = np.zeros([total, glimpses, n_labels+1])
    num_blobs = min_blobs
    img_count = 0

    for i in range(total):
        img = np.ones(img_height*img_width) # input img size # MT: change zeros to ones
        glimpse_count = 0
        num_count = 0  
        while (glimpse_count < glimpses):
            if num_count < n_labels:
                count_word[img_count, glimpse_count, num_count + 1] = 1 # "one":[0,1,0,0,0], "two":[0,0,1,0,0] 
            else:
                count_word[img_count, glimpse_count, 0] = 1 # "I'm done!":[1,0,0,0,0]
                
            glimpse_count += 1
            num_count += 1
            
        imgs[img_count] = img
        labels[img_count, n_labels-1] = 1
        mask_list[img_count] = get_mask(n_labels)
        num_list[img_count] = n_labels
        img_count = img_count + 1

    np.set_printoptions(threshold=np.nan)

    return imgs, labels, blob_list, size_list, mask_list, num_list, count_word

