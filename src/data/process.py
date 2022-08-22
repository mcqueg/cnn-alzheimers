import numpy as np
import cv2
import os
import time


def process_img(img_path):
    '''
    Purpose: process one image when it is called during data generation. Strips the skull
                from the image and then expands the image to have 3 channels due to the input 
                needs of the models.
    Parameters:
            img - path to grayscale image (1 channel) to process 
    Returns:
            processed_img - image processed, ready to be used for training.
    '''
    # strip the skull from the image
    #img = strip_skull(img_path)
    # create a 3channel grayscale image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    processed_img = stack_image(img)

    return processed_img

#------------------------------------------------------------------------------------------

def strip_skull(img_path):
    '''
    Purpose: given an image, the binary threshold is found using Otsu's method before computing
        the connected components in order to extract the brain. Potential holes in the mask are
        closed using a closing transformation, before returning the new image without the skull.
    Parameters: 
        -img: brain scan image with skull, one channel
    Returns:
        -brain_out: copy of the original image with the skull removed.
    '''
    # ensure the image is in grayscale
    #img = cv2.imread(img_path)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, img.shape, 0)
    #Threshold the image to binary using Otsu's method
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    ret, markers = cv2.connectedComponents(thresh)

    #Get the area taken by each component. Ignore label 0 since this is the background.
    marker_area = [np.sum(markers==m) for m in range(np.max(markers)) if m!=0] 
    #marker_area = [np.sum(markers==m) for m in range(np.max(markers))] 
    #Get label of largest component by area
    
    try:
        largest_component = np.argmax(marker_area)+1 #Add 1 since we dropped zero above                        
        #Get pixels which correspond to the brain
        brain_mask = markers==largest_component
    except ValueError:
        return img.copy()

    # close the holes in the mask to retain the full brain image using a closing transformation
    brain_mask = np.uint8(brain_mask)
    kernel = np.ones((8,8),np.uint8)
    closing = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)

    brain_out = img.copy()
    #In a copy of the original image, clear those pixels that don't correspond to the brain
    # pixels that arent apart of the brain are set to 0 so they become black.
    brain_out[closing==False] = 0
    
    return brain_out

#------------------------------------------------------------------------------------------

def stack_image(img):
    '''
    Purpose:
        Turn a single channel image into a three channel image. Stacks the image on top
        of itself to achieve the three dimensions. 
    Parameters:
        img - greyscale image to expand to three channels
    Returns:
        img_stack - img with 3 channels
    '''
    img_stack = np.stack((img,)*3, axis=-1)

    return img_stack

#------------------------------------------------------------------------------------------
    # loop through class names generating the path to folder of that class

def process_dir(IN_DIR, OUT_DIR):
    # get class dir names
    CLASSES = os.listdir(IN_DIR)
    # create directories for them based on class
    for dir in CLASSES:
        os.makedirs(os.path.join(OUT_DIR, dir))
    
    #files = 0

    #for dir in os.listdir(IN_DIR):
    #    dir_num = len([f for f in os.listdir(os.path.join(IN_DIR, dir)) if not f.startswith('.')])
    #    #files+=len(os.listdir(os.path.join(IN_DIR, dir)))
    #    files+=dir_num
    #print(f'{files} found to process')


    num_processed = 1
    for c in CLASSES:
        out_class_dir = os.path.join(OUT_DIR, c)
        class_dir = os.path.join(IN_DIR, c)
        #loop through each image in the current class directory
        for image in os.listdir(class_dir):
            img_path = os.path.join(class_dir, image)
            #ignore image if it is of size = 0
            if os.path.getsize(img_path) == 0:
                print(f'{img_path} is zero length so ignoring.')
            else:
                processed_img = process_img(img_path) #process image from ROOT/src/data/process.py
                print(f'\r{num_processed} images processed', end='')
                time.sleep(.05)
                #create path to out image
                out_path = os.path.join(out_class_dir, image)
                # write the processed image into the processed directory within its class folder
                cv2.imwrite(out_path,processed_img)
                num_processed += 1
    print()

if __name__== "__main__":

    ROOT = '/Users/garrettmccue/projects/cnn-alzheimers/'
    data = f'{ROOT}/data/'

    os.makedirs(f'{data}/processed/ALZ/train')
    os.makedirs(f'{data}/processed/ALZ/test')

    # path to save images to
    TRAIN_OUT_DIR = f'{data}/processed/ALZ/train'
    # path to raw data root folder 
    TRAIN_IN_DIR = f'{data}/raw/ALZ/train'

    TEST_OUT_DIR = f'{data}/processed/ALZ/test'
    TEST_IN_DIR = f'{data}/raw/ALZ/test'
    # process the training images
    process_dir(TRAIN_IN_DIR, TRAIN_OUT_DIR)
    # process the test images
    process_dir(TEST_IN_DIR, TEST_OUT_DIR)