import numpy as np
import cv2
from PIL import Image

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
    img = strip_skull(img_path)
    # create a 3channel grayscale image
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