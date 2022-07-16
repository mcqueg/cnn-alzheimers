import numpy as np
import cv2

def strip_skull(img):
    '''
    Purpose: given an image, the binary threshold is found using Otsu's method before computing
        the connected components in order to extract the brain. Potential holes in the mask are
        closed using a closing transformation, before returning the new image without the skull.
    Parameters: 
        -img: brain scan image with skull
    Returns:
        -brain_out: copy of the original image with the skull removed.
    '''
    # ensure the image is in grayscale
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #Threshold the image to binary using Otsu's method
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    ret, markers = cv2.connectedComponents(thresh)

    #Get the area taken by each component. Ignore label 0 since this is the background.
    marker_area = [np.sum(markers==m) for m in range(np.max(markers)) if m!=0] 
    #Get label of largest component by area
    largest_component = np.argmax(marker_area)+1 #Add 1 since we dropped zero above                        
    #Get pixels which correspond to the brain
    brain_mask = markers==largest_component

    # close the holes in the mask to retain the full brain image using a closing transformation
    brain_mask = np.uint8(brain_mask)
    kernel = np.ones((8,8),np.uint8)
    closing = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)

    brain_out = img.copy()
    #In a copy of the original image, clear those pixels that don't correspond to the brain
    # pixels that arent apart of the brain are set to 0 so they become black.
    brain_out[closing==False] = 0
    
    return brain_out