
"""This file makes use of Rasterio lib to import and display raster snippets. Then the label files are converted to image format
and are stored in their respective save folders"""

#  Impoerting required libraries
import rasterio
from rasterio.plot import reshape_as_image
import numpy as np
import matplotlib.pyplot as plt
import glob

# from scipy.misc import imread, imresize

#Declaring global variables
fp_labels = r"D:/Test_task/labels/" # path for the GeoTiff images
sp_labels = r"D:/Test_task/data/Masks/" # Save path for the processed labels Data 

lis =[] #global list containing paths of all images/labels

def normalize(array):
    """Performs normalizing taskon every band of raster image, inputted in the form of numpy array

    Args:
        array ([array]): Band information stored in Numppy arrays

    Returns:
        [array]: Normalized array.
    """
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)

def append(path):
    for file in glob.iglob(path + "**/**.tif"):
        # Specify band number in the pranthesis of the read method. The below method returns 2D numpy array
        #img = rasterio.open(file)
        lis.append(file)
    return lis

def load_img(image):
    """Goes through the given raster images directory and reads the specified the GeoTiff raster dataset.
    Later, returns the list of the paths of the images/labels

    Args:
        path ([string]): [absolute path of the root dir of images/labels]
    Returns:
        [array]: [image/label]
    """
    # Specify band number in the pranthesis of the read method. The below method returns 2D numpy array
    img = rasterio.open(image)
    # lis.append(img)
    # img = np.array(lis)
    return img

def disp_labels(lis,path,index):
    """Reads the labels of the raster images and are later processed and stored in the form of image data 
    """
    img = lis[index]
    img = load_img(img)
    img = img.read()
    # image in the format of (channel, width ,height) changing it now to (height, width, channel) with rasterio lib
    img = reshape_as_image(img)
    plt.imsave(path + "img" + "_" + str(index) + ".jpg", img[:,:,0],  dpi = 1000)
    
            
def main():
    lis = append(fp_labels)
    for index in range(len(lis)):
        disp_labels(lis, sp_labels, index)

if __name__ =="__main__":
    main()