
"""This file makes use of Rasterio lib to import and display raster snippets. Then the required band information is merged
together and then allocated as Training or Validation dataset"""

#  Importing required libraries
import rasterio
from rasterio.plot import reshape_as_image, show
import numpy as np
import matplotlib.pyplot as plt
import glob

#Declaring global variables
# TO DO - Place all paths as argparsers
fp_images = r"D:/Test_task/images/" # Path of the directory containing all the Geotiff images
sp_images = r"D:/Test_task/data/Images/" # Save path for the processed images Data

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
    """Goes through the main directory for images / labels and appends the empty list "lis" with the paths of individial images/labels

    Args:
        path ([str]): paths of dir containing GeoTiff images/labels (.tif)

    Returns:
        [list]: List containing paths of images/labels
    """
    for file in glob.iglob(path + "**/**.tif"):
        #img = rasterio.open(file)
        lis.append(file)
    return lis

def load_img(image):
    """Goes through the given raster image and reads the specified the GeoTiff raster dataset.
    Later, returns the read images/labels

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

def disp(lis, path, index):
    """Reads the bands of the raster images and merges the specified band stack to create a single raster image
    The images are later normalized and stored in respective save folders in form of custom datasets
    """
    # Raster Images have multiple bands. The method "read" can read individual bands and return a 2D numpy array (height,width)
    
    img = lis[index]
    img = load_img(img)
    # Reading bands from the multispectral data 
    # Bands are indexed from 1 and "NOT 0"
    show(img)
    img1 = img.read()
    img1 = reshape_as_image(img1)
    print(img1.shape)

    # img4 = img.read(4) #Band 4
    # show(img4)
    # img3 = img.read(3) #Band 3
    # img2 = img.read(2) #Band 2
    # img5 = img.read(5)
    # img6 = img.read(6)
    # img7 = img.read(7)
    # img8 = img.read(8)
    # img9 = img.read(9)
    # img10 = img.read(10)
    # img11 = img.read(11)
    # img12 = img.read(12)

    # img4_norm = normalize(img4)
    # show(img4_norm)
    # img3_norm = normalize(img3)
    # img2_norm = normalize(img2)
    # img1_norm = normalize(img1)
    # img5_norm = normalize(img5)
    # img6_norm = normalize(img6)
    # img7_norm = normalize(img7)
    # img8_norm = normalize(img8)
    # img9_norm = normalize(img9)
    # img10_norm = normalize(img10)
    # img11_norm = normalize(img11)
    # img12_norm = normalize(img12)

    # # print(img12_norm.shape)
    # img_final = np.dstack((img4_norm,img3_norm,img2_norm,img1_norm,img5_norm,img6_norm,img7_norm,img8_norm,img9_norm,img10_norm,img11_norm,img12_norm)) # Merging band information (4,3,2)
    # # img_final = np.dstack((img4,img3,img2,img1,img5,img6,img7,img8,img9,img10,img11,img12))
    # # img_final = reshape_as_image(img_final)
    # # img_final = np.reshape(img_final, [128,128,3])
    # print(img_final.shape)
    # show(img_final)
    # # img_final.reshape((2,0,1))
    # # img_final = np.moveaxis(img_final, -1, 0)
    # # plt.imsave(path + "img" + "_" + str(index) + ".jpg", img_final,  dpi = 1000)

            
def main():
    lis = append(fp_images)
    for index in range(len(lis)):
        disp(lis, sp_images, index)

if __name__ =="__main__":
    main()