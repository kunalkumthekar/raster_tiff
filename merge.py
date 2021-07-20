    """Helps to merge all the raster images and create a merged map"""
# Imports
import rasterio
from rasterio.merge import merge
import glob

# Declaring paths for datasets
img_path = "D:/Test_task/images/"
label_path = "D:/Test_task/labels/"
op_path = "D:/Test_task/mosaic.tif"

def main():
    """This function reads multiple raster images with multiple channels and attempts to merge them to create a raster map
    The map being a raster map, every grid that the image is divided in consists of information (metadata) like 
     Spatial Extent( X,Y coordinates), CRS (Coordinate Reference System),Resolution etc..
    After the merge of the images is done, open the Mosaic.tif (map) with the help of open source software "QGIS" / Use trial.py
    """
    tif_list = []
    for file in glob.iglob(label_path + "**/**.tif", recursive=True):
        dataset = rasterio.open(file)
        tif_list.append(dataset)
    #print(tif_list)
    # Merging the multispectral raster cells in a single map. If limited memeory, limit number of images for merge providing partial info.
    # Mosaic basically represents the overlapping or sometimes unconnected raster grids sewed together.
    mosaic_label, out_trans = merge(tif_list[:]) 
    out_meta = dataset.meta.copy()
    out_meta.update({"driver": "GTiff",
        "height": mosaic_label.shape[1],
        "width": mosaic_label.shape[2],
        "transform": out_trans,
        "crs": "+proj=utm +zone=35 +ellps=GRS80 +units=m +no_defs" })
    with rasterio.open(op_path, "w", **out_meta) as dest:
        dest.write(mosaic_label)
    print(mosaic_label.shape)

if __name__== "__main__":
    main()