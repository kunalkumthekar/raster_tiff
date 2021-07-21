# Image Segmentation on Raster Data #
This repository deals with image segmentation task performed on raster data. The data is not trained from scratch and transfer learning is implemented to fine tune the pretrained model. This repo is based on the work done by @msminhas93 (https://github.com/msminhas93/DeepLabv3FineTuning/).

The objective was to perform accurate segmentation on the Multispectral raster data with the band order (4,3,2) and compare its results with the raster data (all bands). Currently, the results obtained are limited to raster data with band order (4,3,2). The result obtained is not currently satisfactory and work should be done to improvise.
The following is the sample result obtained: 
![Samples Segmentation output](./experiment/SegmentationOutput_Bands_432.png)

## Introduction ##
### Raster Data ###
Raster maps are divided into grid cells (pixels), where every cell may represent multiple information like temperature, magnitude, altitude and so on. Representing data in raster format has its own advantages like:
1. Uniformly storing multiple information which is useful for carrying out advanced and detailed statistical and spatial analysis.
2. Useful to perform surface analysis.
3. The Matrix Data structure uniformy stores information in the form of lines, points, plygons etc..

One of the few disadvantages that was observed is that Raster Grid Data when merged together requires huge amount of memory upto number of Gigabytes.

### Raster Bands ###
![Sample raster band info](./experiment/raster_band.gif)

(credits : https://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/raster-bands.htm)

Bands in raster data represents layers. Every layer has particular information (types of info mentioned above) stored in it. The band information is stored either in the form of binary image (grayscale), or color (RGB). The RGB color are defined in the band matrix.
Inorder to create a an RGB composite dataset, 3 bands information was stacked together. Have a look at the "data" folder in the repo.
Sample:

### Creating Dataset ###
The dataset is created using the files ```image_dataset.py``` and ```label_dataset.py```
image_dataset.py reads the specefic band information from the GeoTiff dataset provided. In our case we are concerned with band information 4,3,2. The bands are stacked together to form and RGB image and is written on disk, creating out custom dataset. 

![Samples Images](./data/Images/img_1.jpg)
![Samples Masks](./data/Masks/img_1.jpg)

### Usage ###
Usage : 
```
python main.py --data-directory data --exp_directory experiment --epochs 25 --batch-size 25
  --data-directory TEXT  Specify the data directory.  [required]
  --exp_directory TEXT   Specify the experiment directory.  [required]
  --epochs INTEGER       Specify the number of epochs you want to run the
                         experiment for. Default is 25.

  --batch-size INTEGER   Specify the batch size for the dataloader. Default is 4.
  --help                 Show this message and exit.
```

