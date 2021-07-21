# Image Segmentation on Raster Data #
This repository deals with image segmentation task performed on raster data. The data is not trained from scratch and transfer learning is implemented to fine tune the pretrained model. This repo is based on the work done by @msminhas93 (https://github.com/msminhas93/DeepLabv3FineTuning/).

The objective was to perform accurate segmentation on the Multispectral raster data with the band order (4,3,2) and compare its results with the raster data (all bands). Currently, the results obtained are limited to raster data with band order (4,3,2).
The following is the sample result obtained : 
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


