## Image Segmentation on Raster Data ##
This repository deals with image segmentation task performed on raster data. The data is not trained from scratch and transfer learning is implemented to fine tune the pretrained model. This repo is based on the work done by @msminhas93 (https://github.com/msminhas93/DeepLabv3FineTuning/).

The objective was to perform accurate segmentation on the Multispectral raster data with the band order (4,3,2) and compare its results with the raster data (all bands). Currently, the results obtained are limited to raster data with band order (4,3,2).
The following is the sample result obtained : 
![Samples Segmentation output](./experiment/SegmentationOutput_Bands_432.png) 
