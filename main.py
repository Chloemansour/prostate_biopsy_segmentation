### BMIF804 - Mini Project
### Author: Chloe Mansour
### Student Number: 20061726
### Date: August 7th, 2021

### Link to repository:


### Description of File:

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utils as ut

# read in image
MRI_volume = sitk.ReadImage("Case11.mhd")
given_mask = sitk.ReadImage("Case11_segmentation.mhd")
# Preform image preprocessing
# print meta data
print(MRI_volume.GetSize())
print(MRI_volume.GetSpacing())
print(MRI_volume.GetPixelIDTypeAsString())

# max and min pixel values
pixel_filter = sitk.StatisticsImageFilter()
pixel_filter.Execute(MRI_volume)

print("Min pixel value:", pixel_filter.GetMinimum())
print("Max pixel value:", pixel_filter.GetMaximum())

# visualize MRI volume image pixel intensities
plt.hist(sitk.GetArrayFromImage(MRI_volume).flatten(), bins=70)
plt.show()

# rescale intensity to 0 - 255

# MRI_volume = sitk.RescaleIntensity(MRI_volume, 0, 255)
# MRI_volume = sitk.Cast(MRI_volume, sitk.sitkUInt8)

# extract middle slice on LP plane
z = (0, MRI_volume.GetSize()[2]-1)

# plot data
plt.figure(figsize=(5,5))
plt.imshow(sitk.GetArrayFromImage(MRI_volume)[:,z], cmap = "gray")
plt.axis('off')
plt.show()

# seed points
seeds = [MRI_volume.TransformPhysicalPointToIndex((-19.238, -11.734,27.457)),
         MRI_volume.TransformPhysicalPointToIndex((-3.698, -7.554, 2.696)),
         MRI_volume.TransformPhysicalPointToIndex((-21.334, -5.608, -8.828)),
         MRI_volume.TransformPhysicalPointToIndex((-33.998, -9.346, 13.313))]

'''# create mask
prostate_mask = ut.prostate_segmenter(MRI_volume, seeds, 300, 1850)

# write mask to file
sitk.WriteImage(prostate_mask, "my_segmentation.nrrd")

# overlay mask onto image and view 2D LP slice
img_overlap = sitk.LabelOverlay(MRI_volume, prostate_mask)

plt.figure(figsize=(15,15))
plt.imshow(sitk.GetArrayFromImage(img_overlap))'''

# overlay provided mask onto image and view 2D LP slice

given_overlap = sitk.LabelOverlay(MRI_volume, given_mask)

plt.figure(figsize=(15,15))
plt.imshow(sitk.GetArrayFromImage(given_overlap)[z])