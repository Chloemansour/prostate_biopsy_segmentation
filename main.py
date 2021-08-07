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

MRI_volume = sitk.RescaleIntensity(MRI_volume, 0, 255)
MRI_volume = sitk.Cast(MRI_volume, sitk.sitkUInt8)

# get physical extent
spacing = MRI_volume.GetSpacing()
size = MRI_volume.GetSize()
origin = MRI_volume.GetOrigin()

x0 = origin[0]
x1 = origin[0]+spacing[0]*size[0]
y0 = origin[1]
y1 = origin[1]+spacing[1]*size[1]


# extract middle slice on LP plane
sizez = MRI_volume.GetSize()[2]
z = round(sizez/2)

img = MRI_volume[:,:,z]
# plot data
plt.figure(figsize=(15,5))
plt.imshow(sitk.GetArrayFromImage(img), extent = [x0,x1,y0,y1], cmap = "gray")
plt.axis('off')
plt.show()

# seed points
seeds = [MRI_volume.TransformPhysicalPointToIndex((-16.965, -25.925, 13.076)),
         MRI_volume.TransformPhysicalPointToIndex((-0.661, -0.569, 17.104)),
         MRI_volume.TransformPhysicalPointToIndex((-30.940, -2.501, 18.257)),
         MRI_volume.TransformPhysicalPointToIndex((-35.366, -11.534, 16.008))]

# create mask
prostate_mask = ut.prostate_segmenter(MRI_volume, seeds, 800, 1600)

# write mask to file
#sitk.WriteImage(prostate_mask, "my_segmentation.nrrd")

# overlay mask onto image and view 2D LP slice
img_overlap = sitk.LabelOverlay(MRI_volume, prostate_mask)
img_overlap = img_overlap[:,:,z]
plt.figure(figsize=(15,15))
plt.imshow(sitk.GetArrayFromImage(img_overlap), vmin=0, vmax = 255,  extent = [x0,x1,y0,y1])
plt.show()
# overlay provided mask onto image and view 2D LP slice

given_overlap = sitk.LabelOverlay(MRI_volume, given_mask)
img_given =  given_overlap[:,:,z]
plt.figure(figsize=(15,15))
plt.imshow(sitk.GetArrayFromImage(img_given), vmin=0, vmax = 255,  extent = [x0,x1,y0,y1])
plt.axis('off')
plt.show()