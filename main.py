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

ex_viewer = sitk.ImageViewer()
slicer_location = "C:/Users/chloe/AppData/Local/NA-MIC/Slicer 4.11.20210226/Slicer.exe"
ex_viewer.SetApplication(slicer_location)

# read in image and segment
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

# get physical extent
spacing = MRI_volume.GetSpacing()
size = MRI_volume.GetSize()
origin = MRI_volume.GetOrigin()

x0 = origin[0]
x1 = origin[0]+spacing[0]*size[0]
y0 = origin[1]
y1 = origin[1]+spacing[1]*size[1]


# seed points
seeds = [MRI_volume.TransformPhysicalPointToIndex((-16.965, -25.925, 13.076)),
         MRI_volume.TransformPhysicalPointToIndex((-0.661, -0.569, 17.104)),
         MRI_volume.TransformPhysicalPointToIndex((-30.940, -2.501, 18.257)),
         MRI_volume.TransformPhysicalPointToIndex((-35.366, -11.534, 16.008)),
         MRI_volume.TransformPhysicalPointToIndex((-15.800, -6.391, 16.342)),
         MRI_volume.TransformPhysicalPointToIndex((-18.363, -19.968, 14.120)),
         MRI_volume.TransformPhysicalPointToIndex((-29.554,-6.220, 16.747)),
         MRI_volume.TransformPhysicalPointToIndex((-38.414,-0.970,17.875)),
         MRI_volume.TransformPhysicalPointToIndex((-18.178,-13.323,26.599)),
         MRI_volume.TransformPhysicalPointToIndex((-16.937, -14.708, 29.173))]




# create mask

prostate_mask = ut.prostate_segmenter(MRI_volume, seeds, sigma=1.5)

# write mask to file
sitk.WriteImage(prostate_mask, "my_segmentation.nrrd")

# overlay mask onto image and view 2D LP slice

img_overlap = sitk.LabelOverlay(MRI_volume,prostate_mask)
img_overlap_scaled = sitk.RescaleIntensity(img_overlap)

#ex_viewer.Execute(img_overlap)

# view middle 2D LP slice
sizez = img_overlap_scaled.GetSize()[2]
z = round(sizez/2)

img_overlap_slice = img_overlap_scaled[:,:,z]
plt.figure(figsize=(15,15))
plt.imshow(sitk.GetArrayFromImage(img_overlap_slice), vmin=0, vmax = 255,  extent = [x0,x1,y0,y1])
plt.show()

# overlay provided mask onto image and view 2D LP slice
given_overlap = sitk.LabelOverlay(MRI_volume, given_mask)

# view middle 2D LP slice
img_given = given_overlap[:,:,z]
img_given = sitk.RescaleIntensity(img_given)
plt.figure(figsize=(15,15))
plt.imshow(sitk.GetArrayFromImage(img_given), vmin=0, vmax=255,  extent = [x0,x1,y0,y1])
plt.axis('off')
plt.show()



centroid_final = ut.get_target_loc(given_mask)
voxel_biopsy = ut.pixel_extract(MRI_volume, centroid_final, 6)

# plot distribution of pixel intensities around biopsy target
plt.boxplot(voxel_biopsy)
plt.ylabel("Pixel Intensity")
plt.xlabel(centroid_final)
plt.show()