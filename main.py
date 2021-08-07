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
         MRI_volume.TransformPhysicalPointToIndex((-35.366, -11.534, 16.008)),
         MRI_volume.TransformPhysicalPointToIndex((-15.800, -6.391, 16.342)),
         MRI_volume.TransformPhysicalPointToIndex((-18.363, -19.968, 14.120)),
         MRI_volume.TransformPhysicalPointToIndex((-29.554,-6.220, 16.747)),
         MRI_volume.TransformPhysicalPointToIndex((-38.414,-0.970,17.875)),
         MRI_volume.TransformPhysicalPointToIndex((-27.456,-19.319,14.478)),
         MRI_volume.TransformPhysicalPointToIndex((-0.177, -4.018, 16.316))]


'''feature_img = sitk.GradientMagnitudeRecursiveGaussian(MRI_volume, sigma=1.5)
speed_img = sitk.BoundedReciprocal(feature_img)

fm_filter = sitk.FastMarchingBaseImageFilter()
fm_filter.SetTrialPoints(seeds)
fm_filter.SetStoppingValue(500)
seg = fm_filter.Execute(speed_img)

thres = sitk.Threshold(seg, lower=0.0, upper=fm_filter.GetStoppingValue(),
                       outsideValue=fm_filter.GetStoppingValue()+1)

thres = sitk.Cast(thres, sitk.sitkInt8)
prostate_mask = thres[:,:,z]'''

'''plt.imshow(sitk.GetArrayFromImage(thres_img), vmin=0, vmax = 255, cmap = "gray", extent = [x0,x1,y0,y1])
plt.show()'''

# create mask
#prostate_mask = ut.prostate_segmenter(MRI_volume, seeds, 800, 1600)

# write mask to file
#sitk.WriteImage(thres, "my_segmentation.nrrd")

# overlay mask onto image and view 2D LP slice
#img_overlap = sitk.LabelOverlay(MRI_volume,thres)
#ex_viewer.Execute(img_overlap)
# view middle 2D LP slice
'''img_overlap = img_overlap[:,:,z]
plt.figure(figsize=(15,15))
plt.imshow(sitk.GetArrayFromImage(img_overlap), vmin=0, vmax = 255,  extent = [x0,x1,y0,y1])
plt.show()'''

# overlay provided mask onto image and view 2D LP slice
given_overlap = sitk.LabelOverlay(MRI_volume, given_mask)

# view middle 2D LP slice
img_given = given_overlap[:,:,z]
plt.figure(figsize=(15,15))
plt.imshow(sitk.GetArrayFromImage(img_given), vmin=0, vmax=255,  extent = [x0,x1,y0,y1])
plt.axis('off')
plt.show()

area_dict = {}

for s in range(10,32):
    img = given_mask[:,:,s]
    overlay_stats = sitk.LabelShapeStatisticsImageFilter()
    overlay_stats.Execute(img)
    overlay_area = overlay_stats.GetPhysicalSize(1)
    area_dict[s] = overlay_area

print(f"The image slide with the largest surface area of {max(area_dict.values())} is the image slice {max(area_dict,key=area_dict.get)}")

# image with largest
thick_mask = given_mask[:, :, max(area_dict, key=area_dict.get)]

# get centroid point in LPS
centroid = ut.get_target_loc(thick_mask)
print("The centroid point to preform the biopsy is at:", centroid)

centroid_idx = thick_mask.TransformPhysicalPointToIndex(centroid)

centroid_final = (centroid_idx[0],centroid_idx[1], max(area_dict, key=area_dict.get))
centroid_final = given_mask.TransformIndexToPhysicalPoint(centroid_final)
print(centroid_final)

