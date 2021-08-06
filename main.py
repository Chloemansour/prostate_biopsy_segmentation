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

# print meta data
print(MRI_volume.GetSize())
print(MRI_volume.GetSpacing())
print(MRI_volume.GetPixelIDTypeAsString())


# create mask
prostate_mask = ut.prostate_segmenter(MRI_volume)

# write mask to file
sitk.WriteImage(prostate_mask, "my_segmentation.nrrd")

# overlay mask onto image and view 2D LP slice
img_overlap = sitk.LabelOverlay(MRI_volume, prostate_mask)

plt.figure(figsize=(15,15))
plt.imshow(sitk.GetArrayFromImage(img_overlap))

