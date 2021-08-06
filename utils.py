### BMIF804 - Mini Project
### Author: Chloe Mansour
### Student Number: 20061726
### Date: August 7th, 2021

### Description of File:

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def prostate_segmenter(volumetric_data, seed, lower, upper):

    connected_filter = sitk.ConnectedThresholdImageFilter()
    connected_filter.SetSeedList(seed)
    connected_filter.SetUpper(upper)
    connected_filter.SetLower(lower)

    volume_mask = connected_filter.Execute(volumetric_data)

    return volume_mask


def seg_eval_dice(ref_mask, mask):
    dice_coefficient = sitk.LabelOverlapMeasuresImageFilter()
    dice_coefficient.GetDiceCoefficient(ref_mask, mask)

    return dice_coefficient.Execute(ref_mask, mask)

def seg_eval_hausdorff(ref_mask, mask):
    haus_filter = sitk.HausdorffDistanceImageFilter()
    haus_filter.GetHausdorffDistance(ref_mask,mask)

    return haus_filter.Execute(ref_mask, mask)

def get_target_loc(mask):
    pass


def pixel_extract(img, width):
    pass