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
    '''

    :param volumetric_data:
    :param seed:
    :param lower:
    :param upper:
    :return:
    '''

    connected_filter = sitk.ConnectedThresholdImageFilter()
    connected_filter.SetSeedList(seed)
    connected_filter.SetUpper(upper)
    connected_filter.SetLower(lower)

    volume_mask = connected_filter.Execute(volumetric_data)

    return volume_mask


def seg_eval_dice(ref_mask, mask):
    '''

    :param ref_mask:
    :param mask:
    :return:
    '''
    dice_coefficient = sitk.LabelOverlapMeasuresImageFilter()
    dice_coefficient.GetDiceCoefficient(ref_mask, mask)

    return dice_coefficient.Execute(ref_mask, mask)

def seg_eval_hausdorff(ref_mask, mask):
    '''

    :param ref_mask:
    :param mask:
    :return:
    '''

    haus_filter = sitk.HausdorffDistanceImageFilter()
    haus_filter.GetHausdorffDistance(ref_mask,mask)

    return haus_filter.Execute(ref_mask, mask)

def get_target_loc(ref_mask):
    '''

    :param ref_mask:
    :return:
    '''

    x = [points[0] for points in ref_mask]
    y = [points[1] for points in ref_mask]

    centroid_mask = (sum(x) / len(ref_mask),
                     sum(y) / len(ref_mask))

    centroid_mask = (centroid_mask[0].TransformIndexToPhysicalPoint, centroid_mask[1].TransformIndexToPhysicalPoint)

    return centroid_mask

def pixel_extract(img, point, width):
    pass