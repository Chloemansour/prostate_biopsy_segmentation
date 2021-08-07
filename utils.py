### BMIF804 - Mini Project
### Author: Chloe Mansour
### Student Number: 20061726
### Date: August 7th, 2021

### Description of File:

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def prostate_segmenter(volumetric_data, seed, multiplier, radius, num_iteration):
    '''

    :param volumetric_data:
    :param seed:
    :param multiplier:
    :param radius:
    :param num_iteration:
    :return:
    '''

    confid_filter = sitk.ConfidenceConnectedImageFilter()
    confid_filter.SetSeedList(seed)
    confid_filter.SetMultiplier(multiplier)
    confid_filter.SetInitialNeighborhoodRadius(radius)
    confid_filter.SetNumberOfIterations(num_iteration)

    volume_mask = confid_filter.Execute(volumetric_data)

    return volume_mask


def seg_eval_dice(ref_mask, mask):
    '''

    :param ref_mask:
    :param mask:
    :return:
    '''
    dice_coefficient = sitk.LabelOverlapMeasuresImageFilter()
    dice_coefficient.Execute(ref_mask, mask)
    dice_coefficient.GetDiceCoefficient(ref_mask, mask)

    return dice_coefficient.GetDiceCoefficient(ref_mask, mask)

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

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(ref_mask)
    centroid = stats.GetCentroid(1)

    return centroid

def pixel_extract(img, point, width):
    pass


