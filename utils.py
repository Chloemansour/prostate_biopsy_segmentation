### BMIF804 - Mini Project
### Author: Chloe Mansour
### Student Number: 20061726
### Date: August 7th, 2021

### Description of File:

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def prostate_segmenter(volumetric_data, seed, sigma=1.5):


    feature_img = sitk.GradientMagnitudeRecursiveGaussian(volumetric_data, sigma=sigma)

    speed_img = sitk.BoundedReciprocal(feature_img)

    fm_filter = sitk.FastMarchingBaseImageFilter()
    fm_filter.SetTrialPoints(seed)
    fm_filter.SetStoppingValue(1000)
    seg = fm_filter.Execute(speed_img)

    threshold_img = sitk.Threshold(seg, lower=0.0, upper=fm_filter.GetStoppingValue(),
                           outsideValue=0)

    cleaned_overlay = sitk.BinaryMorphologicalClosing(
        sitk.Cast(threshold_img, sitk.sitkUInt16),
        (1, 1, 1),
        sitk.sitkBall)

    binary_overlay = sitk.BinaryThreshold(cleaned_overlay,
                                          upperThreshold=0,
                                          insideValue=0,
                                          outsideValue=1)

    volume_mask = sitk.Cast(binary_overlay, sitk.sitkInt8)

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
    haus_filter.Execute(ref_mask, mask)
    haus_filter.GetHausdorffDistance(ref_mask,mask)

    return haus_filter.GetHausdorffDistance(ref_mask,mask)

def get_target_loc(ref_mask):
    '''

    :param ref_mask:
    :return:
    '''

    area_dict = {}

    for s in range(10, 32):
        img = ref_mask[:, :, s]
        overlay_stats = sitk.LabelShapeStatisticsImageFilter()
        overlay_stats.Execute(img)
        overlay_area = overlay_stats.GetPhysicalSize(1)
        area_dict[s] = overlay_area

    print(
        f"The image slide with the largest surface area of {max(area_dict.values())} is the image slice {max(area_dict, key=area_dict.get)}")

    # image with largest
    thick_mask = ref_mask[:, :, max(area_dict, key=area_dict.get)]

    # get centroid point in LPS
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(thick_mask)
    centroid = stats.GetCentroid(1)

    print("The centroid point to preform the biopsy is at:", centroid)

    centroid_idx = thick_mask.TransformPhysicalPointToIndex(centroid)

    centroid_final = (centroid_idx[0], centroid_idx[1], max(area_dict, key=area_dict.get))
    centroid_final = ref_mask.TransformIndexToPhysicalPoint(centroid_final)

    return centroid_final

def pixel_extract(img, point, width):
    lower_section = [location - width / 2 for location in point]
    upper_section = [location + width / 2 for location in point]

    low_value = img.TransformPhysicalPointToIndex(lower_section)
    upper_value = img.TransformPhysicalPointToIndex(upper_section)

    return img[low_value[0]:upper_value[0] + 1, low_value[1]:upper_value[1] + 1, low_value[2]:upper_value[2] + 1]