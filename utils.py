### BMIF804 - Mini Project
### Author: Chloe Mansour
### Student Number: 20061726
### Date: August 7th, 2021

### Description of File: contains functions for prostate biopsy workflow

import SimpleITK as sitk



def prostate_segmenter(volumetric_data, seed, sigma):
    '''
       This function takes a volumetric image and creates a segmentation over a specified location
    :param volumetric_data: MRI image
    :param seed: list of seed points obtained from 3D slicer
    :param sigma: value from 0-2
    :return: volume mask - a segment image
    '''

    # apply gradient gaussian to image
    feature_img = sitk.GradientMagnitudeRecursiveGaussian(volumetric_data, sigma=sigma)
    # apply bounded reciprocol to image
    speed_img = sitk.BoundedReciprocal(feature_img)

    # apply fast-marching image filter to image
    fm_filter = sitk.FastMarchingBaseImageFilter()
    fm_filter.SetTrialPoints(seed)
    fm_filter.SetStoppingValue(1000)
    seg = fm_filter.Execute(speed_img)
    # threshold image
    threshold_img = sitk.Threshold(seg, lower=0.0, upper=fm_filter.GetStoppingValue(),
                           outsideValue=0)
    # clean segment borders and fill in gaps
    cleaned_overlay = sitk.BinaryMorphologicalClosing(
        sitk.Cast(threshold_img, sitk.sitkUInt16),
        (1, 1, 1),
        sitk.sitkBall)

    binary_overlay = sitk.BinaryThreshold(cleaned_overlay,
                                          upperThreshold=0,
                                          insideValue=0,
                                          outsideValue=1)
    # create segment and cast to 8 bit integer
    volume_mask = sitk.Cast(binary_overlay, sitk.sitkInt8)

    return volume_mask


def seg_eval_dice(ref_mask, mask):
    '''
    Preforms dice coeiffient on two image segments
    :param ref_mask: mask segment provided by professional
    :param mask: mask segment created using prostate_segmenter function
    :return: value of dice coefficient
    '''
    # apply LabelOverLapMeasuresImageFilter image filter to access dice coefficient evaluation
    dice_coefficient = sitk.LabelOverlapMeasuresImageFilter()
    dice_coefficient.Execute(ref_mask, mask)

    return dice_coefficient.GetDiceCoefficient()

def seg_eval_hausdorff(ref_mask, mask):
    '''
    Preforms hausdorff evaluation on two image segments
    :param ref_mask: mask segment provided by professional
    :param mask: mask segment created using prostate_segmenter function
    :return: value of hausdorff evaluation
    '''

    # apply Hausdorff image filter
    haus_filter = sitk.HausdorffDistanceImageFilter()
    haus_filter.Execute(ref_mask, mask)

    return haus_filter.GetHausdorffDistance()

def get_target_loc(ref_mask):
    '''
    Determine image slice with largest cross-sectional area of prostate, using that image slice,
    determine centroid location for biopsy placement
    :param ref_mask: segmentation mask provided by professional
    :return: centroid location in LPS coordinates
    '''

    area_dict = {}
    # get area of every image slice with a label value of 1, add to dictionary
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

    # transform to pixel index
    centroid_idx = thick_mask.TransformPhysicalPointToIndex(centroid)

    # add z dimension, and convert back to physical point (LPS)
    centroid_final = (centroid_idx[0], centroid_idx[1], max(area_dict, key=area_dict.get))
    centroid_final = ref_mask.TransformIndexToPhysicalPoint(centroid_final)
    print("The centroid point to preform the biopsy is at:", centroid_final)

    return centroid_final

def pixel_extract(img, point, width):
    '''

    :param img: MRI volumetric img, image
    :param point: centroid point of biopsy placement, tuple
    :param width: width of cube to be extracted, integer
    :return: dimensions of extracted tissues and pixel intensities
    '''
    # determine upper and lower dimensions of cude to be extracted in LPS coordinates
    lower_section = [location - width / 2 for location in point]
    upper_section = [location + width / 2 for location in point]

    # convert locations to pixel indexes
    low_value = img.TransformPhysicalPointToIndex(lower_section)
    upper_value = img.TransformPhysicalPointToIndex(upper_section)

    return img[low_value[0]:upper_value[0] + 1, low_value[1]:upper_value[1] + 1, low_value[2]:upper_value[2] + 1]