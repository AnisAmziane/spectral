# -*- coding: utf-8 -*-
import numpy as np
import spectral.io.envi as envi
import sys
import os
from tqdm import tqdm
from skimage import io
import glob
from Spectral2Color import ColorTransform as Ct

def get_hdr_and_raw_paths(images_path):
    """
    :param images_path: path to image dataset
    :return:  list of tuples (hdr,raw), assuming that hyper/multi spectral images are stored in ENVI format
    """
    contain_hdrs = glob.iglob(images_path + '**/*.hdr', recursive=True)
    contain_raws = glob.iglob(images_path + '**/*.raw', recursive=True)
    hdrs = []
    raws = []
    imageLists = []
    for f1 in contain_hdrs:
        hdrs.append(f1)
    for f2 in contain_raws:
        raws.append(f2)
    for i in range(len(hdrs)):
        filename = os.path.basename(hdrs[i])
        name, extension = os.path.splitext(filename)
        for ii in range(len(raws)):
            filename2 = os.path.basename(raws[ii])
            name2, extension2 = os.path.splitext(filename2)
            if name == name2:
                temp = (hdrs[i], raws[ii])
                imageLists.append(temp)
    return imageLists

base_path = 'path to reflectance base' # path to image database of reflectance images
imageLists = get_hdr_and_raw_paths(base_path)
#
t = Ct.Transform() # Instance of ColorTransform class
monno_ssfs = np.load('/home/anis/PycharmProjects/ms_propject/venv/Codes/requirements/new_monno_ssfs_1nm.npy')
##-------------------------------------------------------
for number in tqdm(range(len(imageLists))):
    current_image = imageLists[number]
    hdr = current_image[0]
    raw = current_image[1]
    obj = envi.open(hdr, raw)
    camera_wvs = obj.metadata['wavelength']
    camera_wvs = np.asarray([np.round(float(camera_wvs[i]), 1) for i in range(len(camera_wvs))])
    cube = obj.asarray()
    directory = os.path.dirname(obj.filename)
    filename = os.path.basename(obj.filename)
    name, extension = os.path.splitext(filename)
    image_name = os.path.basename(directory)
    print('Procesing image ' + os.path.basename(obj.filename))
    #----- Simulate RGB+NIR reflectance images images from K-channel reflectance images using Monno SSFS
    closest_reflectance_spectra, closest_idx_bands_rgbnir = t.get_closest_reflectance_spectra_to_targetssfs(cube,camera_wvs, monno_ssfs)
    RGBNIR_reflectance = t.Spectralreflectance_to_RGBNIRreflectance(closest_reflectance_spectra, monno_ssfs, closest_bands_rgbnir, 2048, 2048) # Nb pixels = 2048x2048
    np.save(directory + '/'+'RGBNIR_Monno_reflectance_rw.npy',RGBNIR_reflectance) #Save RGBNIR image in numpy format




