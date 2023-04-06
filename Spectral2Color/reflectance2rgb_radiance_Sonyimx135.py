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
t = Ct.Transform() # Instance of ColorTransform class
#
interpolate = True
E = 'extended_D65' # choose among: extended_D65, extended_A, Solar, D65, D50, or A
rgb_camera = 'sonyimx135' # choose among: sonyimx135, NikonD810, NikonD7000, or basler
###
if E =='extended_D65':
    illuminant = np.load('/utils/extended_D65_380_1000_interp1nm.npy')
elif E == 'extended_A':
    illuminant = np.load('/utils/extended_A_380_1000_interp1nm.npy')
elif E == 'Solar':
    illuminant = np.load('/utils/solar_380_1000_step_interp1nm.npy')
elif E =='D65':
    illuminant = np.loadtxt('/utils/illuminantD65.txt',delimiter=',')
    illuminant[:,1] = illuminant[:,1]/np.max(illuminant[:,1]) # make its values range between 0 and 1
elif E =='D50':
    illuminant = np.load('/utils/D50_1nm_305_780.npy')
    illuminant[:,1] = illuminant[:,1]/np.max(illuminant[:,1]) # make its values range between 0 and 1
elif E =='A':
    illuminant = np.loadtxt('/utils/illuminantA.txt',delimiter=',')
    illuminant[:,1] = illuminant[:,1]/np.max(illuminant[:,1]) # make its values range between 0 and 1
else:
    print('Unavailable or Unkown illuminant\nDefault illuminant is: Extended D65')
    illuminant = np.load('/utils/extended_D65_380_1000_interp1nm.npy')
###
if rgb_camera == 'sonyimx135':
    rgb_ssfs = np.load('/utils/sonyimx135_ssfs.npy')
elif rgb_camera == 'NikonD810':
    rgb_ssfs = np.load('/utils/requirements/NikonD810.npy')
elif rgb_camera == 'NikonD7000':
    rgb_ssfs = np.load('/utils/NikonD7000_ssfs.npy')
elif rgb_camera == 'basler':
    rgb_ssfs = np.load('/utils/requirements/basler_SSFS.npy')
else:
    print('Unavailable or Unkown camera\nDefault camera is: sonyimx135')
    rgb_ssfs = np.load('/utils/sonyimx135_ssfs.npy')
##
t = Ct.Transform() # Instance of ColorTransform class
##-------------------------------------------------------
for number in tqdm(range(len(imageLists))):
    current_image = imageLists[number]
    hdr = current_image[0]
    raw = current_image[1]
    obj = envi.open(hdr, raw)
    camera_wvs = obj.metadata['wavelength']
    camera_wvs = np.asarray([np.round(float(camera_wvs[i]), 1) for i in range(len(camera_wvs))])
    cube = obj.asarray()
    rows, cols, K = cube.shape
    directory = os.path.dirname(obj.filename)
    filename = os.path.basename(obj.filename)
    name, extension = os.path.splitext(filename)
    image_name = os.path.basename(directory)
    print('Procesing image ' + os.path.basename(obj.filename))
    if interpolate:
        min_wv = np.min(camera_wvs)
        max_wv = np.max(camera_wvs)
        camera_new_wvs = np.arange(min_wv, max_wv, 1)
        spectra = cube.reshape((rows * cols, K))
        spectra = t.interpolate_spectra(spectra, camera_wvs,step=1)  # Interpolate reflectance spectra at 1nm between 475 and 901 nm
    #----- Simulate RGB radiance image K-channel reflectance image using Sony Sonyimx135 SSFS
    closest_idx_bands_camera, closest_idx_bands_rgb,closest_idx_bands_illuminant = t.get_closest_reflectance_spectra_to_rgbssfs_and_illuminant(camera_new_wvs, sony_ssfs,illuminant)
    Sony_RGB_D65 = t.Spectralreflectance_to_RGB(spectra, sony_ssfs,illuminant,closest_idx_bands_camera,closest_idx_bands_rgb,closest_idx_bands_illuminant,rows,cols) # Nb pixels = 2048x2048
    Sony_RGB_D65 = t.normalize8(Sony_RGB_D65)
    gamma_Sony_RGB_D65 = t.adjust_gamma(Sony_RGB_D65, gamma=2.0)
    io.imsave(directory + '/'+'RGB_'+rgb_camera+'_'+E+'.png',RGBNIR_reflectance) #Save RGBNIR image in PNG format
