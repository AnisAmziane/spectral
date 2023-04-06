# -*- coding: utf-8 -*-
import numpy as np
import spectral.io.envi as envi
import sys,os
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

base_path = 'path to radiance base' # path to image database of radiance images
imageLists = get_hdr_and_raw_paths(base_path)
t = Ct.Transform() # Instance of ColorTransform class

# Convert all images in database to CIE-RGB color space
for number in tqdm(range(len(imageLists))):
    current_image = imageLists[number]
    hdr = current_image[0]
    raw = current_image[1]
    obj = envi.open(hdr, raw)
    cube = obj.asarray() # K channel radiance
    rows, cols, K = cube.shape
    spectra = cube.reshape((rows * cols, K))
    directory = os.path.dirname(obj.filename)
    filename = os.path.basename(obj.filename)
    name, extension = os.path.splitext(filename)
    image_name = os.path.basename(directory)
    print('Procesing image ' + os.path.basename(image_name))
    ### 1- Radiance spectra to XYZ color space
    camera_wvs = obj.metadata['wavelength']
    camera_wvs = np.asarray([np.round(float(camera_wvs[i]), 1) for i in range(len(camera_wvs))])
    closest_idx, cmf_values = t.find_closest_spectra_cmf_illuminant_with_interpolation(camera_wvs)
    interp_spectra = t.interpolate_spectra(spectra, camera_wvs,step=1)  # Interpolate reflectance spectra at 1nm between 475 and 901 nm
    XYZ_img = t.radiance_to_xyz_v2(interp_spectra, closest_idx, cmf_values, rows, cols)
    ### 2- XYZ to CIE-RGB color space
    CIERGB = t.XYZ_2_CIERGB(XYZ_img)
    CIERGB = t.normalize8(CIERGB)
    gamma_ciergb = t.adjust_gamma(CIERGB, gamma=2.2)
    np.save(directory + '/' + 'CIERGB_D65.npy', gamma_ciergb)  # Save AdobeRGB image in numpy format
    io.imsave(directory + '/' + 'CIERGB_D65.png', gamma_ciergb)  # Save AdobeRGB image in PNG format



