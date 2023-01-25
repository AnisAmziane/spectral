import numpy as np
import cv2,os
from math import *
from numba import jit
from joblib import Parallel, delayed
# from scipy.interpolate import interp1d

class Transform:
    """ Color Space Tranformation Class.
    # The CIE 2-deg XYZ transformed from the CIE (2006) 2-deg LMS cone fundamentals
    # Color matching function for 390 -- 830 nm with 1nm step available at http://cvrl.ucl.ac.uk/cmfs.htm
    """
    cmf = np.loadtxt('/home/anis/PycharmProjects/ms_project/venv/Codes/requirements/lin2012xyz2e_1_7sf.txt')
    illuminant = np.load('/home/anis/PycharmProjects/ms_project/venv/Codes/requirements/extended_D65_380_1000_interp1nm.npy')
    closests_idx = []
    closest_illuminant = []
    closests_wvs = []
    cmf_values = 0
    #
    def find_closest_spectra_cmf_illuminant_with_interpolation(self,camera_wvs):
        """ This function finds the shared wavelengths among camera SSFs, illuminant,  and the CIE color matching function.
        It returns the closet wavelength index in camera SSFs and nthe closest CIE color matching function values.
        Illuminant is Extended D65 by default.
        Inputs:
        camera_wvs:  Wavelengths sampled by camera filters. They are interpolated above for better accuracy
        Outputs:
        Closest wavelength index: array
        cmf_values: [len(Closest wavelength index), 3]
        """
        illuminant_wvs = self.illuminant[:,0] # wvs depend on illuminaant type
        cmf_wvs = self.cmf[:, 0] # 390 nm -- 830 nm with 1 nm step
        # Interpolate camera wvs with a step of 1nm
        min_wv = np.min(camera_wvs)
        max_wv = np.max(camera_wvs)
        camera_new_wvs = np.arange(min_wv, max_wv, 1)
        #------ For imec spectra
        closestFound_1 = self.get_closest_wavelengths(cmf_wvs, camera_new_wvs)
        #------ For cmf functions
        closestFound_2 = self.get_closest_wavelengths(camera_new_wvs,cmf_wvs)
        min_count = np.min([len(closestFound_1), len(closestFound_2)])
        closestFound_1 = closestFound_1[:min_count]
        closestFound_2 = closestFound_2[:min_count]
        #------- For Illuminant
        new_imec_wvs = camera_new_wvs[closestFound_1]
        closestFound_3 = self.get_closest_wavelengths(new_imec_wvs, illuminant_wvs)[:min_count]
        #-------------------------------------------------------------------------------------------
        self.closests_idx = closestFound_1
        self.closests_wvs = cmf_wvs[closestFound_2]
        self.closest_illuminant = self.illuminant[closestFound_3, 1]
        self.cmf_values = self.cmf[closestFound_2, 1:]
        #--------------------------------------------------------------------------------------
        return self.closests_idx,self.cmf_values

    def find_closest_spectra_cmf_illuminant(self,camera_wvs):
        """ This function finds the shared wavelengths among camera SSFs, illuminant,  and the CIE color matching function.
           It returns the closet wavelength index in camera SSFs and nthe closest CIE color matching function values.
           Illuminant is Extended D65 by default.
           Inputs:
           camera_wvs: Wavelengths sampled by camera filters. Interpolation is not considered here.
           Outputs:
           Closest wavelength index: array
           cmf_values: [len(Closest wavelength index), 3]
           """
        illuminant_wvs = self.illuminant[:,0] #
        cmf_wvs = self.cmf[:, 0] # 390 nm -- 830 nm with 1 nm step
        #------ For imec spectra
        closestFound_1 = self.get_closest_wavelengths(cmf_wvs, camera_wvs)
        #------ For cmf functions
        closestFound_2 = self.get_closest_wavelengths(camera_wvs,cmf_wvs)
        min_count = np.min([len(closestFound_1), len(closestFound_2)])
        closestFound_1 = closestFound_1[:min_count]
        new_camera_wvs = camera_wvs[closestFound_1]
        self.closests_wvs = new_camera_wvs
        closestFound_2 = closestFound_2[:min_count]
        self.cmf_values = self.cmf[closestFound_2, 1:]
        #------- For Illuminant
        closestFound_3 = self.get_closest_wavelengths(new_camera_wvs, illuminant_wvs)[:min_count]
        #--------------------------------------------------------------------------------------
        self.closests_idx = closestFound_1
        self.closest_illuminant = self.illuminant[closestFound_3, 1]
        #--------------------------------------------------------------------------------------
        return self.closests_idx,self.cmf_values

    @staticmethod # trick to use numba to speed up class methods, however we can not acces variables defined outside the method using self
    @jit(nopython=True)
    def radiance_to_xyz(radiance_spectra, cmf_values,lines,columns):
        """Convert radiance spectra to an XYZ color space.
        The spectrum must be on the same grid of points as the colour-matchingfunction"""
        x = np.zeros(len(radiance_spectra), dtype=np.float64)
        y = np.zeros(len(radiance_spectra), dtype=np.float64)
        z = np.zeros(len(radiance_spectra), dtype=np.float64)
        cmf_values[:, 0] = cmf_values[:, 0]/np.max(cmf_values[:, 0])
        cmf_values[:, 1] = cmf_values[:, 1]/np.max(cmf_values[:, 1])
        cmf_values[:, 2] = cmf_values[:, 2]/np.max(cmf_values[:, 2])
        for i in range(len(radiance_spectra)):
            x[i] = np.sum((radiance_spectra[i, :].T) * cmf_values[:, 0])  #
            y[i] = np.sum((radiance_spectra[i, :].T) * cmf_values[:, 1])  #
            z[i] = np.sum((radiance_spectra[i, :].T) * cmf_values[:, 2])  #
        # reshape
        X = x.reshape(lines, columns)
        Y = y.reshape(lines, columns)
        Z = z.reshape(lines, columns)
        XYZ_img = np.dstack((X, Y, Z))
        return XYZ_img

    @staticmethod # trick to use numba to speed up class methods, however we can not acces variables defined outside the method using self
    @jit(nopython=True)
    def radiance_to_xyz_v2(radiance_spectra,closest_idx, cmf_values,lines,columns):
        """Convert radiance spectra to an XYZ color space. Radiance is interpolated for better accuracy
        The spectrum must be on the same grid of points as the colour-matchingfunction"""
        x = np.zeros(len(radiance_spectra), dtype=np.float64)
        y = np.zeros(len(radiance_spectra), dtype=np.float64)
        z = np.zeros(len(radiance_spectra), dtype=np.float64)
        cmf_values[:, 0] = cmf_values[:, 0]/np.max(cmf_values[:, 0])
        cmf_values[:, 1] = cmf_values[:, 1]/np.max(cmf_values[:, 1])
        cmf_values[:, 2] = cmf_values[:, 2]/np.max(cmf_values[:, 2])
        for i in range(len(radiance_spectra)):
            this_spectrum = np.asarray(radiance_spectra[i])[closest_idx]
            x[i] = np.sum((this_spectrum.T) * cmf_values[:, 0])  #
            y[i] = np.sum((this_spectrum.T) * cmf_values[:, 1])  #
            z[i] = np.sum((this_spectrum.T) * cmf_values[:, 2])  #
        # reshape
        X = x.reshape(lines, columns)
        Y = y.reshape(lines, columns)
        Z = z.reshape(lines, columns)
        XYZ_img = np.dstack((X, Y, Z))
        return XYZ_img

    @staticmethod
    @jit(nopython=True)
    def reflectance_to_xyz(reflectance_spectra, cmf_values,illumination, normalization_factor, lines, columns):
        """Convert reflectance spectra to XYZ space.
        The spectrum must be on the same grid of points as the colour-matching function"""
        Nb_pixels = len(reflectance_spectra)
        x = np.zeros(Nb_pixels, dtype=np.float64)
        y = np.zeros(Nb_pixels, dtype=np.float64)
        z = np.zeros(Nb_pixels, dtype=np.float64)
        cmf_values[:, 0] = cmf_values[:, 0]/np.sum(cmf_values[:, 0])
        cmf_values[:, 1] = cmf_values[:, 1]/np.sum(cmf_values[:, 1])
        cmf_values[:, 2] = cmf_values[:, 2]/np.sum(cmf_values[:, 2])
        for i in range(Nb_pixels):
            x[i] = np.sum((reflectance_spectra[i, :].T) * cmf_values[:, 0]*illumination) * normalization_factor  #
            y[i] = np.sum((reflectance_spectra[i, :].T) * cmf_values[:, 1]*illumination) * normalization_factor  #
            z[i] = np.sum((reflectance_spectra[i, :].T) * cmf_values[:, 2]*illumination) * normalization_factor  #
        # reshape
        X = x.reshape(lines, columns)
        Y = y.reshape(lines, columns)
        Z = z.reshape(lines, columns)
        XYZ_img = np.dstack((X, Y, Z))
        return XYZ_img

    @staticmethod
    @jit(nopython=True)
    def reflectance_to_xyz_v2(reflectance_spectra,closest_idx, cmf_values, illumination, normalization_factor, lines, columns):
        """ Convert reflectance spectra to XYZ space. Reflectance is interpolated for better accuarcy.
        The spectrum must be on the same grid of points as the colour-matching function"""
        Nb_pixels = len(reflectance_spectra)
        x = np.zeros(Nb_pixels, dtype=np.float64)
        y = np.zeros(Nb_pixels, dtype=np.float64)
        z = np.zeros(Nb_pixels, dtype=np.float64)
        cmf_values[:, 0] = cmf_values[:, 0] / np.sum(cmf_values[:, 0])
        cmf_values[:, 1] = cmf_values[:, 1] / np.sum(cmf_values[:, 1])
        cmf_values[:, 2] = cmf_values[:, 2] / np.sum(cmf_values[:, 2])
        for i in range(Nb_pixels):
            this_spectrum = np.asarray(reflectance_spectra[i])[closest_idx]
            x[i] = np.sum(this_spectrum.T * cmf_values[:, 0] * illumination) * normalization_factor  #
            y[i] = np.sum(this_spectrum.T * cmf_values[:, 1] * illumination) * normalization_factor  #
            z[i] = np.sum(this_spectrum.T * cmf_values[:, 2] * illumination) * normalization_factor  #
        # reshape
        X = x.reshape(lines, columns)
        Y = y.reshape(lines, columns)
        Z = z.reshape(lines, columns)
        XYZ_img = np.dstack((X, Y, Z))
        return XYZ_img

    @staticmethod
    @jit(nopython=True)
    def XYZ_2_sRGB(XYZImage):
        '''
        To simulate sRGB Image, we multiply XYZ Image with static array
        The reference white is D65
         [[3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [0.0556434, -0.2040259,  1.0572252]]
        Check: http://www.brucelindbloom.com/index.html?Eqn_Spect_to_XYZ.html
        '''
        rows = XYZImage.shape[0]
        cols = XYZImage.shape[1]
        bands = XYZImage.shape[2]
        sRGB =  np.empty((rows, cols, bands), dtype=np.float64)
        multiplierMatrix = np.array([[3.2404, -1.5371, -0.4985],
                                     [-0.9692, 1.8760, 0.0415],
                                     [0.0556, -0.2040, 1.0572]])
        for i in range(0, rows):
            for j in range(0, cols):
                X,Y,Z = XYZImage[i,j]
                Xn  = X
                Yn = Y
                Zn = Z
                rgbList = np.dot(multiplierMatrix, np.array([Xn,Yn,Zn]))
                for index, val in enumerate(rgbList):
                    if val<0:
                        rgbList[index]=0
                    if val>1:
                        rgbList[index]=1
                sRGB[i,j]=rgbList
        return sRGB

    @staticmethod
    @jit(nopython=True)
    def XYZ_2_CIERGB(XYZImage):
        '''
        The white reference  is E
        '''
        rows = XYZImage.shape[0]
        cols = XYZImage.shape[1]
        bands = XYZImage.shape[2]
        RGB = np.empty((rows, cols, bands), dtype=np.float64)
        multiplierMatrix = np.array([[2.3706743 , -0.9000405 , -0.4706338],
                                     [-0.5138850 , 1.4253036 , 0.0885814],
                                     [0.0052982 ,-0.0146949 , 1.0093968]])
        for i in range(0, rows):
            for j in range(0, cols):
                X,Y,Z = XYZImage[i,j]
                Xn  = X
                Yn = Y
                Zn = Z
                rgbList = np.dot(multiplierMatrix, np.array([Xn,Yn,Zn]))
                for index, val in enumerate(rgbList):
                    if val<0:
                        rgbList[index]=0
                    if val>1:
                        rgbList[index]=1
                RGB[i,j]=rgbList
        return RGB

    @staticmethod
    @jit(nopython=True)
    def XYZ_2_AdobeRGB(XYZImage):
        '''
        To simulate AdobeRGB, we multiply XYZImage with static array
        The reference white is D65
           [[2.0413690, -0.5649464, -0.3446944],
            [-0.9692660, 1.8760108, 0.0415560],
            [ 0.0134474, -0.1183897, 1.0154096]]

        '''
        rows, cols, bands = XYZImage.shape # bands == 3
        AdobeRGB_img =  np.zeros((rows, cols, bands), dtype=np.float64)
        ### multiplierMatrix in  http://brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
        # D65 AdobeRGB
        multiplierMatrix = np.array([[2.0413690, -0.5649464, -0.3446944],
                                     [-0.9692660, 1.8760108, 0.0415560],
                                     [ 0.0134474, -0.1183897, 1.0154096]])
        for i in range(0, rows):
            for j in range(0, cols):
                X,Y,Z = XYZImage[i,j]
                rgbList = np.dot(multiplierMatrix, np.array([X,Y,Z]))
                for index, val in enumerate(rgbList):
                    if val<0:
                        rgbList[index]=0
                    if val>1:
                        rgbList[index]=1
                AdobeRGB_img[i,j]=rgbList
        return AdobeRGB_img

    @staticmethod
    @jit(nopython=True)
    def XYZ_2_LMS(XYZImage):
        '''
        To simulate LMS_img, we multiply XYZImage with static array
           [[0.4002, 0.7076, -0.0808],
             [-0.2263, 1.1653, 0.0457],
             [0, 0, 0.9182]]
        The reference white is D65
        '''
        rows, cols, bands = XYZImage.shape  # bands == 3
        LMS_img = np.zeros((rows, cols, bands), dtype=np.float64)
        ### multiplierMatrix in https://en.wikipedia.org/wiki/LMS_color_space
        # Normalized to D65
        multiplierMatrix = np.array([[0.4002, 0.7076, -0.0808],
                                     [-0.2263, 1.1653, 0.0457],
                                     [0, 0, 0.9182]])
        for i in range(0, rows):
            for j in range(0, cols):
                X, Y, Z = XYZImage[i, j]
                rgbList = np.dot(multiplierMatrix, np.array([X, Y, Z]))
                for index, val in enumerate(rgbList):
                    if val < 0:
                        rgbList[index] = 0
                    if val > 1:
                        rgbList[index] = 1
                LMS_img[i, j] = rgbList
        return LMS_img

    @staticmethod
    @jit(nopython=True)
    def XYZ_adapt(XYZImage):
        '''
        Von-kries:   [[0.4002400,0.7076000,-0.0808100],
                    [-0.2263000,1.1653200,0.0457000],
                     [0.0000000,0.0000000,0.9182200]]

         Bradford:   [0.8951000,0.2664000,-0.1614000],
                     [-0.7502000,1.7135000,0.0367000],
                     [0.0389000,-0.0685000,1.0296000]]

        '''
        rows = XYZImage.shape[0]
        cols = XYZImage.shape[1]
        bands = XYZImage.shape[2]
        adapted_XYZ = np.empty((rows, cols, bands), dtype=np.float64)
        multiplierMatrix = np.array( [[0.4002400,0.7076000,-0.0808100],
                                     [-0.2263000,1.1653200,0.0457000],
                                     [0.0000000,0.0000000,0.9182200]])
        for i in range(0, rows):
            for j in range(0, cols):
                X, Y, Z = XYZImage[i, j]
                adaptedList = np.dot(multiplierMatrix, np.array([X, Y, Z]))
                for index, val in enumerate(adaptedList):
                    if val<0:
                        adaptedList[index]=0
                    if val>1:
                        adaptedList[index]=1
                adapted_XYZ[i, j] = adaptedList
        return adapted_XYZ

    def setIlluminant(self,closests_wvs, nom, T):
        wvs = closests_wvs
        illuminant = np.zeros(len(wvs))
        if nom == "CN":
            for k in range(len(wvs)):
                illuminant[k] = self.corpsNoir(wvs[k], T)
        if nom == "D65":
            illuminant = self.closest_illuminant # Extended D65 380nm --> 1000nm; 1nm step
        if nom == "A":
            T = 2855.5
            for k in range(len(wvs)):
                for k in range(len(wvs)):
                    illuminant[k] = self.corpsNoir(wvs[k], T)
        if nom == 'E':
            for k in range(len(wvs)):
                illuminant[k] = 1
        return illuminant


    def corpsNoir(self, L, T):
        return 100 * pow(560 / L, 5) * (exp(2.569e4 / T) - 1) / (exp(1.4388e7 / (L * T)) - 1)

    def normalization_factor(self,closests_wvs,illuminant,cmf_values,T):
        yn = 0
        illu_max=0
        for i in range(len(illuminant)):
            if T != 0:
                illu = self.corpsNoir(closests_wvs[i],T)
                if (illu_max<illu):
                    illu_max = illu
            if T == 0:
                illu_max = np.max(illuminant)
        for i in range(len(illuminant)):
            if T != 0:
                illu = self.corpsNoir(closests_wvs[i],T)
            if T == 0:
                illu = illuminant[i]
            # if np.max(illuminant) >1.0:
            #     yn += cmf_values[i, 1]*(illu/illu_max) # Multiply with normalized illuminant
            # else:
            yn += cmf_values[i, 1]*illu
        self.K = 100 / yn


    @staticmethod
    @jit(nopython=True)
    def apply_srgb_gamma(rgb):
        gamma_rgb = np.empty((rgb.shape[0], rgb.shape[1], rgb.shape[2]), dtype=np.float64)
        #RGB = np.array(3,dtype=np.float64)
        for i in range(0, rgb.shape[0]):
            for j in range(0, rgb.shape[1]):
                RGB = rgb[i, j]
                for idx in range(3):
                    if (RGB[idx] <0): RGB[idx] = 0
                    if (RGB[idx] >0 and RGB[idx] <= 0.0031308 ) :
                        RGB[idx] = (RGB[idx]* 12.92)
                    else:
                        RGB[idx] = (1.055 * (RGB[idx]**(1 / 2.4)) - 0.055)
                gamma_rgb[i,j] = RGB
        return gamma_rgb

    @staticmethod
    @jit(nopython=True)
    def apply_adobe_gamma(rgb):
            gamma_rgb = np.empty((rgb.shape[0], rgb.shape[1], rgb.shape[2]), dtype=np.float64)
            #RGB = np.array(3,dtype=np.float64)
            for i in range(0, rgb.shape[0]):
                for j in range(0, rgb.shape[1]):
                    RGB = rgb[i, j]
                    for idx in range(3):
                        if (RGB[idx] < 0):
                            RGB[idx] = 0
                        RGB[idx] = RGB[idx]**(1.0 / 2.19921875)
                    gamma_rgb[i,j] = RGB
            return gamma_rgb

    def adjust_gamma(self,image, gamma=1.0):
        """ Function from https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction
        """
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    def normalize8(self,I):
      mn = I.min()
      mx = I.max()
      mx -= mn
      I = ((I - mn)/mx) * 255
      return I.astype(np.uint8)

    # @jit(nopython=True)
    def get_closest_wavelengths(self,sampling_w, scan):
        # get the closest wavelengths in scan array to match those in sampling_w array
        dist = np.abs(sampling_w[:, np.newaxis] - scan)
        potentialClosest = dist.argmin(axis=1)
        closest_bands, counts = np.unique(potentialClosest, return_counts=True)
        return closest_bands

    def get_closest_radiance_spectra_to_rgbssfs(self,envi_imec, rgb_ssfs):
        ignored_bands_idx1 = [0, 1, 2, 3, 4, 5, 6, 7, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70] # bandes redondantes Snapscan
        ignored_bands_idx2 = [x for x in range(169, 192)]# bandes redondantes Snapscan
        all_ignored_bands = ignored_bands_idx1 + ignored_bands_idx2  #
        used_bands = [k for k in range(192) if k not in all_ignored_bands]
        imec_wvs = envi_imec.metadata['wavelength']
        imec_wvs = [np.round(np.float(imec_wvs[i]), 1) for i in range(len(imec_wvs))]
        imec_wvs = np.asarray(imec_wvs).astype(np.float32)
        imec_wvs = imec_wvs[used_bands]
        rgb_wvs = rgb_ssfs[:, 0]
        closest_bands_imec = self.get_closest_wavelengths(rgb_wvs, imec_wvs)  # imec side
        closest_bands_rgb = self.get_closest_wavelengths(imec_wvs, rgb_wvs)  # rgb side
        min_count = np.min((len(closest_bands_imec), len(closest_bands_rgb)))  # make even closest indexes
        closest_bands_imec = closest_bands_imec[:min_count]  # imec side
        closest_bands_rgb = closest_bands_rgb[:min_count]  # rgb side
        cube = envi_imec.asarray()
        cube = cube[:,:,used_bands]
        closest_radiance_spectra = cube[:,:,closest_bands_imec]
        lines,columns, K = closest_radiance_spectra.shape
        closest_radiance_spectra = closest_radiance_spectra.reshape((lines * columns), K)
        return closest_radiance_spectra,closest_bands_rgb

    def get_closest_reflectance_spectra_to_rgbssfs_and_illuminant(self,camera_wvs, rgb_ssfs,illuminant):
        rgb_wvs = rgb_ssfs[:, 0]
        illuminant_wvs = illuminant[:, 0]
        # IMEC & RGB side
        closest_idx_bands_camera = self.get_closest_wavelengths(rgb_wvs, camera_wvs)  # imec side
        closest_bands_rgb = self.get_closest_wavelengths(camera_wvs, rgb_wvs)  # rgb side
        min_count = np.min([len(closest_idx_bands_camera), len(closest_bands_rgb)])  # make it even
        closest_idx_bands_camera = closest_idx_bands_camera[:min_count]  # imec side
        closest_idx_bands_rgb = closest_bands_rgb[:min_count]  # rgb side
        new_camera_wvs = camera_wvs[closest_idx_bands_camera]
        # Illuminant side
        closest_idx_bands_illuminant = self.get_closest_wavelengths(new_camera_wvs, illuminant_wvs)  # illuminant side
        min_count2 = np.min([len(closest_idx_bands_camera), len(closest_idx_bands_rgb),len(closest_idx_bands_illuminant)])  # make it even
        # closest_bands_imec = closest_bands_imec[:min_count2]  # imec side
        # closest_bands_rgb = closest_bands_rgb[:min_count2]  # rgb side
        closest_idx_bands_illuminant = closest_idx_bands_illuminant[:min_count2]
        #----------------------------------------------------------------------------------------------------------------
        # closest_reflectance_spectra = cube[:, :, closest_bands_camera]
        # lines, columns, K = closest_reflectance_spectra.shape
        # closest_reflectance_spectra = closest_reflectance_spectra.reshape((lines * columns), K)
        return closest_idx_bands_camera,closest_idx_bands_rgb,closest_idx_bands_illuminant

    def get_closest_reflectance_spectra_to_targetssfs(self, cube,camera_wvs, target_ssfs):
        rgb_wvs = target_ssfs[:, 0]
        # IMEC & RGB side
        closest_bands_imec = self.get_closest_wavelengths(rgb_wvs, camera_wvs)  # imec side
        closest_bands_rgb = self.get_closest_wavelengths(camera_wvs, rgb_wvs)  # rgb side
        min_count = np.min([len(closest_bands_camera), len(closest_bands_rgb)])  # make it even
        closest_bands_imec = closest_bands_imec[:min_count]  # imec side
        closest_idx_target_bands = closest_bands_rgb[:min_count]  # rgb side
        closest_reflectance_spectra = cube[:, :, closest_bands_imec]
        lines, columns, K = closest_reflectance_spectra.shape
        closest_reflectance_spectra = closest_reflectance_spectra.reshape((lines * columns), K)
        return closest_reflectance_spectra, closest_idx_target_bands

    @staticmethod
    @jit(nopython=True)
    def Spectralreflectance_to_RGBNIRreflectance(closest_reflectance_spectra, rgb_nir_ssfs, closest_bands_rgb_nir, lines, columns):
        # ---------------------------------------------------------------------------------------------------------------
        closest_blue_ssfs = rgb_nir_ssfs[closest_bands_rgb_nir, 1]
        closest_green_ssfs = rgb_nir_ssfs[closest_bands_rgb_nir, 2]
        closest_red_ssfs = rgb_nir_ssfs[closest_bands_rgb_nir, 3]
        closest_nir_ssfs = rgb_nir_ssfs[closest_bands_rgb_nir, 4]
        # ---------------------------------------------------------------------------------------------------------------
        B = np.zeros(len(closest_reflectance_spectra), dtype=np.float32)
        G = np.zeros(len(closest_reflectance_spectra), dtype=np.float32)
        R = np.zeros(len(closest_reflectance_spectra), dtype=np.float32)
        NIR = np.zeros(len(closest_reflectance_spectra), dtype=np.float32)
        #Normalize SSFS
        closest_blue_ssfs = closest_blue_ssfs / np.sum(closest_blue_ssfs)
        closest_green_ssfs = closest_green_ssfs / np.sum(closest_green_ssfs)
        closest_red_ssfs = closest_red_ssfs / np.sum(closest_red_ssfs)
        closest_nir_ssfs = closest_nir_ssfs / np.sum(closest_nir_ssfs)

        for i in range(len(closest_reflectance_spectra)):
            B[i] = np.sum((closest_reflectance_spectra[i, :].T) * closest_blue_ssfs)
            G[i] = np.sum((closest_reflectance_spectra[i, :].T) * closest_green_ssfs)
            R[i] = np.sum((closest_reflectance_spectra[i, :].T) * closest_red_ssfs)
            NIR[i] = np.sum((closest_reflectance_spectra[i, :].T) * closest_nir_ssfs)
        # Reshape to 3d matrix
        B = B.reshape(lines, columns)
        G = G.reshape(lines, columns)
        R = R.reshape(lines, columns)
        NIR = NIR.reshape(lines, columns)
        # Normalize each channel
        # norm_value = np.max(np.asarray([np.max(R),np.max(G),np.max(B)]))
        # R = R / np.max(G)
        # G = G /  np.max(G)
        # B = B /  np.max(G)
        BGR_NIR = np.dstack((B, G, R,NIR))
        return BGR_NIR


    @staticmethod  # Trick to use numba in a Class
    @jit(nopython=True)
    def Spectralreflectance_to_RGBNIR_radiance(closest_reflectance_spectra, rgb_nir_ssfs,illuminant,closest_bands_rgb_nir,closest_bands_illuminant,lines,columns):
        closest_blue_ssfs = rgb_nir_ssfs[closest_bands_rgb_nir, 1]
        closest_green_ssfs = rgb_nir_ssfs[closest_bands_rgb_nir, 2]
        closest_red_ssfs = rgb_nir_ssfs[closest_bands_rgb_nir, 3]
        closest_nir_ssfs = rgb_nir_ssfs[closest_bands_rgb_nir, 4]
        closest_illuminant = illuminant[closest_bands_illuminant, 1]
        # ---------------------------------------------------------------------------------------------------------------
        B = np.zeros(len(closest_reflectance_spectra), dtype=np.float32)
        G = np.zeros(len(closest_reflectance_spectra), dtype=np.float32)
        R = np.zeros(len(closest_reflectance_spectra), dtype=np.float32)
        NIR = np.zeros(len(closest_reflectance_spectra), dtype=np.float32)
        # Normalize SSFS
        closest_blue_ssfs = closest_blue_ssfs / np.sum(closest_blue_ssfs)
        closest_green_ssfs = closest_green_ssfs / np.sum(closest_green_ssfs)
        closest_red_ssfs = closest_red_ssfs / np.sum(closest_red_ssfs)
        closest_nir_ssfs = closest_nir_ssfs / np.sum(closest_nir_ssfs)
        #----------------------------------------------------------------------------------------------------------------
        for i in range(len(closest_reflectance_spectra)):
            B[i] = np.sum((closest_reflectance_spectra[i, :].T) * closest_blue_ssfs*closest_illuminant)  # / Blue_channel_normalization
            G[i] = np.sum((closest_reflectance_spectra[i, :].T) * closest_green_ssfs*closest_illuminant)  # / Green_channel_normalization
            R[i] = np.sum((closest_reflectance_spectra[i, :].T) * closest_red_ssfs*closest_illuminant)  # / Red_channel_normalization
            NIR[i] = np.sum((closest_reflectance_spectra[i, :].T) * closest_nir_ssfs*closest_illuminant)  # / Red_channel_normalization
        # Reshape to 3d matrix
        R = R.reshape(lines, columns)
        G = G.reshape(lines, columns)
        B = B.reshape(lines, columns)
        NIR = NIR.reshape(lines, columns)
        RGB_NIR = np.dstack((B, G, R,NIR))
        return RGB_NIR

    @staticmethod
    @jit(nopython=True)
    def Spectralradiance_to_RGB(closest_radiance_spectra,rgb_ssfs,closest_bands_rgb,lines,columns):
        #---------------------------------------------------------------------------------------------------------------
        # rgb_ssfs[:,1:]  =   rgb_ssfs[:,1:] / np.max(np.sum(rgb_ssfs[:,1:], 0)) # Normalize SSFS
        closest_blue_ssfs = rgb_ssfs[closest_bands_rgb, 1]
        closest_green_ssfs = rgb_ssfs[closest_bands_rgb, 2]
        closest_red_ssfs = rgb_ssfs[closest_bands_rgb, 3]
        #---------------------------------------------------------------------------------------------------------------
        B = np.zeros(len(closest_radiance_spectra), dtype=np.float32)
        G = np.zeros(len(closest_radiance_spectra), dtype=np.float32)
        R = np.zeros(len(closest_radiance_spectra), dtype=np.float32)
        closest_blue_ssfs = closest_blue_ssfs/np.sum(closest_blue_ssfs)
        closest_green_ssfs = closest_green_ssfs/np.sum(closest_green_ssfs)
        closest_red_ssfs = closest_red_ssfs/np.sum(closest_red_ssfs)
        for i in range(len(closest_radiance_spectra)):
            B[i] = np.sum((closest_radiance_spectra[i, :].T) * closest_blue_ssfs)  #/ Blue_channel_normalization
            G[i] = np.sum((closest_radiance_spectra[i, :].T) * closest_green_ssfs) #/ Green_channel_normalization
            R[i] = np.sum((closest_radiance_spectra[i, :].T) * closest_red_ssfs)  #/ Red_channel_normalization
        # norm_value = np.max(np.asarray([np.max(R),np.max(G),np.max(B)]))
        # Reshape to 3d matrix
        B = (B.reshape(lines, columns))
        G = (G.reshape(lines, columns))
        R = (R.reshape(lines, columns))
        # Normalize each channel
        # norm_value = np.max(np.asarray([np.max(R),np.max(G),np.max(B)]))
        # R = R / np.max(G)
        # G = G /  np.max(G)
        # B = B /  np.max(G)
        RGB = np.dstack((R, G, B))
        return RGB

    @staticmethod
    @jit(nopython=True)
    def Spectralreflectance_to_RGB(reflectance_spectra, rgb_ssfs,illuminant,closest_idx_bands_camera,closest_idx_bands_rgb,closest_idx_bands_illuminant,lines,columns):
        # closest_cube = cube[:, :, closest_bands_imec]
        # lines,columns, K = closest_cube.shape
        # closest_spectra = closest_cube.reshape((lines * columns),K)
        # -------------------------Normalize SSFS and illuminant between 0 and 1 ------------------------------------------------
        rgb_ssfs[:, 1] = rgb_ssfs[:, 1] / np.sum(rgb_ssfs[:, 1])
        rgb_ssfs[:, 2] = rgb_ssfs[:, 2] / np.sum(rgb_ssfs[:, 2])
        rgb_ssfs[:, 3] = rgb_ssfs[:, 3] / np.sum(rgb_ssfs[:, 3])
        #-----------------------------------------------------------------------------------------------------------
        closest_blue_ssfs = rgb_ssfs[closest_idx_bands_rgb, 1]
        closest_green_ssfs = rgb_ssfs[closest_idx_bands_rgb, 2]
        closest_red_ssfs = rgb_ssfs[closest_idx_bands_rgb, 3]
        closest_illuminant = illuminant[closest_idx_bands_illuminant, 1] # illuminanat must be already normalized between 0 and 1
        #----------------------------------------------------------------------------------------------------------------
        B = np.zeros(len(reflectance_spectra), dtype=np.float32)
        G = np.zeros(len(reflectance_spectra), dtype=np.float32)
        R = np.zeros(len(reflectance_spectra), dtype=np.float32)
        Blue_channel_normalization = np.sum(closest_blue_ssfs*closest_illuminant)
        Green_channel_normalization = np.sum(closest_green_ssfs*closest_illuminant)
        Red_channel_normalization = np.sum(closest_red_ssfs*closest_illuminant)
        for i in range(len(reflectance_spectra)):
            this_spectrum = reflectance_spectra[i][closest_idx_bands_camera]
            B[i] = np.sum((this_spectrum.T) * closest_blue_ssfs*closest_illuminant) / Blue_channel_normalization
            G[i] = np.sum((this_spectrum.T) * closest_green_ssfs*closest_illuminant) / Green_channel_normalization
            R[i] = np.sum((this_spectrum.T) * closest_red_ssfs*closest_illuminant) / Red_channel_normalization
        # Reshape to 3d matrix
        R = R.reshape(lines, columns)
        G = G.reshape(lines, columns)
        B = B.reshape(lines, columns)
        # Normalize each channel with respect to its max value
        # R = R/np.max(R)
        # G = G/np.max(G)
        # B = B/np.max(B)
        #
        RGB = np.dstack((R, G, B))
        return RGB

    @staticmethod
    @jit(nopython=True)
    def simulate_NIR_band_from_reflectance(cube, imec_ssfs,illuminant):
        lines, columns, _ = cube.shape
        imec_NIR_wvs = imec_ssfs['virtual_centers'][81:]
        illuminant_wvs = illuminant[:,0]
        # illuminant side
        dist = np.abs(imec_NIR_wvs[:,np.newaxis], illuminant_wvs)
        temp1 = dist.argmin(axis=1)
        closest_bands_illuminant, _ = np.unique(temp1, return_counts=True)
        # imec side
        dist2 = np.abs(illuminant_wvs[:,np.newaxis],imec_NIR_wvs)
        temp2 = dist2.argmin(axis=1)
        closest_NIR_bands_imec, _ = np.unique(temp2, return_counts=True)
        # closest_bands_illuminant = self.get_closest_wavelengths(imec_NIR_wvs, illuminant_wvs)
        # closest_NIR_bands_imec = self.get_closest_wavelengths(illuminant_wvs,imec_NIR_wvs)
        min_count = np.min((len(closest_bands_illuminant),len(closest_NIR_bands)))
        closest_bands_illuminant = closest_bands_illuminant[:min_count]  # illuminant side
        closest_NIR_bands_imec = closest_NIR_bands_imec[:min_count]  # imec side
        #-------- get data according to closest wavelength indexes
        closest_NIR_illuminant = illuminant[closest_bands_illuminant,:]
        closest_imec_NIR_ssfs = imec_ssfs['responses'][closest_NIR_bands_imec, :]  # SSFS at 721 nm -- 901.7 nm
        closest_NIR_reflectance = cube[:, :, closest_NIR_bands_imec]
        closest_NIR_reflectance = closest_NIR_reflectance.reshape((lines * columns), len(closest_NIR_bands_imec))
        # ------------------------------------------------------------------------------------
        Max_SSFS_responses = np.asarray([np.max(closest_imec_NIR_ssfs[i,:]) for i in range(len(closest_imec_NIR_ssfs))])  # consider the SSFS as delta dirac functions
        NIR_channel = np.zeros(len(closest_NIR_reflectance), dtype=np.float32)
        for i in range(len(closest_NIR_reflectance)):
            NIR_channel[i] = (np.sum((closest_NIR_reflectance[i, :].T) * Max_SSFS_responses * closest_NIR_illuminant))/ (np.sum(Max_SSFS_responses*closest_NIR_illuminant))
        NIR_channel = NIR_channel.reshape(lines, columns)
        return NIR_channel

    @staticmethod
    @jit(nopython=True)
    def simulate_NIR_band_from_radiance(closest_NIR_radiance,imec_NIR_ssfs,lines,columns):
        # ignored_bands_idx1 = [0, 56, 57, 58, 59, 60, 61, 62, 63, 64]
        # ignored_bands_idx2 = [x for x in range(172, 192)]
        # all_ignored_bands = ignored_bands_idx1 + ignored_bands_idx2  #
        # used_bands = [k for k in range(192) if k not in all_ignored_bands]
        # NIR_idx = used_bands[84:]  # 721 nm -- 901.7 nm
        # imec_NIR_ssfs = imec_ssfs['responses'][NIR_idx, :]  # SSFS at 721 nm -- 901.7 nm
        # closest_NIR_radiance = cube[:, :, NIR_idx]
        # lines, columns, K = closest_NIR_radiance_cube.shape
        # ------------------------------------------------------------------------------------
        Max_SSFS_responses = np.asarray([np.max(imec_NIR_ssfs[i, :]) for i in range(len(imec_NIR_ssfs))])  # Max because we consider the SSFS as delta dirac functions
        NIR_channel = np.zeros(len(closest_NIR_radiance), dtype=np.float32)
        for i in range(len(closest_NIR_radiance)):
            NIR_channel[i] = np.sum((closest_NIR_radiance[i, :].T) * Max_SSFS_responses) / np.sum(Max_SSFS_responses)  # Normalize by the sum of all NIR SSFS
        NIR_channel = NIR_channel.reshape(lines, columns)
        # NIR_channel = NIR_channel/np.max(NIR_channel) # normalize NIR channel between 0 and 1
        return NIR_channel

    def get_closest_NIR_radiance(self,cube,imec_ssfs):
        ignored_bands_idx1 = [0, 56, 57, 58, 59, 60, 61, 62, 63, 64]
        ignored_bands_idx2 = [x for x in range(172, 192)]
        all_ignored_bands = ignored_bands_idx1 + ignored_bands_idx2  #
        used_bands = [k for k in range(192) if k not in all_ignored_bands]
        NIR_idx = used_bands[84:]  # 721 nm -- 901.7 nm
        imec_NIR_ssfs = imec_ssfs['responses'][NIR_idx, :]  # SSFS at 721 nm -- 901.7 nm
        closest_NIR_radiance= cube[:, :, NIR_idx]
        lines, columns, K = closest_NIR_radiance.shape
        closest_NIR_radiance =  closest_NIR_radiance.reshape((lines * columns), K)
        return closest_NIR_radiance,imec_NIR_ssfs

    def interpolate_ssfs(self,ssfs_values,wvs_ssfs,step=1):
        ### Used for RGB-NIR SSFS
        length_ssf, Nb_ssfs  = ssfs_values.shape
        interpoted_ssfs = []
        min_wv = np.min(wvs_ssfs)
        max_wv = np.max(wvs_ssfs)
        new_wvs = np.arange(min_wv,max_wv+step,step)
        for i in range(Nb_ssfs):
            interp_this_ssf = interp1d(wvs_ssfs,ssfs_values[:,i],'linear')
            interpoted_ssfs.append(interp1d(new_wvs,interp_this_ssf(new_wvs)).y)
        interpoted_ssfs = np.asarray(interpoted_ssfs) #
        BGR_NIR_ssfs = np.zeros(interpoted_ssfs.shape,dtype=interpoted_ssfs.dtype)
        BGR_NIR_ssfs[0,:] = interpoted_ssfs[2,:] ## -----> Make ssfs ordered from visible (BGR) to NIR band
        BGR_NIR_ssfs[1,:] = interpoted_ssfs[1,:]
        BGR_NIR_ssfs[2,:] = interpoted_ssfs[0,:]
        BGR_NIR_ssfs[3,:] = interpoted_ssfs[3,:]
        BGR_NIR_ssfs = np.concatenate((new_wvs.reshape(len(new_wvs),1),BGR_NIR_ssfs.T),axis=1)
        ### BGR_NIR_ssfs shape is  NB wavelengths * 5
        return BGR_NIR_ssfs

    def interpolate_spectrum(self,spectrum_values,wvs,step=1):
        min_wv = np.min(wvs)
        max_wv = np.max(wvs)
        new_wvs = np.arange(min_wv, max_wv, step)
        interpoted_spectrum = np.interp(new_wvs, wvs, spectrum_values)
        # interp_this_spectrum = interp1d(wvs, spectrum_values, 'linear')
        # interpoted_spectrum = interp1d(new_wvs, interp_this_spectrum(new_wvs)).y
        return np.asarray(interpoted_spectrum),new_wvs

    @staticmethod
    @jit(nopython=True)
    def interpolate_spectra(spectra,wvs,step=1):
        min_wv = np.min(wvs)
        max_wv = np.max(wvs)
        new_wvs = np.arange(min_wv, max_wv, step)
        Nb_samples = len(spectra)
        interpolated_spectra = []
        for i in range(Nb_samples):
            s = np.interp(new_wvs, wvs, spectra[i])
            interpolated_spectra.append(s)
        return interpolated_spectra

    def parallel_interpolate(self,spectra,wvs,step=1):
        executor = Parallel(n_jobs=12, backend='multiprocessing')
        min_wv = np.min(wvs)
        max_wv = np.max(wvs)
        new_wvs = np.arange(min_wv, max_wv, step)
        Nb_samples = len(spectra)
        interpolated_spectra = executor(delayed(np.interp)(new_wvs, wvs, spectra[i]) for i in range(Nb_samples))
        return interpolated_spectra

    # def interp(self,x):
    #     min_wv = 475
    #     max_wv = 901
    #     new_wvs = np.arange(min_wv, max_wv, 1)
    #     np.interp(np.interp(new_wvs, wvs, x))
    #
    # def parallel_predict(self,x):
    #     import multiprocessing as mp
    #     pool = mp.Pool(12)
    #     results = pool.map(np.interp, [spectra.reshape(1,-1) for spectra in x]) # x[s,:].reshape(1,-1)
    #     pool.close()
    #     pool.terminate()
    #     elapsed_time = (time.time()-start)
    #     print('\nElapsed time = '+str(elapsed_time)+' s\n')
    #     return results, elapsed_time