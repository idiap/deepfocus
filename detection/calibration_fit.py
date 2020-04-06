'''
Code for the PyTorch implementation of
"DeepFocus: a Few-Shot Microscope Slide Auto-Focus using a Sample-invariant CNN-based Sharpness Function"

Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
Written by Adrian Shajkofci <adrian.shajkofci@idiap.ch>,
All rights reserved.

This file is part of DeepFocus.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. Neither the name of mosquitto nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
'''

import numpy as np
from toolbox import gaussian_kernel
from toolbox import convolve
from toolbox import rand_int
from toolbox import center_crop_pixel, scale
from toolbox import noisy
from toolbox import random_crop
from toolbox import plot_images
from toolbox import pickle_save
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import random
import scipy
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.constants import golden_ratio
from numba import jit
from lmfit.models import GaussianModel, LinearModel,  MoffatModel
from lmfit.model import ModelResult
from scipy import optimize
import pywt
from skimage.filters import sobel, rank, laplace
from skimage.morphology import square
from skimage.transform import resize
from unet_detector import *
from sklearn.cluster import MeanShift
import logging

logging.basicConfig(
    format="%(asctime)s [FIT] %(message)s".format(00),
    handlers=[
        logging.FileHandler("output_log_{}.log".format(00)),
        logging.StreamHandler()
    ])

log = logging.getLogger('')
log.setLevel(logging.INFO)

learn = get_learner()

class Calibration:
    
    def __init__(self):
        self.gaussian2_center = None
        self.peak_center = None
        self.gaussian2_sigma = None
        self.peak_sigma = None
        self.c = None
        self.calibration = None
        self.params = None
        self.focus_map_1d = None
        self.z = None
        self.mode = "model"

    def load(self, input):
        self.peak_center = input[0]
        self.peak_sigma = input[1]
        self.peak_beta = input[8]
        self.gaussian2_center = input[2]
        self.gaussian2_sigma = input[3]
        self.c = input[4]
        self.params = input[5]
        self.focus_map_1d = input[6]
        self.z = input[7]
        #self.params['peak_center'].value = 0
        #self.params['gaussian2_center'].value = 0
        self.calibration = ModelResult(get_model(), self.params)

    def get_width(self):
        # for moffat function
        return 2.0 * self.peak_sigma * np.sqrt(2**(1/self.peak_beta) - 1)

    def save(self):
        return [self.peak_center, self.peak_sigma, self.gaussian2_center, self.gaussian2_sigma, self.c, self.params, self.focus_map_1d, self.z, self.peak_beta]

    def eval(self, x, mode = None):
        if self.mode == 'model' and (mode is None or mode == 'model'):
            return self.calibration.eval(self.calibration.params, x=x+self.peak_center*1.0) # I don't know if we should center it here and remove the calculations about that elsewhere
        else:
            inter_func = interp1d(self.z, self.focus_map_1d[:], kind='linear', bounds_error=False,
                                  fill_value=(self.focus_map_1d[0], self.focus_map_1d[-1]))

            return inter_func(x + self.peak_center*1.0)#inter_func(x + self.z[np.argmin(self.focus_map_1d)])

class ImageStack:

    def _init_(self):
        self.image_stack = None
        self.width = None
        self.height = None
        self.z_positions = None
        self.focus_map = None
        self.downsample = None

    def add_image_to_stack(self, image, z_position, update_focus_map=True):
        self.image_stack = np.concatenate((self.image_stack, image), axis=0)
        self.z_positions = np.concatenate((self.z_positions, [z_position]), axis=0)
        log.info('z positions shape {} (last z = {})'.format(self.z_positions.shape, self.z_positions[-1]))

        if update_focus_map:
            log.info('Set focus map from new image ...')
            focus_map = get_focus_map_from_stack(image, downsample=self.downsample, num_iterations=1, gain_sigma=0)
            log.info('Format = {} Score mean of all map = {}'.format( focus_map.shape, focus_map.mean()))
            self.focus_map = np.concatenate((self.focus_map, focus_map), axis=0)


    def set_image_stack(self, image_stack, width, height, downsample, z_positions):
        self.image_stack = image_stack
        self.width = width
        self.height = height
        self.z_positions = np.atleast_1d(np.asarray(z_positions))
        log.info('z position shape {}'.format(self.z_positions.shape))
        self.downsample = downsample
        log.info('Set focus map from stack ({} images)...'.format(self.get_num_z()))
        self.focus_map = get_focus_map_from_stack(self.image_stack, downsample=downsample, num_iterations=1, gain_sigma=0)

        
    def get_max_z(self):
        return np.max(self.z_positions)
    
    def get_min_z(self):
        return np.min(self.z_positions)
    
    def get_num_z(self):
        return self.image_stack.shape[0]
    
    def get_image_stack(self):
        return self.image_stack, np.linspace(self.get_min_z(), self.get_max_z(), self.get_num_z()), self.focus_map

    def get_focus_map(self):
        return self.focus_map

    def get_z_positions(self):
        return self.z_positions

    def get_resized_focus_map(self):
        return np.asarray([resize(self.focus_map[i], (self.width, self.height), order=0) for i in range(self.get_num_z())])

    def is_in_roi(self, roi, _x, _y):
        coeff_x = self.width / (self.focus_map.shape[0]+1)
        roi_transformed = roi // coeff_x
        #print('roi trans {} x {} y {}'.format(roi_transformed, _x, _y))
        return (_x >= roi_transformed[0] and _x <= roi_transformed[2]) and (_y >= roi_transformed[1] and _y <=roi_transformed[3])

    def print_focus_map(self):
        plot_images(self.focus_map)

    def print_data(self):
        plot_images(self.image_stack)




def CMSL(img, window_size):
    """
    Contrast Measure based on squared Laplacian according to
    'Robust Automatic Focus Algorithm for Low Contrast Images
    Using a New Contrast Measure'
    by Xu et Al. doi:10.3390/s110908281
    window: window size= window X window"""
    ky1 = np.array(([0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]))
    ky2 = np.array(([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]))
    kx1 = np.array(([0.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]))
    kx2 = np.array(([0.0, 0.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 0.0]))

    dst = np.abs(scipy.ndimage.filters.convolve(img, kx1, mode='reflect')) + np.abs(scipy.ndimage.filters.convolve(img, kx2, mode='reflect'))\
             + np.abs(scipy.ndimage.filters.convolve(img, ky1, mode='reflect')) + np.abs(scipy.ndimage.filters.convolve(img, ky2, mode='reflect'))

    return rank.mean(dst//dst.max(), selem=square(window_size))


def wavelet(img):
    #http://tonghanghang.org/pdfs/icme04_blur.pdf
    c = pywt.wavedec2(img, 'db2', mode='periodization', level=1)
    # normalize each coefficient array independently for better visibility
    d = np.sqrt((c[1][0]/np.abs(c[1][0]).max())**2 + (c[1][1]/np.abs(c[1][1]).max())**2 + (c[1][2]/np.abs(c[1][2]).max())**2)
    return resize(d, (img.shape[0], img.shape[1]))

def wavelet_liebling(img):
    #https://www.osapublishing.org/josaa/abstract.cfm?uri=josaa-21-12-2424
    c = pywt.wavedec2(img, 'db2', mode='symmetric', level=pywt.dwtn_max_level(img.shape, 'db2'))
    c, slices = pywt.coeffs_to_array(c)
    c = np.abs(c.flatten())
    c /= c.sum()
    c = np.sort(c)[::-1]
    _sum = 0
    for i in range(c.shape[0]):
        _sum += c[i]
        if _sum > 0.95:
            break
    #i = i/c.shape[0]
    i = float(i)
    return np.ones((img.shape[0], img.shape[1]))*i

def LAPV(img):
    """Implements the Variance of Laplacian (LAP4) focus measure
    operator. Measures the amount of edges present in the image.
    :param img: the image the measure is applied to
    :type img: numpy.ndarray
    :returns: numpy.float32 -- the degree of focus
    """
    return np.std(laplace(img)) ** 2

def SML(img, window_size, threshold):
    """
    Sum of modified Laplacian according to
    'Depth Map Estimation Using Multi-Focus Imaging'
    by Mendapara
    """
    # kernels in x- and y -direction for Laplacian
    ky = np.array(([0.0, -1.0, 0.0], [0.0, 2.0, 0.0], [0.0, -1.0, 0.0]))
    kx = np.array(([0.0, 0.0, 0.0], [-1.0, 2.0, -1.0], [0.0, 0.0, 0.0]))

    dst = np.abs(scipy.ndimage.filters.convolve(img, ky, mode='reflect')) + np.abs(scipy.ndimage.filters.convolve(img, kx, mode='reflect'))

    # sum up all values that are bigger than threshold in window
    dst = np.clip(dst, threshold, dst.max())
    # return thresholded image summed up in each window:
    return dst



def tenengrad1(img, window_size, threshold):
    """
    Tenengrad2b: squared gradient absolute thresholded and
    summed up in each window
    according to
    'Autofocusing Algorithm Selection in Computer Microscopy'
    by Sun et Al.
    """
    # calculate gradient magnitude:

    dst = sobel(img)
    dst = np.clip(dst, threshold, dst.max())
    # return thresholded image summed up in each window:
    return rank.mean(dst, selem=square(window_size))


def get_hpf_image(image=None, size=128, method = 'hpf'):

    if method is 'hpf':
        kernel = [1, 0, -1]
        output = scipy.ndimage.filters.convolve1d(image, kernel, mode='reflect')**2
    elif method is 'tenengrad1':
        output = tenengrad1(image, 7, 0)
    elif method is 'CMSL':
        output = CMSL(image, 3)
    elif method is 'wavelet':
        output = wavelet(image)
    elif method is 'SML':
        output = 1/SML(image,4,0)
    elif method is 'wavelet_liebling':
        output = wavelet_liebling(image)
    else:
        output = image

    x = size
    y = size
    im = output[0:output.shape[0]//size * size, 0:output.shape[1]//size * size]

    tile_dataset = []
    y_size = 0
    i = 0

    while x <= im.shape[0]:
        x_size = 0
        while y <= im.shape[1]:
            a = im[x - size:x, y - size:y]
            score = a[:].sum().sum()
            if method == 'LAPV':
                score = 1/LAPV(a)
            #elif method == 'wavelet_liebling':
            #    score = wavelet_liebling(a)
            tile_dataset.append(score)
            y += size
            x_size += 1
            i += 1
        y = size
        y_size += 1
        x += size

    final = np.reshape(tile_dataset, (x_size, y_size))
    #plt.imshow(final)
    #plt.figure()
    #plt.imshow(output)
    return final



def get_synthetic_image(flip = False):
    image = io.imread('data/texture.png', as_gray=True)
    square_size = np.min(image.shape) // 2
    image = random_crop(image, square_size)
    #image = image [:square_size, :square_size]
    if flip:
        image = np.flip(image, axis=0)
    return image


def blur_image_stack(image, num_z, min_z_calib = None, max_z_calib = None, z_focus=0, noise_sigma=0.0, input_noise = 0.0, width_coeff = 1.0):
    im_size = image.shape[0]
    #kernels = np.zeros((im_size, num_z, num_z))
    log.info('Generating a blurred stack from {} to {} with {} images and centered at z={}.'.format(min_z_calib, max_z_calib, num_z, z_focus))
    kernels = []
    z_coeff = 1.7*width_coeff
    noise = np.random.normal(0, noise_sigma, num_z)
    kernel_size = im_size // 2 + 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    if num_z == 1:
        dist = abs(float(max_z_calib-z_focus) * z_coeff)
        dist += noise[0]
        kernels.append(gaussian_kernel(kernel_size, fwhmx=dist, fwhmy=dist) * (im_size ** 2))
    else:
        z_list = np.linspace (min_z_calib-z_focus+1, max_z_calib-z_focus, num_z).tolist()
        for z_idx, z in enumerate(z_list):
            if not isinstance(z, float):
                z = z[0]
            dist = np.abs(z*z_coeff)
            dist += noise[z_idx]
            kernels.append(gaussian_kernel(kernel_size, fwhmx=dist, fwhmy=dist) * (im_size ** 2))
    #plot_images(kernels)

    all_images = []
    i = 0
    uni = np.random.uniform(input_noise // 2, input_noise * 2, len(kernels))
    for kernel in kernels:
        c = convolve(image, kernel, padding='reflect')
        c = noisy(c, 'gauss', uni[i])
        c = c.clip(0.01,0.95)
        i +=1

        all_images.append(center_crop_pixel(c,image.shape[0]))
    #plot_images(all_images)
    #plt.show()
    return np.asarray(all_images), np.linspace(min_z_calib, max_z_calib, num_z)


def get_focus_map_from_stack(stack=None, downsample=8, num_iterations=1, gain_sigma=0.0):
    '''
    From a stack of image, get the focus map and return it as a stack
    '''

    focusmap = test_unet(learn, stack, downsample=downsample)
    return focusmap[:,0,:,:]


def get_model():

    return MoffatModel(prefix='peak_') + GaussianModel(prefix='gaussian2_') + LinearModel(prefix='constant_')


def create_calibration_curve_stack(image_stack, z_size_high = 200, std_deviation = None, center_method = 'gaussian'):
    '''
    Creates the calibration curve from the input focus map
    :param focus_map_4d: # z * x * y * c focus map
    :param min_z:
    :param max_z:
    :param z_calibration:
    :param z_size_high:
    :param std_deviation:
    :param center_method:
    :return:
        '''

    focus_map_4d = image_stack.get_focus_map()
    min_z = image_stack.get_min_z()
    max_z = image_stack.get_max_z()
    z_calibration = image_stack.get_z_positions()

    return create_calibration_curve(focus_map_4d, min_z, max_z, z_calibration, z_size_high, std_deviation, center_method)


def create_calibration_curve(focus_map_4d, min_z, max_z, z_calibration, z_size_high = 200, std_deviation = None, center_method = 'gaussian'):
    '''
    Creates the calibration curve from the input focus map
    :param focus_map_4d: # z * x * y * c focus map
    :param min_z:
    :param max_z:
    :param z_calibration:
    :param z_size_high:
    :param std_deviation:
    :param center_method:
    :return:
        '''
    if len(focus_map_4d.shape) == 3:
        focus_map_4d = focus_map_4d[:,:,:,np.newaxis]

    focus_map_4d = focus_map_4d.reshape(focus_map_4d.shape[0], focus_map_4d.shape[1]*focus_map_4d.shape[2], focus_map_4d.shape[3])
    log.info('Focus map 4D shape :{}'.format(focus_map_4d.shape))
    ## REMOVE OUTLIERS
    std = focus_map_4d.std(axis=1)
    mean = focus_map_4d.mean(axis=1)
    focus_map_3d = []
    # We filter pixels to use for the mean.
    for x in range(focus_map_4d.shape[1]):
        if np.all(np.abs(focus_map_4d[:,x,:] - mean) < 3*std):
            focus_map_3d.append(focus_map_4d[:,x,:])
    focus_map_3d = np.asarray(focus_map_3d)
    focus_map_3d = focus_map_3d.swapaxes(0,1)
    # we average all points in the image
    focus_map_2d = np.median(focus_map_3d, axis=(1))
    # We average over the features
    data = focus_map_2d.mean(axis=1)
    #std_deviation = np.std(focus_map_3d, axis=(1)).mean(axis=1)
    log.info('Data shape: {}'.format(data.shape))
    #inter_func = interp1d(z_calibration, data, kind='linear', bounds_error=False, fill_value=(data[0],data[-1]))
    
    calibration = Calibration()
    calibration.z = z_calibration
    calibration.focus_map_1d = data

    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    if center_method == 'gaussian':
        model = get_model()
        start_center = z_calibration[np.argmin(data)]
        weights = np.ones(data.shape[0]) * np.array([gaussian(z_calibration[x], start_center, (max_z-min_z)/10.0) for x in range(data.shape[0])])# * 1.0/(1 + np.abs(np.arange(0, data.shape[0]) - float(np.argmin(data)))**0.5)
        #weights[np.argmin(focus_map_2d[:,0])-3:np.argmin(focus_map_2d[:,0])+4] *= 3
        if std_deviation is not None:
            weights /= np.clip(scale(std_deviation), 0.01, 1.0)
        weights = scale(weights)
        #weights = np.ones(data.shape[0])
        log.info('Center started at {}'.format(start_center))
        params = model.make_params(constant_intercept=data.max(), constant_slope = 0.0, weights = weights,
                                   peak_center=start_center,
                                   peak_sigma=(max_z-min_z)/5.0,
                                   peak_amplitude=(data.min()-data.max()),
                                   gaussian2_sigma=(max_z-min_z),
                                   gaussian2_amplitude = 0.0, gaussian2_center = start_center)

        log.info\
            ('min z : {}, max z = {}'.format(min_z, max_z))
        params['peak_center'].min = min_z
        params['peak_center'].max = max_z
        params['peak_amplitude'].max = 0.0#(data.min()-data.max())
        params['gaussian2_amplitude'].max = 0.0
        #params['peak_center'].vary = False
        params['gaussian2_center'].min = min_z
        params['gaussian2_center'].max = max_z
        params['peak_sigma'].min = 10.0
        params['peak_sigma'].max = max_z-min_z

        #params['gaussian2_sigma'].min = 10.0
        params['gaussian2_sigma'].max = max_z-min_z
        params['constant_slope'].vary = False
        params['constant_intercept'].min = 0
        params['constant_intercept'].max = data.max()
        params['gaussian2_amplitude'].vary = False

        #params['peak_amplitude']
        #mi = lmfit.minimize(model, params, method='Nelder', reduce_fcn='neglogcauchy')

        result = model.fit(data, params, x=z_calibration,  method='nelder')
        log.info(result.fit_report())
        sigma_1 = result.params['peak_amplitude'].value
        sigma_2 = result.params['gaussian2_amplitude'].value
        calibration.params = result.params
        calibration.calibration = result


        if abs(sigma_1) > abs(sigma_2):
            calibration.peak_center = result.params['peak_center'].value
            calibration.peak_beta = result.params['peak_beta'].value
            calibration.gaussian2_center =  result.params['gaussian2_center'].value
            calibration.peak_sigma = result.params['peak_sigma'].value
            calibration.gaussian2_sigma =  result.params['gaussian2_sigma'].value
            calibration.peak_amplitude = result.params['peak_amplitude'].value
            calibration.gaussian2_amplitude =  result.params['gaussian2_amplitude'].value
        else:
            log.info('Gaussian 2 is chosen as the peak !!')
            exit()
            calibration.peak_center = result.params['gaussian2_center'].value
            calibration.gaussian2_center =  result.params['peak_center'].value
            calibration.peak_sigma = result.params['gaussian2_sigma'].value
            calibration.gaussian2_sigma =  result.params['peak_sigma'].value
            calibration.peak_amplitude = result.params['gaussian2_amplitude'].value
            calibration.gaussian2_amplitude =  result.params['peak_amplitude'].value

        calibration.c = result.params['constant_intercept'].value

        log.info('Found mu = {}, sigma = {}, c = {}, width= {}'.format(calibration.peak_center, calibration.peak_sigma, calibration.c, calibration.get_width()))

    elif center_method == 'polynomial':
        yp = np.linspace(min_z, max_z, z_size_high)
        fitted_curve = np.polyfit(z_calibration, focus_map_2d[:, 0], 5)
        p = np.poly1d(fitted_curve)
        calibration.peak_center = yp[np.argmin(p(yp))]
    elif center_method == 'minimum':
        calibration.peak_center = np.min(focus_map_2d[:,0])

    yp = np.linspace(min_z, max_z, z_size_high)
    plt.figure()
    plt.plot(z_calibration, data, '.')
    plt.plot(yp, result.eval(result.params, x=yp))
    plt.plot(z_calibration, weights)
    plt.plot(yp, calibration.eval(yp-calibration.peak_center))
    plt.legend(['original calibration curve', 'gaussian fitted curve', 'weights', 'calibration'])
    log.info('Calibration curve shifted by {}'.format(calibration.peak_center))
    #plt.show()
    return calibration


def detect_4d_acquisition_synthetic(image, min_z, max_z, num_z, noise_sigma, num_iterations, gain_sigma, best_focus=-20):
    '''
    From an original image, get a few "points" and blur them accordingly
    :return:
    '''
    focus_map = []
    rand_z = rand_int(min_z, max_z, num_z)
    for z in rand_z:
        blurred,_ = blur_image_stack(image, 1, min_z_calib=z, max_z_calib=z, z_focus=best_focus)
        focus = get_focus_map_from_stack(blurred, num_iterations=num_iterations, gain_sigma=gain_sigma)[0]
        log.info('Z generated : {} , focus found {} '.format(z, focus))
        focus_map.append(focus)

    focus_map = np.asarray(focus_map)
    noise = np.random.normal(0, noise_sigma, focus_map.size)
    noise = noise.reshape(focus_map.shape)
    focus_map += noise
    return np.asarray(focus_map) + noise, rand_z


def plot_focus_acquisition(calibration_curve, two_z, two_aqu, best_shift, values_shift):
    plt.figure()
    yp = np.linspace(np.min(two_z), np.max(two_z), 1000)
    plt.plot(yp, calibration_curve.eval(yp-best_shift))
    plt.plot(yp, calibration_curve.eval(yp-best_shift, mode='fir'))
    plt.plot(yp, calibration_curve.eval(yp-values_shift, mode='fir'))

    plt.scatter(two_z, two_aqu[:, 0, 0])
    plt.scatter(two_z, two_aqu[:, -1, -1])
    plt.legend(['Calibration curve', 'Calibration acquisition',  'Calibration with found fit', 'Acquisition for pixel (0,0)', 'acquisition for pixel (n,n)'])


    plt.xlabel('Physical distance (im)')
    plt.ylabel('Focus unit')
    plt.title('Acquisition and fit')


def find_nearest(array, value):
    '''
    FIND NEAREST ARRAY INDEX FROM A VALUE
    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


@jit(nopython=True, cache=True)
def correlate_all_points(x_max, y_max, calib, two_acqu_p, two_acqu_p_clipped, shift_idx, correlated):
    '''
    Todo: use signal processing methods (i.e. convolution) to make this correlation happening.
    '''
    for x in range(x_max):
        for y in range(y_max):
            weights = two_acqu_p_clipped[x, y]  # / (1 + two_acqu_p[x, y])
            distance_matrix = (calib - two_acqu_p[x, y])**2
            distance_to_sum = distance_matrix * weights
            distance = np.mean(distance_to_sum)
            correlated[shift_idx, x, y] = distance
    return correlated


def fit_to_calibration_correlation(search_min_x, search_max_x, points_to_correlate_value, points_to_correlate_z, calibration_curve_y, z_size_correlation = None):
    '''
    USING THE CALIBRATION CURVE, SLIDE IT AND MEASURE DISTANCE (CORRELATION)
    :return:
    '''
    if z_size_correlation is None:
        z_size_correlation = z_size_high

    calib_min_x = np.min(calibration_curve_y.z)
    calib_max_x = np.max(np.min(calibration_curve_y.z))
    yp = np.linspace(search_min_x, search_max_x, z_size_correlation)
    yp_with_padding = np.linspace(search_min_x - (calib_max_x - calib_min_x) //2  , search_max_x + (calib_max_x - calib_min_x) //2, z_size_correlation)

    two_acqu_p = np.zeros((points_to_correlate_value.shape[1], points_to_correlate_value.shape[2], yp.shape[0]))
    two_acqu_p_clipped = np.zeros((points_to_correlate_value.shape[1], points_to_correlate_value.shape[2], yp.shape[0]))

    for z_idx in range(points_to_correlate_z.shape[0]):
        value_z, index_z = find_nearest(yp_with_padding, points_to_correlate_z[z_idx])
        for x in range(points_to_correlate_value.shape[1]):
            for y in range(points_to_correlate_value.shape[2]):
                two_acqu_p[x,y,index_z] = points_to_correlate_value[z_idx, x, y]
                two_acqu_p_clipped[x,y,index_z] = 1.0

    ## CORRELATE
    correlated = np.zeros((yp.shape[0], points_to_correlate_value.shape[1], points_to_correlate_value.shape[2]))
    log.info('start correlation')
    calib = []

    for shift_idx, shift in enumerate(yp):
        calib.append(calibration_curve_y.eval(yp_with_padding-shift))
    #calib = np.asarray(calib)
        #print('shift {}, idx  {}'.format(shift, shift_idx))

        #dirac = np.zeros(yp.shape[0])
        #dirac[shift_idx] = 1.0
        #calib = np.asarray([convolve(dirac, augmented_curve, 'same')]).transpose().squeeze()
        #plt.figure()
        #plt.plot(yp_with_padding, calib)
        #plt.show()
        #start = time.time()

        correlated = correlate_all_points(points_to_correlate_value.shape[1], points_to_correlate_value.shape[2],
                                              calib[-1], two_acqu_p,two_acqu_p_clipped, shift_idx, correlated)

        #end = time.time()
        #runtime = end - start
        #print('runtime {}'.format(runtime))

    return correlated, yp_with_padding

def plot_correlation(py, correlated, minimum_arg, minimums):
    plt.figure()
    plt.plot(py, correlated[:,0,0])
    plt.plot(py, correlated[:,-1,-1])
#    plt.scatter(py[minimum_arg[0,0]], minimums[0,0])
#    plt.scatter(py[minimum_arg[-1,-1]], minimums[-1,-1])
    plt.legend(['pixel 0,0', 'pixel n,n', 'minimum 0 0', 'minimum n n'])
    plt.title('cross correlation between calibration curve and pixel values')
    plt.xlabel('Physical distance (im)')
    plt.ylabel('Focus unit')

def plot_final_best_values(final_best_values):
    plt.figure()
    plt.imshow(final_best_values)
    plt.title('best shift values per pixel')
    plt.colorbar()

def get_best_focus_from_image_stack(image_stack, calibration_curve, research_boundaries, z_size_correlation = 5000):
    
    ################################## START OF DETECTION PART ###########################################
    ################ CORRELATE WITH THE CALIBRATION CURVE AND DRAW THE CORRLELATION CURVE ################
    log.info('Correlate...')

    correlation_results, py = fit_to_calibration_correlation(research_boundaries[0], research_boundaries[1], image_stack.get_focus_map(), image_stack.get_z_positions(), calibration_curve, z_size_correlation=z_size_correlation)
    ######################### GET THE MOST CORRELATED POINT AND SET THE SHIFT ############################
    minimum_arg = np.argmin(correlation_results,axis=0)
    #bornes_moyenne = np.asarray(research_boundaries).mean()
    final_best_values = py[minimum_arg]
    log.info('Minimums for px 0,0 {}, px -1,-1 {}'.format(final_best_values[0,0], final_best_values[-1,-1]))
    ##################################### END OF DETECTION PART ##########################################
    return final_best_values, correlation_results, py


def get_gss_points(xL, xR):
    log.info('xL: {}'.format(xL))
    log.info('xR: {}'.format(xR))

    delta = (golden_ratio - 1) * (xR - xL)
    a = xR - delta
    log.info('a: {}'.format(a))
    b = xL + delta
    log.info('b: {}'.format(b))
    return a, b


def golden_ratio_search_step(gss_data_stack):
    # GSS
    if gss_data_stack[-4]:
        d_xL_xR = (gss_data_stack[1] - gss_data_stack[0]) / (2 * golden_ratio - 3) * (golden_ratio - 1)

    if gss_data_stack[3] < gss_data_stack[2]:
        gss_data_stack[2] = gss_data_stack[3]
        gss_data_stack[0], gss_data_stack[1] = get_gss_points(gss_data_stack[0],
                                                              gss_data_stack[0] + d_xL_xR)
        gss_data_stack[4] = 1.
    else:
        xL = gss_data_stack[1]
        gss_data_stack[3] = gss_data_stack[2]
        gss_data_stack[0], gss_data_stack[1] = get_gss_points(gss_data_stack[1] - d_xL_xR,
                                                              gss_data_stack[1])
        gss_data_stack[4] = 0.

    return gss_data_stack


def get_focus_mean(focus_map):
    return focus_map.min()

def synth_data():
    '''
    Generate synthetic data (triangle wave) and correlate
    Unused
    '''
    # Generate triangle curve
    search_min_x = 500
    search_max_x = 1100

    calib_min_x = -200
    calib_max_x = 200
    calib_min = -100
    calib_max = 100
    num_points = 200
    z_size_correlation = 500

    calibration_curve_x = np.linspace(calib_min, calib_max, num_points)
    dy = np.linspace(calib_min_x, calib_max_x, num_points)
    calibration_curve_y_data = np.abs(calibration_curve_x)/50.0 + 0.01
    calibration_curve_y = interp1d(calibration_curve_x, calibration_curve_y_data, kind='linear', bounds_error=False,
                          fill_value=(calibration_curve_y_data[0], calibration_curve_y_data[-1]))
    ypp = np.linspace(-(search_max_x-search_min_x)+calib_min, (search_max_x-search_min_x)+calib_max, z_size_correlation)

    plt.figure()
    plt.plot(dy, calibration_curve_y(dy))

    plt.title("Calibration curve")


    # generate 3 points
    #points_to_correlate_z = np.asarray([600, 1000, 800])
    #points_to_correlate_value = np.asarray([2.0, 2.0, 0])[:, np.newaxis, np.newaxis]

    points_to_correlate_z = np.asarray([850])
    points_to_correlate_value = np.asarray([0])[:, np.newaxis, np.newaxis]

    yp = np.linspace(search_min_x, search_max_x, z_size_correlation)
    yp_with_padding = np.linspace(search_min_x - (calib_max_x - calib_min_x) //2  , search_max_x + (calib_max_x - calib_min_x) //2, z_size_correlation)

    two_acqu_p = np.zeros((points_to_correlate_value.shape[1], points_to_correlate_value.shape[2], yp.shape[0]))
    two_acqu_p_clipped = np.zeros((points_to_correlate_value.shape[1], points_to_correlate_value.shape[2], yp.shape[0]))

    for z_idx in range(points_to_correlate_z.shape[0]):
        value_z, index_z = find_nearest(yp_with_padding, points_to_correlate_z[z_idx])
        for x in range(points_to_correlate_value.shape[1]):
            for y in range(points_to_correlate_value.shape[2]):
                two_acqu_p[x,y,index_z] = points_to_correlate_value[z_idx, x, y]
                two_acqu_p_clipped[x,y,index_z] = 1.0

    # ADD BORDERS TO CURVE
    #augmented_curve = calibration_curve_y(ypp).flatten()
    augmented_curve = calibration_curve_y(ypp).flatten()
    plt.figure()
    plt.plot(ypp, augmented_curve)
    plt.title('Smooth calibration curve centered at zero and with augmented boundaries')
    ## CORRELATE
    correlated = np.zeros((yp.shape[0], points_to_correlate_value.shape[1], points_to_correlate_value.shape[2]))
    for shift_idx, shift in enumerate(yp):
        #print('shift {}, idx  {}'.format(shift, shift_idx))

        #dirac = np.zeros(yp.shape[0])
        #dirac[shift_idx] = 1.0
        #calib = np.asarray([convolve(dirac, augmented_curve, 'same')]).transpose().squeeze()
        calib = calibration_curve_y(yp_with_padding-shift)
        #plt.figure()
        #plt.plot(yp_with_padding, calib)
        #plt.show()
        correlated = correlate_all_points(points_to_correlate_value.shape[1], points_to_correlate_value.shape[2],calib,
                                          two_acqu_p,two_acqu_p_clipped, shift_idx, correlated)

    minimums = np.min(correlated,axis=0)
    minimum_arg = np.argmin(correlated,axis=0)
    plot_correlation(yp, correlated, minimum_arg, minimums)

    plt.show()


def synth_image(method='cnn', min_points_acquisition = 3, max_points_acquisition = 6):
    '''
    Comparison between different scoring function and simulation of autofocus.
    '''

    ################################ BOUNDARIES AND PARAMETERS ###########################################
    num_calibration_acquisitions = 1 # number of times the calibration curve is computer with a bit of random error
    num_iterations = 1 # number of detections with different random light
    bornes_research = 600, 1000 # RESEARCH BOUNDARIES (THE POINTS WILL BE GUESSED THERE
    bornes_calibration = 750, 850 # CALIBRATION BOUNDARIES (THE TRUE FOCUS IS IN THIS RANGE)
    num_z_points_calibration = 111 # NUMBER OF POINTS FOR INITIAL CALIBRATION
    noise_sigma=0.05
    gain_sigma=0.1
    downsample = 40
    z_size_high = 1500

    range_param = 2.0
    criterion = 2.0
    absolute_z_limit_min = bornes_research[0]
    absolute_z_limit_max = bornes_research[1]


    ################################## CREATE A STACK OF BLURRY IMAGES ###################################
    real_focus = rand_int(bornes_research[0]+150, bornes_research[1]-150)
    calibration_focus = rand_int(bornes_calibration[0]+80, bornes_calibration[1]-80)
    log.info('Aquisition focus : {}. Calibration focus : {}'.format(real_focus, calibration_focus))
    log.info('The real best shift point is {}'.format(real_focus-calibration_focus))

    # GET SYNTHETIC BLURRED IMAGES
    log.info('Get image stack...')
    image = get_synthetic_image()
    stack, z_calibration = blur_image_stack(image, num_z_points_calibration, min_z_calib=bornes_calibration[0], max_z_calib = bornes_calibration[1], z_focus=calibration_focus, noise_sigma=0.2, width_coeff=0.9)
    
    ################################# START OF CALIBRATION PART ##########################################
    ##################### DETECT THE FOCUS MAP FOR THE CALIBRATION CURVE #################################
    # GET A FEW FOCUS MAPS WITH DIFFERENT GAINS
    if method is 'cnn':
        log.info('Get focus maps with {} different gains...'.format(num_calibration_acquisitions))
        focus_maps = []
        for i in range(num_calibration_acquisitions):
            log.info('Calibration ...')
            rn = np.random.normal(0,gain_sigma,1)
            focus_map = get_focus_map_from_stack(stack+rn, num_iterations=1, gain_sigma=0, downsample=downsample)
            focus_maps.append(focus_map)

        # AVERAGE ALL THE FOCUS MAPS
        focus_map_mean = np.asarray(focus_maps).mean(axis=0)
        #focus_map_mean = focus_map_mean[:,np.newaxis]
        focus_map_std = np.asarray(focus_maps).std(axis=0)

        # PLOT
        #plt.figure()
        #py = np.linspace(bornes_research[0], bornes_research[1],z_size_high)
        #plt.plot(z_calibration, focus_map_mean[:, 0, 0],'-o')
        #plt.plot(z_calibration, focus_map_mean[:, -1, -1], '-o')
        #plt.fill_between(z_calibration, focus_map_mean[:, 0, 0] - focus_map_std[:,0,0], focus_map_mean[:, 0, 0] + focus_map_std[:,0,0], alpha=0.5)
        #plt.fill_between(z_calibration, focus_map_mean[:, -1, -1] - focus_map_std[:,-1, -1], focus_map_mean[:, -1, -1] + focus_map_std[:,-1, -1], alpha=0.5)
        #plt.legend(['focus map X+Y 0 0', 'focus map X+Y 32 32'])
        #plt.xlabel('Physical distance (im)')
        #plt.ylabel('Focus unit')
        #plt.title('Calibration curve for 2 pixels')
        ##################### SHIFT AND CREATE INTERPOLATED CALIBRATION CURVE FUNCTION #########################
        log.info('Create calibration curve...')
        focus_map_mean = focus_map_mean[:,:,:,np.newaxis]
        std = focus_map_std.mean(axis=1).mean(axis=1)
        calibration_curve_real = create_calibration_curve(focus_map_mean, bornes_research[0], bornes_research[1], z_calibration, z_size_high, std_deviation = std)

        ################################# END OF CALIBRATION PART ##############################################
        ################################## START OF DETECTION PART #############################################

         #plt.show()

        ################### GET A FEW POINTS FOR ACQUISITION WITHIN THE RESEARCH BOUNDARIES ####################
        log.info('Get {} acquisitions points...'.format(min_points_acquisition))
        image = get_synthetic_image(flip=True)
        image_stack = None
        current_z = random.choice([real_focus-20, real_focus+20])[0]

        while i < min_points_acquisition:
            i+=1
            #two_aqu_real,two_z_real = detect_4d_acquisition_synthetic(image, bornes_research[0], bornes_research[1],num_points_acquisition, noise_sigma, num_iterations, gain_sigma, real_focus)
            blurred, _ = blur_image_stack(image, 1, min_z_calib=current_z, max_z_calib=current_z, z_focus=real_focus, width_coeff=0.9, noise_sigma=noise_sigma)
            if image_stack is None:
                image_stack = ImageStack()
                image_stack.set_image_stack(blurred, blurred.shape[0], blurred.shape[1], downsample=downsample, z_positions = current_z)
            else:
                image_stack.add_image_to_stack(blurred, current_z)

            #plot_focus_acquisition(bornes_calibration[0], bornes_calibration[1], focus_map_mean, two_z_real,two_aqu_real, real_focus)
            #plt.show()

            ################ CORRELATE WITH THE CALIBRATION CURVE AND DRAW THE CORRLELATION CURVE ##################
            log.info('Correlate...')
            correlated, ypp = fit_to_calibration_correlation(bornes_research[0], bornes_research[1], image_stack.get_focus_map(), image_stack.get_z_positions(),
                                            calibration_curve_real, z_size_correlation=z_size_high)

            ######################### GET THE MOST CORRELATED POINT AND SET THE SHIFT ##############################
            minimums = np.min(correlated,axis=0)
            minimum_arg = np.argmin(correlated,axis=0)
            final_best_values = ypp[minimum_arg]

            log.info('Minimums for px 0,0 {}, px -1,-1 {}'.format(final_best_values[0,0], final_best_values[-1,-1]))

            #plot_correlation(ypp, correlated, minimum_arg, minimums)
            #plot_focus_acquisition(calibration_curve_real, image_stack.get_z_positions(), image_stack.get_focus_map(), real_focus, final_best_values.mean())
            #plt.show()
            message = 1

            # For the first image
            if image_stack.get_num_z() == 1:
                init_half_range = range_param * calibration_curve_real.get_width()
                xL, xR = image_stack.get_min_z() - init_half_range, image_stack.get_min_z() + init_half_range
                optimizer_data = [0, 0, 0, 0, 0, False, True, 0, 0, 0]
                optimizer_data[0], optimizer_data[1] = get_gss_points(xL=xL, xR=xR)
                new_point = optimizer_data[0]
            elif image_stack.get_num_z() == 2:
                optimizer_data[2] = get_focus_mean(image_stack.get_focus_map()[-1])
                new_point = optimizer_data[1]

            elif not optimizer_data[5]:
                if image_stack.get_num_z() == 3:
                    optimizer_data[3] = get_focus_mean(image_stack.get_focus_map()[-1])
                else:
                    focus_mean = get_focus_mean(image_stack.get_focus_map()[-1])

                    if optimizer_data[4] == 1:
                        optimizer_data[3] = focus_mean
                    else:
                        optimizer_data[2] = focus_mean

                if not (optimizer_data[-2] == optimizer_data[-1] - 1):
                    optimizer_data = golden_ratio_search_step(optimizer_data)

                if optimizer_data[4] == 1:
                    new_point = optimizer_data[1]
                else:
                    new_point = optimizer_data[0]

                if ((optimizer_data[1] - optimizer_data[0]) / (2 * golden_ratio - 3) < criterion):
                    log.info('Criterion Satisfied => Best Focus Point Found')
                    new_point = (optimizer_data[0] + optimizer_data[1]) / 2
                    optimizer_data[5] = True

                elif image_stack.get_num_z() >= min_points_acquisition and not optimizer_data[5]:
                    log.info('Criterion not Satisfied but too many images yet. Convexity tests...')
                    score = 0.0
                    i = 0
                    for _x in range(image_stack.get_focus_map().shape[1]):
                        for _y in range(image_stack.get_focus_map().shape[2]):
                            x = image_stack.get_focus_map()[:, _x, _y]
                            y = image_stack.get_z_positions().reshape((-1, 1))
                            x_lin = PolynomialFeatures(degree=1, include_bias=True).fit_transform(y)
                            x_poly = PolynomialFeatures(degree=2, include_bias=True).fit_transform(y)
                            model_lin = LinearRegression().fit(x_lin, x)
                            model_poly = LinearRegression().fit(x_poly, x)
                            #plt.plot(y[:, 0], model_poly.predict(PolynomialFeatures(degree=2, include_bias=True).fit_transform(y)))
                            #plt.plot(y[:, 0], model_lin.predict(PolynomialFeatures(degree=1, include_bias=True).fit_transform(y)))
                            score += model_poly.score(x_poly,y) - model_lin.score(x_lin, y)
                            i += 1
                    plt.plot(ypp, model_poly.predict(PolynomialFeatures(degree=2, include_bias=True).fit_transform(ypp.reshape((-1, 1)))))
                    plt.plot(ypp, model_lin.predict(PolynomialFeatures(degree=1, include_bias=True).fit_transform(ypp.reshape((-1, 1)))))
                    score /= i
                    log.info('Final score = {}'.format(score))
                    if score > 0:
                        log.info('convex function found')
                        optimizer_data[5] = True
                        message = 2
                    elif image_stack.get_num_z() <= max_points_acquisition:
                        log.info('not convex function found, add one point')
                        min_points_acquisition += 1
                    else:
                        log.info('not convex function found, but too many images')
                        optimizer_data[5] = True
                        message = 2

                #fb = final_best_values[
                #    (final_best_values > bornes_research[0]) & (final_best_values < bornes_research[1])]
                #clustering = MeanShift(bandwidth=calibration_curve_real.get_width())
                #clustering.fit(fb.reshape(-1, 1))
                #print('new point center available : {}'.format(clustering.cluster_centers_))
                #new_point = clustering.cluster_centers_[np.argmax(np.bincount(clustering.labels_))]

                new_point = np.clip(new_point, absolute_z_limit_min, absolute_z_limit_max)
            else:
                log.info("Comparing focus values")
                best_focus = get_focus_mean(image_stack.get_focus_map()[0])
                best_focus_idx = 0
                log.info("Index: ", best_focus_idx)
                log.info("Focus: {}".format(best_focus))
                for i, focus_map in enumerate(image_stack.get_focus_map()[1::]):
                    temp = get_focus_mean(focus_map)
                    log.info("Index: ", i + 1)
                    log.info("Focus: {}".format(temp))
                    if temp < best_focus:
                        best_focus = temp
                        best_focus_idx = i + 1
                        log.info("Current Best")
                        log.info("Index: ", best_focus_idx)
                        log.info("Focus: {}".format(best_focus))

                #new_point = image_stack.get_z_positions()[best_focus_idx]
                message = 2
            current_z = new_point#np.random.normal(image_stack.get_z_positions()[np.argmin(image_stack.get_focus_map().mean(axis=(1,2,3)))],calibration_curve_real.peak_sigma / 5.0)
            current_z = float(np.round(current_z*4.0))/4.0

            log.info('Current points {}'.format(image_stack.get_z_positions()))
    else:
        # brent method for other
        ################### GET A FEW POINTS FOR ACQUISITION WITHIN THE RESEARCH BOUNDARIES ####################
        log.info('Get {} acquisitions points...'.format(min_points_acquisition))
        image = get_synthetic_image(flip=True)

        image_stack = None

        def f(current_z):
            nonlocal image_stack
            current_z = float(np.round(current_z*4.0))/4.0
            blurred, _ = blur_image_stack(image, 1, min_z_calib=current_z, max_z_calib=current_z, z_focus=real_focus,
                                          width_coeff=0.9, noise_sigma=noise_sigma)
            if image_stack is None:
                image_stack = ImageStack()
                image_stack.set_image_stack(blurred, blurred.shape[0], blurred.shape[1], 128, current_z)
            else:
                image_stack.add_image_to_stack(blurred, current_z, update_focus_map=False)
            score = 1/get_hpf_image(image_stack.image_stack[-1],size=128, method = method).mean()
            log.info('score for z={}  is {}'.format(current_z, score))
            return score

        result = optimize.minimize_scalar(f, method='bounded', bounds = bornes_research, options= {'maxiter':max_points_acquisition})
        final_best_values = result.x
        log.info('Found : {}'.format(result.x))

    log.info('Current points {}'.format(image_stack.get_z_positions()))
    #plot_final_best_values(final_best_values)
    #plt.figure()
    #plt.imshow(image)
    #plt.title('Original image')
    #image_stack.print_data()
    #image_stack.print_focus_map()
    if not np.isscalar(final_best_values):
        final_best_values = final_best_values[(final_best_values > bornes_research[0]) & (final_best_values < bornes_research[1])]
        log.info('Best values {}'.format(final_best_values))
        clustering = MeanShift(bandwidth=calibration_curve_real.get_width())
        clustering.fit(final_best_values.reshape(-1,1))
        log.info('Centers available : {}'.format(clustering.cluster_centers_))
        center = clustering.cluster_centers_[np.argmax(np.bincount(clustering.labels_))]
        #stds = []
        #for i in range(len(clustering.cluster_centers_)):
        #    point = final_best_values[clustering.labels_ == i]
        #    stds.append(point.std())

        ## au lieu de choisir le cluster le plus gros il faudrait peut être regarder le cluster le plus compact = avec la moins de std
    else:
        center = final_best_values

    log.info('Found value = {}, Real value = {}'.format(center, real_focus))
    error = (np.abs(real_focus-center)).mean()
    log.info('Error : {}'.format(error))
    ##################################### END OF DETECTION PART #############################################
    #plt.show(block=True)
    return error


if __name__ == '__main__':
    methods = ['cnn', 'hpf', 'tenengrad1']
    number_of_images = 100
    range_of_acquisitions = np.arange(3, 8, 1).tolist()

    for method in methods:
        all_errors = []
        for idx, max_acqu in enumerate(range_of_acquisitions):
            errors = []
            for i in range(number_of_images):
                errors.append((synth_image(method=method, max_points_acquisition=max_acqu, min_points_acquisition=max_acqu)))
            all_errors.append(errors)

        for idx, max_acqu in enumerate(range_of_acquisitions):
            log.info('For {} acquisitions, the error is {} +- {}'.format(max_acqu, np.mean(all_errors[idx]), np.std(all_errors[idx])))
        pickle_save('errors_simulations_{}.pkl'.format(method), all_errors)
