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

import logging
import numpy as np
import grpc
import deepfocus_pb2
from scipy.ndimage import gaussian_filter
import deepfocus_pb2_grpc
from deepfocus_pb2 import BestFocusResponse
from concurrent import futures
import time
import pickle
from skimage.measure import label
from skimage.measure import regionprops
from skimage.transform import resize
import math
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
from toolbox import scale3d, multipage

logging.basicConfig(
    format="%(asctime)s [SERVER] %(message)s".format(00),
    handlers=[
        logging.FileHandler("output_log_{}.log".format(00)),
        logging.StreamHandler()
    ])

log = logging.getLogger('')
log.setLevel(logging.INFO)

from unet_detector import *
import calibration_fit as calib_fit


def get_focus_mean(focus_map, width, height, roi, mode = 'min'):
    focus_map_resized = resize(focus_map, (width, height), order=0)
    focus_map_roi = focus_map_resized[roi[0]:roi[2], roi[1]:roi[3]]
    if mode is 'min':
        return focus_map_roi.min()
    elif mode is 'mean':
        return focus_map_roi.mean()
    elif mode is 'median':
        return np.median(focus_map_roi)
    else:
        log.error('get_focus_mean - mode does not exist')
        return 0


class DeepFocusServicer(deepfocus_pb2_grpc.DeepFocusServicer):

    def __init__(self):
        self.calibration_curve = None
        self.image_stack = None
        self.optimizer_name = 'Correlation'
        self.optimizer_data = None  # [a,b,focus(a),focus(b),b<a, last_found,GSS,interval_k,fib_k,fib_n] (list)
        self.gss = {}
        self.i = 0

    def LoadCalibration(self, request, context):
        self.calibration_curve = calib_fit.Calibration()
        self.calibration_curve.load(pickle.load(open(request.calibration_curve_pickle_path, 'rb')))
        log.info("Calibration successfully loaded!")
        return deepfocus_pb2.CurveResponse(pickle_found=True)

    def StartAutofocus(self, request, context):
        log.info('StartAutofocus command received.')
        ### ReponseType => 0-> no calibration file, 1-> need one more image, 2-> autofocus okay

        ### PARAMS ###
        z_size = 3000
        downsample = 16
        range_param = 1
        criterion = 4 * 10 * request.threshold # UNUSED
        correlation_threshold = request.threshold / 50.0 # UNUSED
        std_threshold = request.threshold * 50.0 # UNUSED
        minimum_images = request.min_iter # UNUSED
        max_images = request.max_iter
        self.optimizer_name = request.optimizer
        roi_3d = False

        if self.calibration_curve is None:
            log.error('The calibration curve is not loaded!')
            return deepfocus_pb2.BestFocusResponse(message=0)

        ## INPUT CONVERSION
        if request.bytes_per_pixel == 2:
            dt = np.dtype(np.int16)
            dt = dt.newbyteorder('<')
            input_stack_array = np.frombuffer(request.image_list, dtype=dt).astype(np.float64) / 32768.0
        elif request.bytes_per_pixel == 1:
            dt = np.dtype(np.int8)
            dt = dt.newbyteorder('<')
            input_stack_array = np.frombuffer(request.image_list, dtype=dt).astype(np.float64) / 128.0

        input_z_positions = np.asarray(request.z_position)
        input_image_stack = np.reshape(np.asarray(input_stack_array), (1, request.width, request.height))
        input_image_stack = np.asarray([gaussian_filter(input_image_stack[i], 1) for i in range(input_image_stack.shape[0])])
        log.info('Received image size : {}'.format(input_image_stack.shape))
        # imsave2('autof_{}.png'.format(self.i), input_image_stack[0])
        self.i += 1
        # Adds the image to image_stack. The image_stack calls the CNN to get the focus map and save it.
        if self.image_stack is None:
            input_image_stack_class = calib_fit.ImageStack()
            input_image_stack_class.downsample = downsample
            input_image_stack_class.set_image_stack(image_stack=input_image_stack, width=request.width,
                                                    height=request.height, z_positions=input_z_positions, downsample=downsample)

            self.image_stack = input_image_stack_class
            log.info('I create the stack.')
            log.info('Optimization: {}'.format(self.optimizer_name))
        else:
            for i in range(input_image_stack.shape[0]):
                self.image_stack.add_image_to_stack(input_image_stack, input_z_positions)

        absolute_z_limit_min = np.max([0, self.image_stack.z_positions[0] + request.min_limit])
        absolute_z_limit_max = self.image_stack.z_positions[0] + request.max_limit

        roi = np.asarray(request.roi_coords)  # 4 dimensional array. p0(x,y) p1(x,y)
        log.info('ROI found : {}'.format(roi))

        if roi[0] == -1 or roi[0] == -2:
            log.info('ROI -> 2D automatic ROI')
            if roi[0] == -2:
                roi_3d = True
            focus_map_resized = [resize(self.image_stack.focus_map[i, :, :, 0], (request.width, request.height),
                                        order=1) for i in range(self.image_stack.focus_map.shape[0])]
            threshold_roi = np.median(focus_map_resized)
            roi_thresholded = np.asarray(
                [(focus_map_resized[i] < threshold_roi) for i in range(self.image_stack.focus_map.shape[0])])
            roi_labeled = np.asarray([label(roi_thresholded[i]) for i in range(roi_thresholded.shape[0])])
            best_region = None
            for i in range(roi_labeled.shape[0]):
                for region_index, region in enumerate(regionprops(roi_labeled[i])):
                    if best_region is None or best_region.area < region.area:
                        best_region = region
            # bbox describes: min_row, min_col, max_row, max_col
            a, b, c, d = region.bbox
            roi[0] = a
            roi[1] = b
            roi[2] = c
            roi[3] = d
            log.info('Found new 2D ROI: {}'.format(region.bbox))
        elif roi[0] == roi[1] == roi[2] == roi[3]:
            log.info('No ROI')
            roi[0] = 0
            roi[1] = 0
            roi[2] = request.width
            roi[3] = request.height
        elif roi[0] == -3:  # only 3d roi
            roi_3d = True



        ######################### GET THE MOST CORRELATED POINT AND SET THE SHIFT ##############################

        message = 1

        invphi = (math.sqrt(5) - 1) / 2  # 1 / phi
        invphi2 = (3 - math.sqrt(5)) / 2  # 1 / phi^2
        #if self.image_stack.get_num_z() > max_images:
        #    self.optimizer_data[5] = True

        # For the first image
        if self.image_stack.get_num_z() == 1:
            init_half_range = range_param * self.calibration_curve.get_width()

            self.gss['a'] = self.image_stack.get_min_z() - init_half_range
            self.gss['b'] = self.image_stack.get_min_z() + init_half_range
            self.gss['h'] = (self.gss['b'] - self.gss['a'])
            self.gss['c'] = self.gss['a'] + self.gss['h'] * invphi2
            self.gss['d'] = self.gss['a'] + self.gss['h'] * invphi
            new_point = self.gss['c']
            self.gss['point'] = ''

        elif self.image_stack.get_num_z() == 2:
            self.gss['fc'] = get_focus_mean(self.image_stack.get_focus_map()[-1], request.width, request.height, roi)
            new_point = self.gss['d']
        else:
            if self.image_stack.get_num_z() == 3:
                self.gss['fd'] = get_focus_mean(self.image_stack.get_focus_map()[-1], request.width, request.height, roi)

            if self.gss['point'] is 'fc':
                self.gss['fc'] = get_focus_mean(self.image_stack.get_focus_map()[-1], request.width, request.height, roi)
            elif self.gss['point'] is 'fd':
                self.gss['fd'] = get_focus_mean(self.image_stack.get_focus_map()[-1], request.width, request.height, roi)

            print(self.gss)

            if self.gss['fc'] < self.gss['fd']:
                self.gss['b'] = self.gss['d']
                self.gss['d'] = self.gss['c']
                self.gss['fd'] = self.gss['fc']
                self.gss['h'] = invphi * self.gss['h']
                self.gss['c'] = self.gss['a'] + invphi2 * self.gss['h']
                new_point = self.gss['c']
                self.gss['point'] = 'fc'
            else:
                self.gss['a'] = self.gss['c']
                self.gss['c'] = self.gss['d']
                self.gss['fc'] = self.gss['fd']
                self.gss['h'] = invphi * self.gss['h']
                self.gss['d'] = self.gss['a'] + invphi * self.gss['h']
                new_point = self.gss['d']
                self.gss['point'] = 'fd'

            if self.image_stack.get_num_z() == max_images:
                log.info('Correlate...')
                research_boundaries = [absolute_z_limit_min,
                                       absolute_z_limit_max]  # RESEARCH BOUNDARIES (THE POINTS WILL BE GUESSED THERE
                correlated, ypp = calib_fit.fit_to_calibration_correlation(research_boundaries[0],
                                                                           research_boundaries[1],
                                                                           self.image_stack.get_focus_map(),
                                                                           self.image_stack.get_z_positions(),
                                                                           self.calibration_curve,
                                                                           z_size_correlation=z_size)
                minimums = np.min(correlated, axis=0)
                minimum_arg = np.argmin(correlated, axis=0)
                final_best_values = ypp[minimum_arg]
                log.info("Final best values = {}".format(final_best_values))

                ## crop the results to the region of interest (a rectangle)
                minimums = resize(minimums, (request.width, request.height), order=0)
                minimum_arg = resize(minimum_arg, (request.width, request.height), order=0)

                final_best_values = resize(final_best_values, (request.width, request.height),
                                           order=0)
                minimums = minimums[roi[0]:roi[2], roi[1]:roi[3]].flatten()
                minimum_arg = minimum_arg[roi[0]:roi[2], roi[1]:roi[3]].flatten()
                final_best_values_with_roi = final_best_values[roi[0]:roi[2], roi[1]:roi[3]].flatten()
                final_best_values = np.unique(final_best_values_with_roi)

                if not np.isscalar(final_best_values):
                    log.info('Multiples values to chose from: filtering and clustering...')
                    final_best_values = final_best_values[
                        (final_best_values > absolute_z_limit_min) & (final_best_values < absolute_z_limit_max)]
                    log.info('Best values after filtering {}'.format(final_best_values))
                    clustering = MeanShift(bandwidth=self.calibration_curve.get_width())
                    log.info("Doing meanshift algorithm...")
                    clustering.fit(final_best_values.reshape(-1, 1))
                    log.info('Centers available : {}'.format(clustering.cluster_centers_))
                    new_point = clustering.cluster_centers_[np.argmax(np.bincount(clustering.labels_))]
                else:
                    log.info('Only one value to chose from...')
                    new_point = final_best_values


                #new_point = final_best_values_with_roi[np.argmin(minimums, axis=0)]
                new_point = float(new_point)
                message = 2

            if self.image_stack.get_num_z() > max_images:
                log.info("Comparing focus values")
                best_focus = get_focus_mean(self.image_stack.get_focus_map()[-1], request.width, request.height, roi)
                best_focus_idx = 0
                log.info("Index: {}".format(best_focus_idx))
                log.info("Focus for current image: {}".format(best_focus))
                for i, focus_map in enumerate(self.image_stack.get_focus_map()[1::]):
                    temp = get_focus_mean(focus_map, request.width, request.height, roi)
                    log.info("Index: {}".format(i + 1))
                    log.info("Focus: {}".format(temp))
                    if temp < best_focus:
                        best_focus = temp
                        best_focus_idx = i + 1
                        log.info("Current Best")
                        log.info("Index: {}".format(best_focus_idx))
                        log.info("Focus: {}".format(best_focus))

                new_point = self.image_stack.get_z_positions()[best_focus_idx]
                message = 3

        log.info("New point: {}, message {}".format(new_point, message))


        if message == 1:
            log.info("I go to {} to get a new image".format(new_point))
            return deepfocus_pb2.BestFocusResponse(message=message, z_shift=new_point)
        elif message == 2:
            log.info("I go to {} to get a new image".format(new_point))
            return deepfocus_pb2.BestFocusResponse(message=message, z_shift=new_point)
        elif message == 3:
            log.info("I have enough points. Number of images = {}, maximum allowed images = {}".format(self.image_stack.get_num_z(), max_images))
            # calib_fit.plot_correlation(ypp, correlated, np.argmin(correlated, axis=0), minimums)
            # calib_fit.plot_final_best_values(final_best_values_with_roi)
            # calib_fit.plot_focus_acquisition(self.calibration_curve, self.image_stack.get_z_positions(),
            #                        self.image_stack.get_focus_map(), final_best_values_with_roi.mean(), final_best_values_with_roi.min())
            # fig = plt.figure()
            # a = plt.imshow(final_best_values)
            # fig.colorbar(a)
            # multipage('output.pdf')
            #
            # focus_values_min = []
            # focus_values_mean = []
            # focus_values_median = []
            #
            # for i in range(self.image_stack.get_num_z()):
            #     focus_values_min.append(
            #         get_focus_mean(self.image_stack.get_focus_map()[i], request.width, request.height, roi, mode='min'))
            #     focus_values_mean.append(
            #         get_focus_mean(self.image_stack.get_focus_map()[i], request.width, request.height, roi, mode='mean'))
            #     focus_values_median.append(
            #         get_focus_mean(self.image_stack.get_focus_map()[i], request.width, request.height, roi, mode='median'))
            # plt.figure()
            # plt.plot(np.asarray(self.image_stack.get_z_positions()), np.asarray(focus_values_mean), '.')
            # plt.plot(np.asarray(self.image_stack.get_z_positions()), np.asarray(focus_values_min), '.')
            # plt.plot(np.asarray(self.image_stack.get_z_positions()), np.asarray(focus_values_median), '.')
            #
            # yp = np.linspace(self.image_stack.get_min_z(), self.image_stack.get_max_z(), 1000)
            # plt.plot(yp, self.calibration_curve.eval(yp - new_point))
            # plt.title('{} Optimization Result'.format(self.optimizer_name))
            # plt.xlabel('z positions')
            # plt.ylabel('focus values')
            # plt.legend(['focus values (mean)', 'focus values (minimum)', 'focus values (median)','best calibration curve shift'])
            # plt.savefig('{}_optimization_result.png'.format(self.optimizer_name))

            # We delete the image stack and send the best focus value to Java
            del self.image_stack
            del self.optimizer_data
            self.image_stack = None
            self.optimizer_data = None
            self.gss = {}
            if roi_3d:
                min_z = new_point - self.calibration_curve.get_width() * request.roi_3d_num_sigma
                max_z = new_point + self.calibration_curve.get_width() * request.roi_3d_num_sigma
                log.info("ROI 3D... stack in the neighborhood of the best focus {} -> {} to {}".format(new_point, min_z,
                                                                                                    max_z))
                return deepfocus_pb2.BestFocusResponse(message=message, z_shift=new_point, roi_min_z=min_z, roi_max_z=max_z)

            log.info("Best focus found : {}".format(new_point))
            return deepfocus_pb2.BestFocusResponse(message=message, z_shift=new_point)
        else:
            log.error('Error, message unknown')
            exit()

    def CreateCalibrationCurve(self, request, context):
        log.info("Receiving incoming message. The number of byte per pixel is = {}".format(request.bytes_per_pixel))
        ## INPUT CONVERSION
        if request.bytes_per_pixel == 2:
            dt = np.dtype(np.int16)
            dt = dt.newbyteorder('<')
            input_stack_array = np.frombuffer(request.image_list, dtype=dt).astype(np.float64) / 32768.0
        elif request.bytes_per_pixel == 1:
            dt = np.dtype(np.int8)
            dt = dt.newbyteorder('<')
            input_stack_array = np.frombuffer(request.image_list, dtype=dt).astype(np.float64) / 128.0

        downsample = 16

        z_positions = np.asarray(request.z_positions)
        image_stack = np.reshape(input_stack_array, (z_positions.shape[0], request.width, request.height))
        image_stack = image_stack[:min(request.width, request.height), :min(request.width, request.height)]
        image_stack = np.asarray([gaussian_filter(image_stack[i], 1) for i in range(image_stack.shape[0])])
        image_stack = scale3d(image_stack) * 0.8
        image_stack_class = calib_fit.ImageStack()

        image_stack_class.set_image_stack(image_stack=image_stack, width=request.width, height=request.height,
                                          z_positions=z_positions, downsample=downsample)

        ## CALIBRATION
        self.calibration_curve = calib_fit.create_calibration_curve_stack(image_stack_class)

        ## SAVE TO FILE
        calibration_curve_path = "{}/calibration_curve.pickle".format(request.calib_curve_pathway)
        log.info('Saving the curve to {} ...'.format(calibration_curve_path))

        with open(calibration_curve_path, 'wb') as file:
            pickle.dump(self.calibration_curve.save(), file)

        ## PLOT
        yp = np.linspace(image_stack_class.get_min_z(), image_stack_class.get_max_z(), 1000)
        stack, ypp, focus_map = image_stack_class.get_image_stack()

        plt.figure()
        plt.plot(ypp, self.calibration_curve.focus_map_1d, '.')
        plt.plot(yp, self.calibration_curve.eval(yp-self.calibration_curve.peak_center))
        plt.ylabel('Focus')
        plt.xlabel('Z position')
        plt.title('Calibration curve')
        plt.legend(['Acquired points', 'Gaussian fit to calibration curve'])

        calibration_curve_image_name = "calib_curve.png"
        plt.savefig("{}/{}".format(request.calib_curve_pathway, calibration_curve_image_name))
        calibCurveImageFilePathName = "{}/{}".format(request.calib_curve_pathway, calibration_curve_image_name)
        multipage('calibration.pdf')
        return deepfocus_pb2.CalibrationCurve(calib_curve_image_file_path=calibCurveImageFilePathName,
                                              gaussian2_center=self.calibration_curve.gaussian2_center,
                                              peak_center=self.calibration_curve.peak_center,
                                              gaussian2_sigma=self.calibration_curve.gaussian2_sigma,
                                              peak_sigma=self.calibration_curve.peak_sigma,
                                              constant_c=self.calibration_curve.c)


def serve():
    gigabyte = 1024 ** 3
    executor = futures.ThreadPoolExecutor()
    server = grpc.server(executor, options=[
        ('grpc.max_send_message_length', gigabyte),
        ('grpc.max_receive_message_length', gigabyte)
    ])
    deepfocus_pb2_grpc.add_DeepFocusServicer_to_server(DeepFocusServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    log.info("Server started.")
    try:
        while True:
            time.sleep(60 * 60 * 24)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
