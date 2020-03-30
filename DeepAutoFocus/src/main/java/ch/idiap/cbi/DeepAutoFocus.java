/**
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
*/


package ch.idiap.cbi;

import com.google.common.eventbus.Subscribe;
import com.google.protobuf.ByteString;

import ij.IJ;
import ij.ImagePlus;
import ij.gui.ImageWindow;
import ij.process.ByteProcessor;
import ij.process.ImageProcessor;
import ij.process.ShortProcessor;
import io.grpc.StatusRuntimeException;
import ij.WindowManager;
import ij.process.ColorProcessor;
import ij.process.FloatProcessor;

import java.awt.Color;
import java.awt.Rectangle;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Date;
import java.util.Properties;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.JFileChooser;
import javax.swing.filechooser.FileSystemView;

import mmcorej.CMMCore;
import mmcorej.StrVector;
import mmcorej.TaggedImage;
import org.apache.commons.lang3.ArrayUtils;
import org.json.JSONException;
import org.json.JSONObject;

import org.micromanager.AutofocusPlugin;
import org.micromanager.Studio;
import org.micromanager.acquisition.SequenceSettings;
import org.micromanager.data.internal.DefaultImage;
import org.micromanager.events.AutofocusPluginShouldInitializeEvent;
import org.micromanager.internal.utils.AutofocusBase;
import org.micromanager.internal.utils.MDUtils;
import org.micromanager.internal.utils.MMException;
import org.micromanager.internal.utils.PropertyItem;
import org.micromanager.internal.utils.ReportingUtils;
import static org.micromanager.internal.utils.imageanalysis.ImageUtils.convertRGB32BytesToInt;

import org.scijava.plugin.Plugin;
import org.scijava.plugin.SciJavaPlugin;

@Plugin(type = AutofocusPlugin.class)
public class DeepAutoFocus extends AutofocusBase implements AutofocusPlugin, SciJavaPlugin {

    private static final String KEY_MIN_LIMIT = "Minimum limit in Z";
    private static final String KEY_MAX_LIMIT = "Maximum limit in Z";
    private static final String KEY_THRESHOLD = "Threshold";
    private static final String KEY_MAX_ITER = "Maximum images at every step";
    private static final String KEY_START_ITER = "Minimum number of images";
    private static final String KEY_AUTO_ROI = "Auto ROI";
    private static final String[] AUTO_ROI_VALUES = {"Manual", "2D ROI", "2D + 3D ROI", "3D ROI"};
    private static final String KEY_OPTIMIZER = "Optimizer";
    private static final String[] OPTIMIZER_VALUES = {"Correlation","CorrelationFIR", "GSS", "Fibonacci"};
    private static final String KEY_SIGMA = "3D ROI width coefficient";
    private static final String KEY_CHANNEL = "Channel";
    private static final String KEY_STEPS = "Precision (px)";
    private static final String KEY_MAX_IMAGE_WIDTH = "Max image width";
    private static final String[] STEPS_VALUES = {"32", "64", "128"};

    private static final String NOCHANNEL = "";
    private static final String AF_DEVICE_NAME = "DeepAutoF";

    private Studio _studio;
    private CMMCore _core;

    public double MIN_LIMIT = -300.0;
    public double MAX_LIMIT = 300.0;
    public int MAX_ITER = 15;
    public int START_ITER = 3;
    public double THRESHOLD = 0.1;
    public double SIGMA = 2;
    public int STEPS = 64;
    public int MAX_IMAGE_WIDTH = 1024;
    public String CHANNEL = "";
    public String AUTO_ROI = "Manual";
    public String OPTIMIZER = "Correlation";

    private String _channelGroup;
    private double _currentPosition;
    private double _nextPosition;
    private double _startPosition;
    private long _t0;
    private long _tPrev;
    private long _tcur;
    private int _byteDepth = 0;
    private double _factor = 1.0;

    Properties _properties = new Properties();

    private DeepAutoFocusClient _grpcclient;
    private boolean _isCalibCurveLoaded;
    private ImageProcessor _imageProcessor;
    private List<Double> _zPositions = new ArrayList<Double>();
    private BestFocusResponse _bestFocusResponse;

    public DeepAutoFocus() {
        super.createProperty(KEY_MIN_LIMIT, Double.toString(MIN_LIMIT));
        super.createProperty(KEY_MAX_LIMIT, Double.toString(MAX_LIMIT));
        super.createProperty(KEY_MAX_ITER, Integer.toString(MAX_ITER));
        super.createProperty(KEY_START_ITER, Integer.toString(START_ITER));
        super.createProperty(KEY_THRESHOLD, Double.toString(THRESHOLD));
        super.createProperty(KEY_SIGMA, Double.toString(SIGMA));
        super.createProperty(KEY_CHANNEL, CHANNEL);
        super.createProperty(KEY_AUTO_ROI, AUTO_ROI, AUTO_ROI_VALUES);
        super.createProperty(KEY_OPTIMIZER, OPTIMIZER, OPTIMIZER_VALUES);
        super.createProperty(KEY_STEPS, Double.toString(STEPS), STEPS_VALUES);
        super.createProperty(CHANNEL);
        super.createProperty(KEY_MAX_IMAGE_WIDTH, Integer.toString(MAX_IMAGE_WIDTH));
        openConfig();
    }

    @Subscribe
    public void onInitialize(AutofocusPluginShouldInitializeEvent event) {
        loadSettings();
    }

    @Override
    public void applySettings() {
        try {
            
            MIN_LIMIT = Double.parseDouble(getPropertyValue(KEY_MIN_LIMIT));
            MAX_LIMIT = Double.parseDouble(getPropertyValue(KEY_MAX_LIMIT));
            MAX_ITER = Integer.parseInt(getPropertyValue(KEY_MAX_ITER));
            START_ITER = Integer.parseInt(getPropertyValue(KEY_START_ITER));
            THRESHOLD = Double.parseDouble(getPropertyValue(KEY_THRESHOLD));
            SIGMA = Double.parseDouble(getPropertyValue(KEY_SIGMA));
            CHANNEL = getPropertyValue(KEY_CHANNEL);
            AUTO_ROI = getPropertyValue(KEY_AUTO_ROI);
            OPTIMIZER = getPropertyValue(KEY_OPTIMIZER);
            STEPS = Integer.parseInt(getPropertyValue(KEY_STEPS));
            MAX_IMAGE_WIDTH = Integer.parseInt(getPropertyValue(KEY_MAX_IMAGE_WIDTH));
            
        } catch (NumberFormatException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    
    public void openConfig()
    {
        String rootPath = new File("").getAbsolutePath() + "/config.xml";
        IJ.log("Loading config file " + rootPath);
        
        try {
            FileInputStream filex = new FileInputStream(rootPath);
            _properties.loadFromXML(filex);
            filex.close();
            
        } catch (FileNotFoundException ex) {
            IJ.log("Config file not found. Creating a default config.");
            _properties.setProperty("host", "localhost");
            _properties.setProperty("port", "50051");
            try {
                _properties.storeToXML(new FileOutputStream(rootPath), "generated");
            }
            catch (IOException ex2) {
                IJ.log("Config file default save error.");
            }
        }
        catch (IOException ex) {
            IJ.log("Config file error.");
        }
    }
    
    public DeepAutoFocusClient getRPCClient() {
        return _grpcclient;

    }

    private void connectToGRPC() {
        Logger.getLogger(DeepAutoFocus.class
                .getName()).log(Level.INFO, "DeepAutoFocusClient tries to connect...");
        //_grpcclient = new DeepAutoFocusClient(_properties.getProperty("host"), Integer.getInteger(_properties.getProperty("port")));
        _grpcclient = new DeepAutoFocusClient("localhost", 50051);

    }

    public void run(String arg) {
        connectToGRPC();
 
        while (!_isCalibCurveLoaded) {
            _isCalibCurveLoaded = openCalibrationCurveFileChooser();
        }

        _t0 = System.currentTimeMillis();

        if (arg.compareTo("options") == 0) {
            _studio.app().showAutofocusDialog();
        }

        if (_core == null) {
            // if core object is not set attempt to get its global handle
            _core = _studio.getCMMCore();
        }

        if (_core == null) {
            IJ.error("Unable to get Micro-Manager Core API handle.\n"
                    + "If this module is used as ImageJ plugin, Micro-Manager Studio must be running first!");
            return;
        }

        applySettings();

        //######################## START THE ROUTINE ###########
        try {
            boolean acquireAgain = true;
            double newZ;

            _zPositions.clear();

            IJ.log("Autofocus started.");
            boolean shutterOpen = _core.getShutterOpen();
            _core.setShutterOpen(true);
            boolean autoShutter = _core.getAutoShutter();
            _core.setAutoShutter(false);

            //########System setup##########
            if (!CHANNEL.equals(NOCHANNEL)) {
                _core.setConfig(_channelGroup, CHANNEL);
            }

            _core.waitForSystem();
            if (_core.getShutterDevice().trim().length() > 0) {
                _core.waitForDevice(_core.getShutterDevice());
            }

            int[] roiCoords = new int[4];
            roiCoords[0] = 0;
            roiCoords[1] = 0;
            roiCoords[2] = 0;
            roiCoords[3] = 0;
            if (AUTO_ROI.equals("Manual")) // Implementation of manual selection ROI using rectangle in interface
            {
                Rectangle _roi = null;
                if (WindowManager.getCurrentImage() == null || WindowManager.getCurrentImage().getProcessor() == null) {

                } else {
                    _roi = WindowManager.getCurrentImage().getProcessor().getRoi();
                }

                if (_roi != null) {
                    IJ.log("ROI found.");
                    roiCoords[0] = (int) _roi.x;
                    roiCoords[1] = (int) _roi.y;
                    roiCoords[2] = (int) (_roi.x + _roi.width);
                    roiCoords[3] = (int) (_roi.y + _roi.height);
                } else {
                    IJ.log("No ROI found.");
                }
            } else if (AUTO_ROI.equals("2D ROI")) // Implementation of 2D ROI using best focus regions
            {
                IJ.log("Asking 2D ROI to Python.");
                roiCoords[0] = -1;
            } else if (AUTO_ROI.equals("3D ROI")) // Implementation of automatic 3D ROI using best focus regions
            {
                IJ.log("Asking 3D ROI to Python.");
                if (!_studio.getAcquisitionManager().getAcquisitionSettings().relativeZSlice) {
                    IJ.log("Warning: z slices are not relative!");
                }
                roiCoords[0] = -2;
            } else if (AUTO_ROI.equals("2D and 3D ROI")) // Implementation of automatic 3D ROI using best focus regions
            {
                IJ.log("Asking 2D + 3D ROI to Python.");
                if (!_studio.getAcquisitionManager().getAcquisitionSettings().relativeZSlice) {
                    IJ.log("Warning: z slices are not relative!");
                }
                roiCoords[0] = -3;
            }
            _currentPosition = _core.getPosition(_core.getFocusDevice());
            _startPosition = _currentPosition;
            _nextPosition = _currentPosition;
            do {
                _tPrev = System.currentTimeMillis();
                goPosition(_nextPosition);
                _currentPosition = _core.getPosition(_core.getFocusDevice());
                if (snapSingleImage()) {

                    _zPositions.add(_currentPosition);

                    byte[] _imagePixels = getRawPixels();
                    if (sendBestFocusRequest(_imagePixels, _imageProcessor.getWidth(), _imageProcessor.getHeight(), roiCoords)) {
                        switch (_bestFocusResponse.getMessage()) {
                            case 0: // I don't have calibration curve
                                IJ.log("Server says that : No calibration curve loaded.");
                                _isCalibCurveLoaded = openCalibrationCurveFileChooser();
                                acquireAgain = true;
                                break;
                            case 1: // I need more images
                                acquireAgain = true;
                                _nextPosition = _bestFocusResponse.getZShift();
                                break;
                            case 2: // I found the best position but I need to acquire another one
                                acquireAgain = true;
                                goPosition(_bestFocusResponse.getZShift());
                                IJ.log("Server says that : I have found an optimal focus in " + _zPositions.size() + " steps.");
                                if (AUTO_ROI.contains("3D ROI")) {
                                    this.modifyZStack(_bestFocusResponse.getZShift(), _bestFocusResponse.getRoiMinZ(), _bestFocusResponse.getRoiMaxZ());
                                }
                                break;
                            case 3: // I go back to the starting position 
                                IJ.log("Server says that : will move to the optimal focus.");
                                acquireAgain = false;
                                goPosition(_bestFocusResponse.getZShift());
                                break;
                        }
                    } else {
                        IJ.log("Error on sendBestFocusRequest !!");
                        acquireAgain = false;
                    }
                }
                _tcur = System.currentTimeMillis() - _tPrev;
                //IJ.log("Single autofocus step time: " + _tcur);
            } while (acquireAgain);

            _currentPosition = _core.getPosition(_core.getFocusDevice());

            _core.setShutterOpen(shutterOpen);
            _core.setAutoShutter(autoShutter);

            IJ.log("Total autofocus step time: " + String.valueOf(System.currentTimeMillis() - _t0));
        } catch (Exception e) {
            _studio.logs().logError(e);
            IJ.error("Unknown error:" + e.toString());
        }
    }

    private void goPosition(double pos) {
        try {
            IJ.log("Goes to position " + pos);
            _core.setPosition(_core.getFocusDevice(), pos);
            _core.waitForDevice(_core.getFocusDevice());
            delay_time(10);
        } catch (Exception e) {
            _studio.logs().logError(e, "Cannot setPosition");
        }
    }

    private boolean openCalibrationCurveFileChooser() {
        JFileChooser jfc = new JFileChooser(FileSystemView.getFileSystemView().getHomeDirectory());
        jfc.setDialogTitle("Choose your calibration file:");
        jfc.setFileSelectionMode(JFileChooser.FILES_ONLY);

        int returnValue = jfc.showSaveDialog(null);
        if (returnValue == JFileChooser.APPROVE_OPTION) {
            if (jfc.getSelectedFile().isFile()) {
                String calibrationCurvePath = jfc.getSelectedFile().toString();
                return loadCalibrationCurve(calibrationCurvePath);
            }
            return false;
        }
        return false;
    }

    private void modifyZStack(double roi_reference_z, double roi_min_z, double roi_max_z) {
        try {
            SequenceSettings settings = _studio.getAcquisitionManager().getAcquisitionSettings();
            int count = settings.slices.size();

            double step_z = (roi_max_z - roi_min_z) / (double) count;

            if (step_z == 0.0) {
                throw new UnsupportedOperationException("Zero Z step size");
            }

            if (roi_min_z > roi_max_z) {
                step_z = -step_z;
            }

            settings.slices.clear();
            for (int i = 0; i < count; i++) {
                settings.slices.add(roi_min_z + i * step_z);
            }
            settings.relativeZSlice = true;
            settings.zReference = roi_reference_z;

            _studio.getAcquisitionManager().setAcquisitionSettings(settings);
            _core.waitForDevice(_core.getFocusDevice());
            delay_time(10);

        } catch (Exception e) {
            _studio.logs().logError(e, "Cannot modify Z stack");
        }
    }
    
    public static ImageProcessor makeProcessor(int type, int w, int h, Object imgArray) {
        switch (type) {
            case ImagePlus.GRAY8:
                return new ByteProcessor(w, h, (byte[]) imgArray, null);
            case ImagePlus.GRAY16:
                return new ShortProcessor(w, h, (short[]) imgArray, null);
            case ImagePlus.GRAY32:
                return new FloatProcessor(w, h, (float[]) imgArray, null);
            case ImagePlus.COLOR_RGB:
                // Micro-Manager RGB32 images are generally composed of byte
                // arrays, but ImageJ only takes int arrays.
                if (imgArray instanceof byte[]) {
                    imgArray = convertRGB32BytesToInt((byte[]) imgArray);
                }
                return new ColorProcessor(w, h, (int[]) imgArray);
            default:
                return null;
        }
    }

    public static ImageProcessor makeProcessor(TaggedImage taggedImage) {
        final JSONObject tags = taggedImage.tags;
        try {
            return makeProcessor(MDUtils.getIJType(tags), MDUtils.getWidth(tags),
                    MDUtils.getHeight(tags), taggedImage.pix);
        } catch (IllegalArgumentException e) {
            ReportingUtils.logError(e);
            return null;
        } catch (JSONException e) {
            ReportingUtils.logError(e);
            return null;
        }
    }
    
    private boolean snapSingleImage() {

        try {
            _core.snapImage();
            TaggedImage i = _core.getTaggedImage();
            _imageProcessor = makeProcessor(i);
            
            if ((int)_core.getImageWidth() > MAX_IMAGE_WIDTH)
            {
                IJ.log("Factor of size " + _factor);
                _imageProcessor = _imageProcessor.resize(MAX_IMAGE_WIDTH);
                _factor = (double)_core.getImageWidth() / (double)MAX_IMAGE_WIDTH;
            }
            
        } catch (Exception e) {
            IJ.log("Error in snapSingleImage");
            IJ.error(e.getMessage());
            IJ.error("Error in snapSingleImage");
            return false;
        }

        return true;
    }

    private byte[] getRawPixels() {
        int width = _imageProcessor.getWidth();
        int height = _imageProcessor.getHeight();
        
        if (_byteDepth == 0)
            _byteDepth = (int)_core.getBytesPerPixel();
        
        byte[] allRawPixels = new byte[width * height * _byteDepth];
        
        try {
            assert(_byteDepth-1 == ImagePlus.GRAY16 || _byteDepth-1 == ImagePlus.GRAY8);
            if (_byteDepth-1 == ImagePlus.GRAY8)
            {
            // 8 bit images
                byte[] rawPixelByte = (byte[]) ((ByteProcessor)_imageProcessor).getPixels();
                int rawPixelByteLength = rawPixelByte.length;
                _studio.logs().logMessage("Our image size : " + rawPixelByteLength);
                System.arraycopy(rawPixelByte, 0, allRawPixels, 0, rawPixelByteLength);
            }
            else if (_byteDepth-1 == ImagePlus.GRAY16)
            {
            // 16 bit images
                short[] rawPixelShort = (short[]) ((ShortProcessor)_imageProcessor).getPixels();
                int rawPixelShortLength = rawPixelShort.length*2;
                _studio.logs().logMessage("Our image size : " + rawPixelShortLength);
                for (int j = 0; j < rawPixelShortLength; j += 2) {
                    allRawPixels[j] = (byte)(rawPixelShort[j/2] & 0xff);
                    allRawPixels[j+1] = (byte)((rawPixelShort[j/2] >> 8) & 0xff);
                }
            }

        } catch (Exception ex) {
            Logger.getLogger(DeepAutoFocus.class
                    .getName()).log(Level.SEVERE, null, ex);
        }
        return allRawPixels;
    }

    private boolean sendBestFocusRequest(byte[] allRawPixels, int width, int height, int[] roiCoords) {

        Integer[] roiCoords_obj = ArrayUtils.toObject(roiCoords);
        List<Integer> roiCoords_list = Arrays.asList(roiCoords_obj);
        
        // We need to scale the ROI from the size reduction
        roiCoords_list.replaceAll(s -> Integer.valueOf((int)Math.round((double)s.doubleValue()/_factor)));
        
        BestFocusRequest bestFocusReq = BestFocusRequest.newBuilder()
                .setImageList(ByteString.copyFrom(allRawPixels)).setBytesPerPixel(_byteDepth)
                .setHeight(height).setWidth(width).setZPosition(_zPositions.get(_zPositions.size()-1)).setRoi3DNumSigma(SIGMA)
                .addAllRoiCoords(roiCoords_list).setThreshold(THRESHOLD).setMinIter(START_ITER)
                .setMaxIter(MAX_ITER).setMinLimit(MIN_LIMIT)
                .setOptimizer(OPTIMIZER).setSteps(STEPS)
                .setMaxLimit(MAX_LIMIT).build();
        try {
            _bestFocusResponse = getRPCClient().blockingStub.startAutofocus(bestFocusReq);
        } catch (StatusRuntimeException e) {
            Logger.getLogger(DeepAutoFocus.class
                    .getName()).log(Level.WARNING, "RPC failed: {0}", e.getStatus());
            return false;
        }
        return true;
    }

    private boolean loadCalibrationCurve(String calibrationCurvePath) {
        CurveRequest _curveReq = CurveRequest.newBuilder().setCalibrationCurvePicklePath(calibrationCurvePath).build();
        CurveResponse _curveRes;
        try {
            _curveRes = getRPCClient().blockingStub.loadCalibration(_curveReq);

        } catch (StatusRuntimeException e) {
            Logger.getLogger(DeepAutoFocus.class
                    .getName()).log(Level.WARNING, "RPC failed: {0}", e.getStatus());
            return false;

        }
        Logger.getLogger(DeepAutoFocus.class
                .getName()).log(Level.INFO, "Curve Response: {0}", _curveRes.toString());
        return _curveRes.getPickleFound();
    }

    @Override
    public double computeScore(final ImageProcessor impro) {
        return 0.0;
    }

    //waiting    
    private void delay_time(double delay) {
        Date date = new Date();
        long sec = date.getTime();
        while (date.getTime() < sec + delay) {
            date = new Date();
        }
    }

    @Override
    public double fullFocus() {
        run("silent");
        return 0;
    }

    @Override
    public String getVerboseStatus() {
        return "OK";
    }

    @Override
    public PropertyItem[] getProperties() {
        // use default dialog
        // make sure we have the right list of channels

        _channelGroup = _core.getChannelGroup();
        StrVector channels = _core.getAvailableConfigs(_channelGroup);
        String allowedChannels[] = new String[(int) channels.size() + 1];
        allowedChannels[0] = NOCHANNEL;

        try {
            PropertyItem p = getProperty(KEY_CHANNEL);
            boolean found = false;
            for (int i = 0; i < channels.size(); i++) {
                allowedChannels[i + 1] = channels.get(i);
                if (p.value.equals(channels.get(i))) {
                    found = true;
                }
            }
            p.allowed = allowedChannels;
            if (!found) {
                p.value = allowedChannels[0];
            }
            setProperty(p);
        } catch (MMException e1) {
            // TODO Auto-generated catch block
            e1.printStackTrace();
        }

        return super.getProperties();
    }

    @Override
    public double getCurrentFocusScore() {
        // TODO Auto-generated method stub
        return 0;
    }

    @Override
    public int getNumberOfImages() {
        // TODO Auto-generated method stub
        return 0;
    }

    public void setContext(Studio app) {
        _studio = app;
        _core = app.getCMMCore();
        _studio.events().registerForEvents(this);
    }

    @Override
    public String getName() {
        return "DeepAutoFocus";
    }

    @Override
    public String getHelpText() {
        return AF_DEVICE_NAME;
    }

    @Override
    public String getCopyright() {
        return "Idiap Research Institute, 2019";
    }

    @Override
    public String getVersion() {
        return "1.0";
    }

    @Override
    public double incrementalFocus() throws Exception {
        return 0.0;
    }
}
