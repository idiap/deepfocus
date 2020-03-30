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

import com.google.protobuf.ByteString;
import ij.IJ;
import ij.ImagePlus;
import ij.process.ByteProcessor;
import ij.process.ColorProcessor;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import ij.process.ShortProcessor;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.stub.StreamObserver;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import mmcorej.CMMCore;
import org.micromanager.Studio;

import java.util.List;
import java.util.Objects;
import java.util.Observable;
import java.util.Properties;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import javax.imageio.ImageIO;
import org.micromanager.data.Coords;
import org.micromanager.data.Datastore;
import org.micromanager.data.DatastoreFrozenException;
import org.micromanager.data.DatastoreRewriteException;
import org.micromanager.data.internal.DefaultImage;
import javax.swing.JOptionPane;
import mmcorej.DeviceType;
import mmcorej.StrVector;
import mmcorej.TaggedImage;
import org.apache.commons.lang3.ArrayUtils;
import org.json.JSONObject;
import org.micromanager.data.Image;
import org.micromanager.data.ImageJConverter;
import org.micromanager.data.Metadata;
import org.micromanager.data.SummaryMetadata;
import org.micromanager.display.DisplayWindow;
import org.micromanager.internal.diagnostics.EDTHangLogger;
import org.micromanager.internal.utils.MDUtils;
import org.micromanager.internal.utils.ReportingUtils;
import static org.micromanager.internal.utils.imageanalysis.ImageUtils.convertRGB32BytesToInt;

public class DeepFocusScripting extends Observable {

    private final Studio _studio;
    private final CMMCore _core;
    private final ZMovement _zMovement;
    private StrVector _driveNames;
    private boolean _isError;
    private String _driveName;
    private DeepFocusGUIClient _grpcclient = null;
    private String _calibCurveImageFilePath;
    private byte[] _allRawPixels;
    private int _byteDepth;
    private Properties _properties = new Properties();

    /**
     * Create the acquisition scripting interface
     *
     * @param studio the MMStudio object for the current session
     */
    public DeepFocusScripting(Studio studio) {
        _studio = studio;
        _core = studio.getCMMCore();
        openConfig();

        _zMovement = new ZMovement(studio);

        setCalibCurveImageFilePath("");
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
                IJ.error("Config file default save error.");
            }
        }
        catch (IOException ex) {
            IJ.error("Config file error.");
        }
    }
    
    public DeepFocusGUIClient getRPCClient() {
        return _grpcclient;
    }

    private void connectToGRPC() {
        _studio.logs().logMessage("DeepFocusGUIClient tries to connect... host=" + _properties.getProperty("host") + " port = "+_properties.getProperty("port"));
        if (_grpcclient != null)
        {
            try {
                _grpcclient.shutdown();
            } catch (InterruptedException ex) {
                Logger.getLogger(DeepFocusScripting.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        //_grpcclient = new DeepFocusGUIClient(_properties.getProperty("host"), Integer.getInteger(_properties.getProperty("port")));
        _grpcclient = new DeepFocusGUIClient("localhost",50051);
    }

    public void setZMotor(String name)
    {
        _driveNames = _core.getLoadedDevicesOfType(DeviceType.StageDevice);
        _isError = _driveNames.isEmpty();
        if (_isError) {
            _studio.alerts().postAlert("Z Drives Alert!",
                    DeepFocusScripting.class, "Cannot find Z Drive");
        } else {
            _driveName = name;
            _zMovement.setZStage(_driveName);
        }
    }

    /**
     * Move the Z stage in micrometer to a relative position
     *
     * @param um
     */
    public void moveZRelative(double um) {
        try {
            _zMovement.moveZ(um, true);
        } catch (Exception e) {
            _studio.logs().logError(e);
        }
    }

    /**
     * Move the Z stage in micrometer to an absolute position
     *
     * @param um
     */
    public void moveZAbsolute(double um) {
        try {
            _zMovement.moveZ(um, false);
        } catch (Exception e) {
            _studio.logs().logError(e);
        }
    }

    /**
     * Get the current position of the Z stage in micrometer
     *
     * @return
     */
    public Double getZPosition() {
        try {
            return _zMovement.getZ();
        } catch (Exception e) {
            _studio.logs().logError(e);
        }
        return null;
    }

    /**
     * Wait for a certain amount of time
     *
     * @param seconds
     */
    public void delay(double seconds) {
        _core.sleep(1000 * seconds);
    }

    /**
     * Snap a single image from the first camera. The snapped image will be
     * saved internally, using the given coordinates
     *
     * @param coordsBuilder
     * @return the snapped image
     */
    public TaggedImage snapImage() {
        try {
            _core.waitForDevice(_core.getCameraDevice());
            _core.snapImage();
            TaggedImage img = _core.getTaggedImage();
            return img;
        } catch (Exception ex) {
            Logger.getLogger(DeepFocusScripting.class.getName()).log(Level.SEVERE, null, ex);
            return null;
        }
    }

    public String getCalibCurveImageFilePath() {
        return _calibCurveImageFilePath;
    }

    public void setCalibCurveImageFilePath(String newCalibCurveImageFilePath) {
        _calibCurveImageFilePath = newCalibCurveImageFilePath;
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
        } catch (Exception e) {
            ReportingUtils.logError(e);
            return null;
        }
    }
    
    /**
     * Aquire a Z stack by specifying start, stop and layer thickness.This
 procedure uses the internal storage and coordinates builder. Only the z
 coordinate is modified through the operation. The acquisition data are
 stored in the internal data store.
     *
     * @param zStart in um
     * @param zStop in um
     * @param stepWidth in um
     * @param calibrationCurvePathString
     * @throws java.lang.InterruptedException
     * @throws java.io.IOException
     */
    public void acquireSendZStack(double zStart, double zStop, double stepWidth, String calibrationCurvePathString) throws InterruptedException, IOException {
        try {
            notifyObservers(new String[]{"1", "Acquisition starting..."});
            EDTHangLogger.stopDefault();
            _core.setAutoShutter(false);
            _studio.live().setLiveMode(false);
            //DEBUG w/0 grpc
            List<ImageProcessor> imageList = new ArrayList<ImageProcessor>();

            moveZAbsolute(zStart);
            int steps = 1+(int) Math.floor((zStop - zStart) / stepWidth);
            stepWidth = stepWidth * (double) Math.signum(steps);
            steps = Math.abs(steps);

            List<Double> allZPos = new ArrayList<Double>();
            _byteDepth = (int)_core.getBytesPerPixel();

                 
            for (int layer = 0; layer < steps; layer++) {
                double goal = zStart + layer * stepWidth;
                moveZAbsolute(goal);
                TaggedImage i = snapImage();
                ImageProcessor ip = makeProcessor(i);
                if ((int)_core.getImageWidth() > 1024)
                    ip = ip.resize(1024);
                ip.setRoi((int)ip.getWidth()/3, (int)ip.getHeight()/3, (int)ip.getWidth()/3, (int)ip.getHeight()/3);
                ip = ip.crop();
                // If the motor does not move, we don't add the image
                _studio.logs().logMessage("allZpos " + allZPos.toString());
                _studio.logs().logMessage("layer " + layer);
                if (layer == 0 || !Objects.equals(allZPos.get(allZPos.size()-1), getZPosition()))
                {
                    allZPos.add(getZPosition());
                    imageList.add(ip);
                    _studio.logs().logMessage("Moving to Position : " + goal);
                    _studio.logs().logMessage("Current Position : " + getZPosition());
                }
            }

            setChanged();
            notifyObservers(new String[]{"1", "Acquisition completed! We have " + String.valueOf(imageList.size())+" images in datastore."});

            int imageListSize = imageList.size();

            int width = imageList.get(0).getWidth();
            int height = imageList.get(0).getHeight();
            notifyObservers(new String[]{"1", "Our images are " + _byteDepth + " byte(s) per pixel."});
            _allRawPixels = new byte[imageListSize * width * height * _byteDepth];
            for (int i = 0; i < imageList.size(); i++) {
                assert(_byteDepth-1 == ImagePlus.GRAY16 || _byteDepth-1 == ImagePlus.GRAY8);
                if (_byteDepth-1 == ImagePlus.GRAY8)
                {
                // 8 bit images
                    byte[] rawPixelByte = (byte[]) ((ByteProcessor)imageList.get(i)).getPixels();
                    int rawPixelByteLength = rawPixelByte.length;
                    _studio.logs().logMessage("Our image size : " + rawPixelByteLength);
                    for (int j = 0; j < rawPixelByteLength; j++) {
                        _allRawPixels[i * rawPixelByteLength + j] = rawPixelByte[j];
                    }
                }
                else if (_byteDepth-1 == ImagePlus.GRAY16)
                {
                // 16 bit images
                    short[] rawPixelShort = (short[]) ((ShortProcessor)imageList.get(i)).getPixels();
                    int rawPixelShortLength = rawPixelShort.length*2;
                    _studio.logs().logMessage("Our image size : " + rawPixelShortLength);
                    for (int j = 0; j < rawPixelShortLength; j += 2) {
                        _allRawPixels[i * rawPixelShortLength + j] = (byte)(rawPixelShort[j/2] & 0xff);
                        _allRawPixels[i * rawPixelShortLength + j+1] = (byte)((rawPixelShort[j/2] >> 8) & 0xff);
                    }
                }
            }
            sendCalibrationRequest(width, height, 64, allZPos, calibrationCurvePathString);

            EDTHangLogger.startDefault(_core, 4500, 1000);
        } catch (Exception ex) {
            _studio.logs().logError(ex, "Error at acquireSendZStack");

            _studio.alerts().postAlert("Error at acquireSendZStack",
                    DeepFocusScripting.class, "Exception: " + ex);
        }
    }

    public void sendCalibrationRequest(int width, int height, int steps, List zPos, String calibrationCurvePathString) {
        connectToGRPC();

        CalibrationRequest.Builder requestBuilder = CalibrationRequest.newBuilder().setHeight(height).
                setWidth(width).setSteps(steps).addAllZPositions(zPos).setCalibCurvePathway(calibrationCurvePathString);
        requestBuilder.setBytesPerPixel(_byteDepth);
        requestBuilder.setImageList(ByteString.copyFrom(_allRawPixels));
        CalibrationRequest request = requestBuilder.build();
        try {
            setChanged();
            notifyObservers(new String[]{"2", "Waiting for processing..."});

            getRPCClient()._asyncStub.withCompression("gzip").createCalibrationCurve(request, new StreamObserver<CalibrationCurve>() {
                long lastCall = System.nanoTime();

                @Override
                public void onNext(CalibrationCurve response) {

                    setCalibCurveImageFilePath(response.getCalibCurveImageFilePath());
                    
                    setChanged();
                    notifyObservers(new String[]{"3", "Calibration is done!"});

                    File calibCurveFile = new File(getCalibCurveImageFilePath());

                    java.awt.Image calibCurveTemp = null;
                    try {
                        setChanged();
                        notifyObservers(new String[]{"4", "Calibration Curve is displayed"});

                        calibCurveTemp = ImageIO.read(calibCurveFile);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    ImagePlus calibCurveImagePlus = new ImagePlus();
                    calibCurveImagePlus.setImage(calibCurveTemp);
                    calibCurveImagePlus.show();
                }

                @Override
                public void onError(Throwable t) {
                    Status status = Status.fromThrowable(t);
                    _studio.logs().logError("Encountered an error in createCalibrationCurve. Status is " + status);
                    t.printStackTrace();
                    
                    setChanged();
                    notifyObservers(new String[]{"5", "GRPC Error :" + status.toString()});
                }

                @Override
                public void onCompleted() {
                    setChanged();
                    notifyObservers(new String[]{"6", "Completed!"});
                }
            });

        } catch (StatusRuntimeException e) {
            _studio.logs().logError(e, "analyseImage RPC failed");
            _studio.alerts().postAlert("Error at analyseImage: RPC failed",
                    DeepFocusScripting.class, "Exception: {0}" + e.getStatus());
        }
    }
}
