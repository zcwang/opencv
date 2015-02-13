package org.opencv.android.services.calibration.sensor;

import java.util.List;
import java.util.ListIterator;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.services.CameraView;
import org.opencv.android.services.calibration.CameraCalibrationResult;
import org.opencv.android.services.calibration.CameraInfo;
import org.opencv.android.services.sensor.SensorRecorder;
import org.opencv.core.Mat;

import android.app.Activity;
import android.content.Context;
import android.hardware.Camera.Size;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.SubMenu;
import android.view.SurfaceView;
import android.view.View;
import android.view.View.OnTouchListener;
import android.view.WindowManager;
import android.widget.Toast;

public class CalibrationActivity extends Activity implements CvCameraViewListener2, OnTouchListener {
    private static final String TAG = "Activity";

    private static final String CALIBRATE_ACTION = "org.opencv.android.services.calibration.sensor.calibrate_orientation";

    private CameraInfo mCameraInfo = new CameraInfo();
    private SensorCalibrator mCalibrator;
    private SensorRecorder mSensorRecorder;

    private CameraView mOpenCvCameraView;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            new Handler().postDelayed(new Runnable() {
                                @Override
                                public void run() {
                                    mOpenCvCameraView.enableView();
                                    mOpenCvCameraView.setOnTouchListener(CalibrationActivity.this);
                                }
                            }, 1000);
                        }
                    });
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "onCreate");
        super.onCreate(savedInstanceState);

        CameraInfo startupCameraInfo = new CameraInfo();
        startupCameraInfo.setPreferredResolution(this);
        if (startupCameraInfo.mWidth > 1280) startupCameraInfo.mWidth = 1280;
        if (startupCameraInfo.mHeight > 720) startupCameraInfo.mHeight = 720;
        if (CALIBRATE_ACTION.equals(getIntent().getAction())) {
            finish(); // TODO !!!!
        }

        mSensorRecorder = new SensorRecorder(this);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.surface_view);
        mOpenCvCameraView = (CameraView) findViewById(R.id.java_surface_view);
        mOpenCvCameraView.setResolution(startupCameraInfo.mWidth, startupCameraInfo.mHeight);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
        mSensorRecorder.stop();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
        mSensorRecorder.start();
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        String text = Integer.valueOf(width).toString() + "x" + Integer.valueOf(height).toString();
        Toast.makeText(this, text, Toast.LENGTH_SHORT).show();
        if (mCameraInfo.mWidth != width || mCameraInfo.mHeight != height) {
            if (mCalibrator != null) {
                Toast.makeText(this, "Camera resolution changed. Recreate calibrator", Toast.LENGTH_LONG).show();
                mCalibrator = null;
            }
            mCameraInfo = new CameraInfo();
            mCameraInfo.mCameraIndex = mOpenCvCameraView.getCameraIndex();
            mCameraInfo.mWidth = width;
            mCameraInfo.mHeight = height;
            final CameraCalibrationResult calibrationResult = new CameraCalibrationResult(mCameraInfo);
            final Context context = this;
            final Runnable request = new Runnable() {
                @Override
                public void run() {
                    mOpenCvCameraView.disableView();
                    final Runnable self = this;
                    calibrationResult.requestData(context, false, new CameraCalibrationResult.RequestCallback() {
                        @Override
                        public void onSuccess() {
                            mCalibrator = new SensorCalibrator(mCameraInfo, mSensorRecorder);
                            mCalibrator.setCalibrationResult(calibrationResult);
                        }
                        @Override
                        public void onFailure() {
                            Toast.makeText(context, "No camera calibration data", Toast.LENGTH_LONG).show();
                            //self.run();
                        }
                    });
                }
            };
            request.run();
        }
    }

    @Override
    public void onCameraViewStopped() {
    }

    @Override
    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        Mat gray = inputFrame.gray();
        Mat rgba = inputFrame.rgba();
        if (mCalibrator != null)
            mCalibrator.processFrame(gray, rgba);
        return rgba;
    }

    private static final int MENU_GROUP_RESOLUTION = 10;
    private List<Size> mResolutionList;
    private MenuItem[] mResolutionMenuItems;
    private SubMenu mResolutionMenu;

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.calibration, menu);

        mResolutionMenu = menu.addSubMenu("Resolution");
        mResolutionList = mOpenCvCameraView.getResolutionList();
        mResolutionMenuItems = new MenuItem[mResolutionList.size()];

        ListIterator<Size> resolutionItr = mResolutionList.listIterator();
        int idx = 0;
        while(resolutionItr.hasNext()) {
            Size element = resolutionItr.next();
            mResolutionMenuItems[idx] = mResolutionMenu.add(MENU_GROUP_RESOLUTION, idx, Menu.NONE,
                    Integer.valueOf(element.width).toString() + "x" + Integer.valueOf(element.height).toString());
            idx++;
        }

        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
        if (item.getGroupId() == MENU_GROUP_RESOLUTION)
        {
            int id = item.getItemId();
            Size resolution = mResolutionList.get(id);
            mOpenCvCameraView.setResolution(resolution.width, resolution.height);
            // TODO Fix OpenCV SDK to call this callback
            onCameraViewStarted(resolution.width, resolution.height);
            return true;
        }
        switch (item.getItemId()) {
            case R.id.calibrate:
            {
                mCalibrator = null;
                final CameraCalibrationResult calibrationResult = new CameraCalibrationResult(mCameraInfo);
                final Context context = this;
                final Runnable request = new Runnable() {
                    public boolean force = true;
                    @Override
                    public void run() {
                        mOpenCvCameraView.disableView();
                        final Runnable self = this;
                        calibrationResult.requestData(context, force, new CameraCalibrationResult.RequestCallback() {
                            @Override
                            public void onSuccess() {
                                mCalibrator = new SensorCalibrator(mCameraInfo, mSensorRecorder);
                                mCalibrator.setCalibrationResult(calibrationResult);
                            }
                            @Override
                            public void onFailure() {
                                force = false;
                                Toast.makeText(context, "No camera calibration data", Toast.LENGTH_LONG).show();
                                //self.run();
                            }
                        });
                    }
                };
                request.run();
                return true;
            }
        }
        return false;
    }

    @Override
    public boolean onTouch(View v, MotionEvent event) {
        Log.d(TAG, "onTouch invoked");
        if (mCalibrator != null) {
            // TODO
        }
        return false;
    }
}
