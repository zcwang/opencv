package org.opencv.android.services.calibration.sensor;

import java.util.List;
import java.util.ListIterator;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.services.CameraView;
import org.opencv.android.services.Utils;
import org.opencv.android.services.calibration.CameraCalibrationResult;
import org.opencv.android.services.calibration.CameraInfo;
import org.opencv.android.services.sensor.SensorCalibrationResult;
import org.opencv.android.services.sensor.SensorRecorder;
import org.opencv.core.Mat;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.hardware.Camera.Size;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.view.ContextMenu;
import android.view.ContextMenu.ContextMenuInfo;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.SubMenu;
import android.view.SurfaceView;
import android.view.View;
import android.view.View.OnTouchListener;
import android.view.WindowManager;
import android.widget.EditText;
import android.widget.ProgressBar;
import android.widget.Toast;
import android.widget.ToggleButton;

public class CalibrationActivity extends Activity implements CvCameraViewListener2, OnTouchListener {
    private static final String TAG = "Activity";

    private static final String CALIBRATE_ACTION = SensorCalibrationResult.CALIBRATE_ACTION;

    private CameraInfo mRequestedCameraInfo = null;

    private CameraInfo mCameraInfo = new CameraInfo();
    private SensorCalibrator mCalibrator;
    private SensorRecorder mSensorRecorder;

    private CameraView mOpenCvCameraView;

    private ProgressBar mProgressBar;
    private ToggleButton mCalibrateButton;

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

    protected void sendResponse(SensorCalibrationResult calibrationResult) {
        if (CALIBRATE_ACTION.equals(getIntent().getAction()) && mRequestedCameraInfo != null) {
            Bundle extras = getIntent().getExtras();
            String responseAction = CalibrationActivity.class.getName() + "!response";
            if (extras != null) {
                responseAction = extras.getString("responseAction", responseAction);
            }
            Intent intent = new Intent(responseAction);
            String response = null;
            if (calibrationResult != null) {
                if (mRequestedCameraInfo.equals(calibrationResult.mCameraInfo)) {
                    response = calibrationResult.getJSON().toString();
                }
            }
            if (response != null)
                intent.putExtra("response", response);
            Log.i(TAG, "Send " + (response == null ? "CANCEL" : "VALID") + " response broadcast: " + responseAction);
            sendBroadcast(intent);
        }
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "onCreate");
        super.onCreate(savedInstanceState);

        CameraInfo startupCameraInfo = new CameraInfo();
        startupCameraInfo.setPreferredResolution(this);
        if (startupCameraInfo.mWidth > 1280) startupCameraInfo.mWidth = 1280;
        if (startupCameraInfo.mHeight > 720) startupCameraInfo.mHeight = 720;
        if (CALIBRATE_ACTION.equals(getIntent().getAction())) {
            Bundle extras = getIntent().getExtras();
            if (extras != null) {
                startupCameraInfo.readFromBundle(extras);
                mRequestedCameraInfo = startupCameraInfo;
                SensorCalibrationResult result = new SensorCalibrationResult(mRequestedCameraInfo);
                if (extras.getBoolean("force", false) == false && result.tryLoad(this)) {
                    Log.e(TAG, "Return loaded calibration result");
                    sendResponse(result);
                    finish();
                    return;
                }
            } else {
                Log.e(TAG, "No camera info. Ignore invalid request");
                finish();
            }
        }

        mSensorRecorder = new SensorRecorder(this);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.surface_view);
        mOpenCvCameraView = (CameraView)findViewById(R.id.java_surface_view);
        mOpenCvCameraView.setResolution(startupCameraInfo.mWidth, startupCameraInfo.mHeight);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        mProgressBar = (ProgressBar)findViewById(R.id.progressBar);
        mCalibrateButton = (ToggleButton)findViewById(R.id.calibrate_btn);

        registerForContextMenu(mOpenCvCameraView);
    }

    @Override
    protected void onStop() {
        sendResponse((mCalibrator == null) ? null : mCalibrator.getSensorCalibrationResult());
        super.onStop();
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
        mCalibrateButton.setChecked(false);
        mCalibrateButton.setEnabled(false);
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
        mSensorRecorder.start();
        if (mCalibrator != null && mCalibrator.isCameraCalibrated())
            mCalibrateButton.setEnabled(true);
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
                    final Runnable self = this;
                    calibrationResult.requestData(context, false, new CameraCalibrationResult.RequestCallback() {
                        @Override
                        public void onSuccess() {
                            mCalibrator = new SensorCalibrator(mCameraInfo, mSensorRecorder);
                            mCalibrator.setCameraCalibrationResult(calibrationResult);
                            mCalibrateButton.setEnabled(true);
                            SensorCalibrationResult calibrationResult = new SensorCalibrationResult(mCameraInfo);
                            if (calibrationResult.tryLoad(CalibrationActivity.this)) {
                                mCalibrator.setSensorCalibrationResult(calibrationResult);
                            }
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
    public void onCreateContextMenu(ContextMenu menu, View v, ContextMenuInfo menuInfo) {
        menu.setHeaderTitle("Sensor calibration");
        onCreateOptionsMenu(menu);
    }

    @Override
    public boolean onContextItemSelected(MenuItem item) {
        return onOptionsItemSelected(item);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.calibration, menu);

        try {
            mResolutionList = mOpenCvCameraView.getResolutionList();
            mResolutionMenu = menu.addSubMenu("Resolution");
            mResolutionMenuItems = new MenuItem[mResolutionList.size()];

            ListIterator<Size> resolutionItr = mResolutionList.listIterator();
            int idx = 0;
            while(resolutionItr.hasNext()) {
                Size element = resolutionItr.next();
                mResolutionMenuItems[idx] = mResolutionMenu.add(MENU_GROUP_RESOLUTION, idx, Menu.NONE,
                        Integer.valueOf(element.width).toString() + "x" + Integer.valueOf(element.height).toString());
                idx++;
            }
        } catch (Exception e) {
            e.printStackTrace();
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
                        final Runnable self = this;
                        calibrationResult.requestData(context, force, new CameraCalibrationResult.RequestCallback() {
                            @Override
                            public void onSuccess() {
                                mCalibrator = new SensorCalibrator(mCameraInfo, mSensorRecorder);
                                mCalibrator.setCameraCalibrationResult(calibrationResult);
                                mCalibrateButton.setEnabled(true);
                                SensorCalibrationResult calibrationResult = new SensorCalibrationResult(mCameraInfo);
                                if (calibrationResult.tryLoad(CalibrationActivity.this)) {
                                    mCalibrator.setSensorCalibrationResult(calibrationResult);
                                }
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
            case R.id.edit:
            {
                AlertDialog.Builder builder = new AlertDialog.Builder(this);
                LayoutInflater inflater = this.getLayoutInflater();

                final View view = inflater.inflate(R.layout.sensor_calibration_values, null);
                builder.setView(view)
                    .setPositiveButton("Save",
                            new DialogInterface.OnClickListener() {
                                @Override
                                public void onClick(DialogInterface dialog, int id) {
                                    try {
                                        Double pitchDiff = Double.valueOf(((EditText)view.findViewById(R.id.editText_Pitch)).getText().toString());
                                        Double rollDiff = Double.valueOf(((EditText)view.findViewById(R.id.editText_Roll)).getText().toString());
                                        SensorCalibrationResult res = new SensorCalibrationResult(mCameraInfo);
                                        res.init(Utils.toRad(pitchDiff), Utils.toRad(rollDiff));
                                        if (mCalibrator != null)
                                            mCalibrator.setSensorCalibrationResult(res);
                                        res.save(CalibrationActivity.this);
                                        res.saveToStorage(CalibrationActivity.this);
                                        Toast.makeText(CalibrationActivity.this, "New values saved", Toast.LENGTH_LONG).show();
                                        dialog.dismiss();
                                        if (CALIBRATE_ACTION.equals(CalibrationActivity.this.getIntent().getAction())) {
                                            Log.e(TAG, "Return sensor calibration result");
                                            sendResponse(res);
                                            finish();
                                        }
                                    } catch (Exception e) {
                                        e.printStackTrace();
                                        Toast.makeText(CalibrationActivity.this, "Can't save values", Toast.LENGTH_LONG).show();
                                    }
                                }
                            })
                    .setNegativeButton("Cancel",
                            new DialogInterface.OnClickListener() {
                                @Override
                                public void onClick(DialogInterface dialog, int id) {
                                    dialog.cancel();
                                }
                            });
                AlertDialog ad = builder.create();
                SensorCalibrationResult res = (mCalibrator != null) ? mCalibrator.getSensorCalibrationResult() : null;
                if (res != null) {
                    ((EditText)view.findViewById(R.id.editText_Pitch)).setText("" + Utils.toDeg(res.mPitchDiff));
                    ((EditText)view.findViewById(R.id.editText_Roll)).setText("" + Utils.toDeg(res.mRollDiff));
                }
                ad.show();
            }
        }
        return false;
    }

    @Override
    public boolean onTouch(View v, MotionEvent event) {
        Log.d(TAG, "onTouch invoked");
        return false;
    }

    public void onStartClick(View v) {
        if (mCalibrateButton.isChecked()) {
            if (mCalibrator != null) {
                mProgressBar.setVisibility(View.VISIBLE);
                mCalibrator.start(new SensorCalibrator.CompletionCallback() {

                    @Override
                    public void onProgress(int value, int total) {
                        mProgressBar.setMax(total);
                        mProgressBar.setProgress(value);
                    }

                    @Override
                    public void onCompleted() {
                        CalibrationActivity.this.runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                mProgressBar.setVisibility(View.GONE);
                                mCalibrateButton.setChecked(false);
                                SensorCalibrationResult res = mCalibrator.getSensorCalibrationResult();
                                res.save(CalibrationActivity.this);
                                res.saveToStorage(CalibrationActivity.this);
                                if (CALIBRATE_ACTION.equals(CalibrationActivity.this.getIntent().getAction())) {
                                    Log.e(TAG, "Return sensor calibration result");
                                    sendResponse(res);
                                    finish();
                                }
                            }
                        });
                    }
                });
            }
        } else {
            if (mCalibrator != null) {
                mCalibrator.stop();
                mProgressBar.setVisibility(View.GONE);
            }
        }
    }
}
