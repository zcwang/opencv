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
import org.opencv.core.Mat;

import android.app.Activity;
import android.app.ProgressDialog;
import android.content.Intent;
import android.content.res.Resources;
import android.hardware.Camera.Size;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.SubMenu;
import android.view.View;
import android.view.View.OnTouchListener;
import android.view.WindowManager;
import android.widget.Toast;

public class CalibrationActivity extends Activity implements CvCameraViewListener2, OnTouchListener {
    private static final String TAG = "Activity";

    private static final String CALIBRATE_ACTION = "org.opencv.android.services.calibration.sensor.calibrate_orientation";

    private CameraInfo mCameraInfo = new CameraInfo();
    private CameraCalibrationResult mCameraCalibrationResult = new CameraCalibrationResult(mCameraInfo);
    private SensorCalibrator mCalibrator;

    private CameraView mOpenCvCameraView;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                    mOpenCvCameraView.setOnTouchListener(CalibrationActivity.this);
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

        if (CALIBRATE_ACTION.equals(getIntent().getAction())) {
            Bundle extras = getIntent().getExtras();
            if (extras != null) {
                mCameraInfo.readFromBundle(extras);
                CameraCalibrationResult result = new CameraCalibrationResult(mCameraInfo);
                if (result.tryLoad(this)) {
                    Log.e(TAG, "Return loaded calibration result");
                    Intent data = new Intent();
                    data.putExtra("result", result.getJSON().toString());
                    setResult(RESULT_OK, data);
                    finish();
                    return;
                }
            } else {
                Log.e(TAG, "No camera info. Ignore invalid request");
                Intent data = new Intent();
                setResult(RESULT_CANCELED, data);
                finish();
            }
        }

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.surface_view);
        mOpenCvCameraView = (CameraView) findViewById(R.id.java_surface_view);
        //mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
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
            Toast.makeText(this, "Camera resolution changed. Recreate calibrator", Toast.LENGTH_LONG).show();
            mCameraInfo = new CameraInfo();
            mCameraInfo.mCameraIndex = mOpenCvCameraView.getCameraIndex();
            mCameraInfo.mWidth = width;
            mCameraInfo.mHeight = height;
            CameraCalibrationResult calibrationResult = new CameraCalibrationResult(mCameraInfo);
            mCalibrator = new SensorCalibrator(mCameraInfo);
            if (calibrationResult.tryLoad(this)) {
                mCalibrator.setCalibrationResult(calibrationResult);
                Toast.makeText(this, "Calibration data loaded from previous launch", Toast.LENGTH_LONG).show();
            }
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
                if (mCalibrator == null) {
                    return true;
                }

                final Resources res = getResources();
                if (mCalibrator.getCornersBufferSize() < 2) {
                    Toast.makeText(this, res.getString(R.string.more_samples), Toast.LENGTH_SHORT).show();
                    return true;
                }

                new AsyncTask<Void, Void, Void>() {
                    private ProgressDialog calibrationProgress;

                    @Override
                    protected void onPreExecute() {
                        mOpenCvCameraView.disableView();
                        calibrationProgress = new ProgressDialog(CalibrationActivity.this);
                        calibrationProgress.setTitle(res.getString(R.string.calibrating));
                        calibrationProgress.setMessage(res.getString(R.string.please_wait));
                        calibrationProgress.setCancelable(false);
                        calibrationProgress.setIndeterminate(true);
                        calibrationProgress.show();
                    }

                    @Override
                    protected Void doInBackground(Void... arg0) {
                        mCalibrator.calibrate();
                        return null;
                    }

                    @Override
                    protected void onPostExecute(Void result) {
                        calibrationProgress.dismiss();
                        mCalibrator.reset();
                        String resultMessage = (mCalibrator.isCalibrated()) ?
                                res.getString(R.string.calibration_successful) :
                                res.getString(R.string.calibration_unsuccessful);
                        (Toast.makeText(CalibrationActivity.this, resultMessage, Toast.LENGTH_SHORT)).show();

                        if (mCalibrator.isCalibrated()) {
                            CameraCalibrationResult calibrationResult = mCalibrator.getCalibrationResult();
                            calibrationResult.save(calibrationResult.getSharedPreferences(CalibrationActivity.this));
                            if (CALIBRATE_ACTION.equals(CalibrationActivity.this.getIntent().getAction())) {
                                Log.e(TAG, "Return received calibration result");
                                Intent data = new Intent();
                                data.putExtra("result", calibrationResult.getJSON().toString());
                                setResult(RESULT_OK, data);
                                finish();
                            }
                        }
                        mOpenCvCameraView.enableView();
                    }
                }.execute();
                return true;
            }
        }
        return false;
    }

    @Override
    public boolean onTouch(View v, MotionEvent event) {
        Log.d(TAG, "onTouch invoked");
        if (mCalibrator != null) {
            mCalibrator.addCorners();
        }
        return false;
    }
}
