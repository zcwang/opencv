package org.opencv.android.services.calibration.sensor;

import java.util.ArrayList;

import org.opencv.android.services.Utils;
import org.opencv.android.services.calibration.CalibrationBoard;
import org.opencv.android.services.calibration.CameraCalibrationResult;
import org.opencv.android.services.calibration.CameraInfo;
import org.opencv.android.services.sensor.SensorCalibrationResult;
import org.opencv.android.services.sensor.SensorRecorder;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;

import android.content.Context;
import android.hardware.SensorManager;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;

public class SensorCalibrator {
    private static final String TAG = "SensorCalibrator";

    protected CameraInfo mCameraInfo;
    protected SensorRecorder mSensorRecorder;
    protected CameraCalibrationResult mCameraCalibrationResult;
    protected SensorCalibrationResult mSensorCalibrationResult;

    protected CalibrationBoard mBoard = new CalibrationBoard();

    private boolean mPatternWasFound = false;
    private MatOfPoint2f mCorners = new MatOfPoint2f();

    private Mat mCameraMatrix = new Mat();
    private Mat mDistortionCoefficients = new Mat();

    public interface CompletionCallback {
        void onProgress(int value, int total);
        void onCompleted();
    }
    private boolean mStartCalibration = false;
    private CompletionCallback mCompletionCallback = null;
    private ArrayList<Double> mPitchDifferences = new ArrayList<Double>();
    private ArrayList<Double> mRollDifferences = new ArrayList<Double>();

    private final int mTotalCalibrationSamples = 50;

    public SensorCalibrator(CameraInfo cameraInfo, SensorRecorder sensorRecorder) {
        mCameraInfo = cameraInfo;
        mSensorRecorder = sensorRecorder;
        Mat.eye(3, 3, CvType.CV_64FC1).copyTo(mCameraMatrix);
        Mat.zeros(5, 1, CvType.CV_64FC1).copyTo(mDistortionCoefficients);
    }

    public void processFrame(Mat grayFrame, Mat rgbaFrame) {
        mPatternWasFound = Calib3d.findCirclesGridDefault(grayFrame, mBoard.mPatternSize,
                mCorners, mBoard.mGridFlags);
        renderFrame(rgbaFrame);
    }

    private void renderFrame(Mat rgbaFrame) {
        Core.line(rgbaFrame, new Point(0, rgbaFrame.rows()/2), new Point(rgbaFrame.cols(), rgbaFrame.rows()/2), new Scalar(255,255,255), 1);
        Core.line(rgbaFrame, new Point(rgbaFrame.cols()/2, 0), new Point(rgbaFrame.cols()/2, rgbaFrame.rows()), new Scalar(255,255,255), 1);

        Calib3d.drawChessboardCorners(rgbaFrame, mBoard.mPatternSize, mCorners, mPatternWasFound);

        Core.putText(rgbaFrame, "Sensor calibration", new Point(50, 50),
                Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255, 255, 0), 2);
        float[] rotM = mSensorRecorder.mRotationMatrix.clone();
        SensorManager.remapCoordinateSystem(rotM, SensorManager.AXIS_Z, SensorManager.AXIS_MINUS_X, rotM);
        float[] orientation = new float[3];
        SensorManager.getOrientation(rotM, orientation);
        float sensor_az = orientation[0];
        float sensor_pitch = orientation[1];
        float sensor_roll = orientation[2];
        Core.putText(rgbaFrame, "az=", new Point(50, 150), Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255, 255, 0), 2);
        Core.putText(rgbaFrame, "pitch=", new Point(50, 200), Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255, 255, 0), 2);
        Core.putText(rgbaFrame, "roll=", new Point(50, 250), Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255, 255, 0), 2);
        Core.putText(rgbaFrame, "Sensor", new Point(200, 100), Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(64, 128, 64), 2);
        Core.putText(rgbaFrame, String.format("%.1f", Utils.toDeg(sensor_az)), new Point(200, 150), Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(64, 128, 64), 2);
        Core.putText(rgbaFrame, String.format("%.1f", Utils.toDeg(sensor_pitch)), new Point(200, 200), Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(64, 128, 64), 2);
        Core.putText(rgbaFrame, String.format("%.1f", Utils.toDeg(sensor_roll)), new Point(200, 250), Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(64, 128, 64), 2);

        if (mStartCalibration || mSensorCalibrationResult != null) {
            Core.putText(rgbaFrame, "calibrated pitch", new Point(50, 300), Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(128, 255, 0), 2);
            Core.putText(rgbaFrame, "calibrated roll", new Point(50, 350), Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(128, 255, 0), 2);
            if (mSensorCalibrationResult != null) {
                Core.putText(rgbaFrame, String.format("%.3f", Utils.toDeg((float)mSensorCalibrationResult.mPitchDiff)), new Point(350, 300), Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(128, 255, 0), 2);
                Core.putText(rgbaFrame, String.format("%.3f", Utils.toDeg((float)mSensorCalibrationResult.mRollDiff)), new Point(350, 350), Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(128, 255, 0), 2);
            }
        }

        if (mPatternWasFound && isCameraCalibrated()) {
            MatOfPoint2f points = new MatOfPoint2f(mCorners);
            MatOfDouble distortionCoefficients = new MatOfDouble(mDistortionCoefficients);
            Mat rvec = new Mat(), tvec = new Mat();
            Mat boardPoints = Mat.zeros(mBoard.mCornersSize, 1, CvType.CV_32FC3);
            mBoard.calcBoardCornerPositions(boardPoints);
            MatOfPoint3f boardPoints3f = new MatOfPoint3f(boardPoints);
            Calib3d.solvePnP(boardPoints3f, points, mCameraMatrix, distortionCoefficients, rvec, tvec);

            MatOfPoint2f cornersProjected = new MatOfPoint2f();
            Calib3d.projectPoints(boardPoints3f, rvec, tvec,
                    mCameraMatrix, distortionCoefficients, cornersProjected);
            double error = Core.norm(mCorners, cornersProjected, Core.NORM_L2) / mBoard.mCornersSize;
            Core.putText(rgbaFrame, "Current frame error: " + error, new Point(50, mCameraInfo.mHeight - 100),
                    Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(0, 0, 255), 2);

            Mat R = new Mat();
            Calib3d.Rodrigues(rvec, R);
            Mat Rfloat = new Mat(3, 3, CvType.CV_32FC1);
            R.convertTo(Rfloat, CvType.CV_32FC1);
            float[] Rvalues = new float[9];
            Rfloat.get(0, 0, Rvalues);

            //Log.e(TAG, Rfloat.dump().replace('\n', ' '));
            Log.d(TAG, "tvec=" + tvec.dump().replace('\n', ' '));

            // ZXY
            double az = Math.atan2(Rvalues[6], Rvalues[8]);
            double pitch = Math.asin(-Rvalues[7]);
            double roll = Math.atan2(-Rvalues[1], Rvalues[4]);

            Log.i(TAG, "o: az=" + az + " p=" + Utils.toDeg((float)pitch) + " r=" + Utils.toDeg((float)roll));
            Core.putText(rgbaFrame, "Camera", new Point(350, 100), Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255, 255, 255), 2);
            Core.putText(rgbaFrame, "Diff", new Point(500, 100), Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255, 0, 0), 2);
            Core.putText(rgbaFrame, String.format("%.1f", Utils.toDeg((float)az)), new Point(350, 150), Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255, 255, 255), 2);
            Core.putText(rgbaFrame, String.format("%.1f", Utils.toDeg((float)pitch)), new Point(350, 200), Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255, 255, 255), 2);
            Core.putText(rgbaFrame, String.format("%.1f", Utils.toDeg((float)roll)), new Point(350, 250), Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255, 255, 255), 2);
            double pitch_diff = pitch - sensor_pitch;
            double roll_diff = roll - sensor_roll;
            // don't show useless azimut diff
            Core.putText(rgbaFrame, String.format("%.3f", Utils.toDeg((float)pitch_diff)), new Point(500, 200), Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255, 0, 0), 2);
            Core.putText(rgbaFrame, String.format("%.3f", Utils.toDeg((float)roll_diff)), new Point(500, 250), Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255, 0, 0), 2);

            if (mStartCalibration) {
                mPitchDifferences.add(pitch_diff);
                mRollDifferences.add(roll_diff);
                Double calibrated_pitch = Utils.medianFilter(mPitchDifferences);
                Double calibrated_roll = Utils.medianFilter(mRollDifferences);
                Core.putText(rgbaFrame, String.format("(%.3f)", Utils.toDeg(calibrated_pitch.floatValue())), new Point(500, 300), Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(192, 0, 0), 2);
                Core.putText(rgbaFrame, String.format("(%.3f)", Utils.toDeg(calibrated_roll.floatValue())), new Point(500, 350), Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(192, 0, 0), 2);
                if (mCompletionCallback != null)
                    mCompletionCallback.onProgress(mPitchDifferences.size(), mTotalCalibrationSamples);
                if (mPitchDifferences.size() == mTotalCalibrationSamples) {
                    mSensorCalibrationResult = new SensorCalibrationResult(mCameraInfo);
                    mSensorCalibrationResult.mPitchDiff = calibrated_pitch.doubleValue();
                    mSensorCalibrationResult.mRollDiff = calibrated_roll.doubleValue();
                    mCompletionCallback.onCompleted();
                    mStartCalibration = false;
                }
            }
        }
    }

    public boolean isCameraCalibrated() {
        return mCameraCalibrationResult != null;
    }

    public void setCameraCalibrationResult(CameraCalibrationResult calibrationResult) {
        mCameraCalibrationResult = calibrationResult;
        mCameraCalibrationResult.getMatInvertFy(mCameraMatrix, mDistortionCoefficients);
    }

    public CameraCalibrationResult getCameraCalibrationResult() {
        return mCameraCalibrationResult;
    }

    public SensorCalibrationResult getSensorCalibrationResult() {
        return mSensorCalibrationResult;
    }

    public void setSensorCalibrationResult(SensorCalibrationResult calibrationResult) {
        mSensorCalibrationResult = calibrationResult;
        Log.e(TAG, "Pitch diff: " + mSensorCalibrationResult.mPitchDiff);
        Log.e(TAG, "Roll diff: " + mSensorCalibrationResult.mRollDiff);
    }

    public void start(CompletionCallback completionCallback) {
        mCompletionCallback = completionCallback;
        mStartCalibration = true;
        mPitchDifferences.clear();
        mRollDifferences.clear();
        if (mCompletionCallback != null)
            mCompletionCallback.onProgress(0, mTotalCalibrationSamples);
    }

    public void stop() {
        mCompletionCallback = null;
        mStartCalibration = false;
        mPitchDifferences.clear();
        mRollDifferences.clear();
    }


    public void onCreateMenu(Menu menu, int menuGroupId) {
        mBoard.onCreateMenu(menu, menuGroupId);
    }
    public boolean onMenuItemSelected(final Context context, MenuItem item) {
        return mBoard.onMenuItemSelected(context, item);
    }
}
