package org.opencv.android.services.calibration.sensor;

import org.opencv.android.services.Utils;
import org.opencv.android.services.calibration.CalibrationBoard;
import org.opencv.android.services.calibration.CameraCalibrationResult;
import org.opencv.android.services.calibration.CameraInfo;
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

import android.hardware.SensorManager;
import android.util.Log;

public class SensorCalibrator {
    private static final String TAG = "SensorCalibrator";

    protected CameraInfo mCameraInfo;
    protected SensorRecorder mSensorRecorder;
    protected CameraCalibrationResult mCalibrationResult;

    protected CalibrationBoard mBoard = new CalibrationBoard();

    private boolean mPatternWasFound = false;
    private MatOfPoint2f mCorners = new MatOfPoint2f();

    private Mat mCameraMatrix = new Mat();
    private Mat mDistortionCoefficients = new Mat();

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

        if (mPatternWasFound && isCalibrated()) {
            MatOfPoint2f points = new MatOfPoint2f(mCorners);
            MatOfDouble distortionCoefficients = new MatOfDouble(mDistortionCoefficients);
            Mat rvec = new Mat(), tvec = new Mat();
            Mat boardPoints = Mat.zeros(mBoard.mCornersSize, 1, CvType.CV_32FC3);
            mBoard.calcBoardCornerPositions(boardPoints);
            MatOfPoint3f boardPoints3f = new MatOfPoint3f(boardPoints);
            Calib3d.solvePnPRansac(boardPoints3f, points, mCameraMatrix, distortionCoefficients, rvec, tvec);

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

            Log.e(TAG, Rfloat.dump().replace('\n', ' '));
            Log.e(TAG, "tvec=" + tvec.dump().replace('\n', ' '));

            float az = (float)Math.atan2(-Rvalues[2], Rvalues[8]);
            float pitch = (float)Math.asin(Rvalues[5]);
            float roll = (float)Math.atan2(Rvalues[3], Rvalues[4]);
            Log.i(TAG, "o: az=" + az + " p=" + Utils.toDeg(pitch) + " r=" + Utils.toDeg(roll));
            Core.putText(rgbaFrame, "Camera", new Point(350, 100), Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255, 255, 255), 2);
            Core.putText(rgbaFrame, "Diff", new Point(500, 100), Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255, 0, 0), 2);
            Core.putText(rgbaFrame, String.format("%.1f", Utils.toDeg(az)), new Point(350, 150), Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255, 255, 255), 2);
            Core.putText(rgbaFrame, String.format("%.1f", Utils.toDeg(pitch)), new Point(350, 200), Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255, 255, 255), 2);
            Core.putText(rgbaFrame, String.format("%.1f", Utils.toDeg(roll)), new Point(350, 250), Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255, 255, 255), 2);
            // don't show useless azimut diff
            Core.putText(rgbaFrame, String.format("%.3f", Utils.toDeg(pitch - sensor_pitch)), new Point(500, 200), Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255, 0, 0), 2);
            Core.putText(rgbaFrame, String.format("%.3f", Utils.toDeg(roll - sensor_roll)), new Point(500, 250), Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255, 0, 0), 2);
        }
    }

    public boolean isCalibrated() {
        return mCalibrationResult != null;
    }

    public void setCalibrationResult(CameraCalibrationResult calibrationResult) {
        mCalibrationResult = calibrationResult;
        mCalibrationResult.getMatInvertFy(mCameraMatrix, mDistortionCoefficients);
    }

    public CameraCalibrationResult getCalibrationResult() {
        return mCalibrationResult;
    }
}
