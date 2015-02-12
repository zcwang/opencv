package org.opencv.android.services.calibration.sensor;

import java.util.ArrayList;
import java.util.List;

import org.opencv.android.services.calibration.CalibrationBoard;
import org.opencv.android.services.calibration.CameraCalibrationResult;
import org.opencv.android.services.calibration.CameraInfo;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;

import android.util.Log;

public class SensorCalibrator {
    private static final String TAG = "CameraCalibrator";

    protected CameraInfo mCameraInfo;
    protected CameraCalibrationResult mCalibrationResult;

    protected CalibrationBoard mBoard = new CalibrationBoard();

    private boolean mPatternWasFound = false;
    private MatOfPoint2f mCorners = new MatOfPoint2f();
    private List<Mat> mCornersBuffer = new ArrayList<Mat>();

    private Mat mCameraMatrix = new Mat();
    private Mat mDistortionCoefficients = new Mat();
    private int mFlags = Calib3d.CALIB_FIX_PRINCIPAL_POINT +
                         Calib3d.CALIB_ZERO_TANGENT_DIST +
                         Calib3d.CALIB_FIX_ASPECT_RATIO +
                         Calib3d.CALIB_FIX_K4 +
                         Calib3d.CALIB_FIX_K5;
    private Double mRms;

    public SensorCalibrator(CameraInfo cameraInfo) {
        mCameraInfo = cameraInfo;
        Mat.eye(3, 3, CvType.CV_64FC1).copyTo(mCameraMatrix);
        Mat.zeros(5, 1, CvType.CV_64FC1).copyTo(mDistortionCoefficients);
    }

    public void processFrame(Mat grayFrame, Mat rgbaFrame) {
        findPattern(grayFrame);
        renderFrame(rgbaFrame);
    }

    public CameraCalibrationResult calibrate() {
        ArrayList<Mat> rvecs = new ArrayList<Mat>();
        ArrayList<Mat> tvecs = new ArrayList<Mat>();
        Mat reprojectionErrors = new Mat();
        ArrayList<Mat> objectPoints = new ArrayList<Mat>();
        objectPoints.add(Mat.zeros(mBoard.mCornersSize, 1, CvType.CV_32FC3));
        mBoard.calcBoardCornerPositions(objectPoints.get(0));
        for (int i = 1; i < mCornersBuffer.size(); i++) {
            objectPoints.add(objectPoints.get(0));
        }

        Calib3d.calibrateCamera(objectPoints, mCornersBuffer,
                new Size(mCameraInfo.mWidth, mCameraInfo.mHeight),
                mCameraMatrix, mDistortionCoefficients, rvecs, tvecs, mFlags);

        boolean isCalibrated = Core.checkRange(mCameraMatrix)
                && Core.checkRange(mDistortionCoefficients);

        if (isCalibrated) {
            mCalibrationResult = new CameraCalibrationResult(mCameraInfo);
            mCalibrationResult.init(mCameraMatrix, mDistortionCoefficients);

            mRms = computeReprojectionErrors(objectPoints, rvecs, tvecs, reprojectionErrors);
            Log.i(TAG, String.format("Average re-projection error: %f", mRms));
            Log.i(TAG, "Camera matrix: " + mCameraMatrix.dump());
            Log.i(TAG, "Distortion coefficients: " + mDistortionCoefficients.dump());
        }

        return mCalibrationResult;
    }

    public void reset() {
        mCornersBuffer.clear();
    }

    private double computeReprojectionErrors(List<Mat> objectPoints,
            List<Mat> rvecs, List<Mat> tvecs, Mat perViewErrors) {
        MatOfPoint2f cornersProjected = new MatOfPoint2f();
        double totalError = 0;
        double error;
        float viewErrors[] = new float[objectPoints.size()];

        MatOfDouble distortionCoefficients = new MatOfDouble(mDistortionCoefficients);
        int totalPoints = 0;
        for (int i = 0; i < objectPoints.size(); i++) {
            MatOfPoint3f points = new MatOfPoint3f(objectPoints.get(i));
            Calib3d.projectPoints(points, rvecs.get(i), tvecs.get(i),
                    mCameraMatrix, distortionCoefficients, cornersProjected);
            error = Core.norm(mCornersBuffer.get(i), cornersProjected, Core.NORM_L2);

            int n = objectPoints.get(i).rows();
            viewErrors[i] = (float) Math.sqrt(error * error / n);
            totalError  += error * error;
            totalPoints += n;
        }
        perViewErrors.create(objectPoints.size(), 1, CvType.CV_32FC1);
        perViewErrors.put(0, 0, viewErrors);

        return Math.sqrt(totalError / totalPoints);
    }

    private void findPattern(Mat grayFrame) {
        mPatternWasFound = Calib3d.findCirclesGridDefault(grayFrame, mBoard.mPatternSize,
                mCorners, mBoard.mGridFlags);
    }

    public void addCorners() {
        if (mPatternWasFound) {
            Log.i(TAG, "Add corners: " + mCorners.dump());
            mCornersBuffer.add(mCorners.clone());
            mPatternWasFound = false;
        }
    }

    public int getCornersBufferSize() {
        return mCornersBuffer.size();
    }

    private void drawPoints(Mat rgbaFrame) {
        Calib3d.drawChessboardCorners(rgbaFrame, mBoard.mPatternSize, mCorners, mPatternWasFound);
    }

    private static String toDeg(float rad) { return Float.valueOf((float)(rad * 180 / Math.PI)).toString(); }

    private void renderFrame(Mat rgbaFrame) {
        drawPoints(rgbaFrame);

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
            double error = Core.norm(mCorners, cornersProjected, Core.NORM_L2);
            Log.i(TAG, "Frame error: " + error + " rvec=" + rvec.dump() + " tvec=" + tvec.dump());
            Core.putText(rgbaFrame, "Current frame error: " + error, new Point(50, 100),
                    Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(0, 0, 255), 2);

            Mat R = new Mat();
            Calib3d.Rodrigues(rvec, R);
            Mat Rfloat = new Mat(3, 3, CvType.CV_32FC1);
            R.convertTo(Rfloat, CvType.CV_32FC1);
            float[] Rvalues = new float[9];
            Rfloat.get(0, 0, Rvalues);

            float pitch = (float)Math.atan2(Rvalues[1], Rvalues[4]);
            float roll = (float)Math.asin(-Rvalues[5]);
            Core.putText(rgbaFrame,
                    "pitch=" + toDeg(pitch) + " roll=" + toDeg(roll), new Point(50, 200),
                    Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255, 255, 255), 2);
        }

        Core.putText(rgbaFrame, "Sensor calibration", new Point(50, 50),
                Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255, 255, 0), 2);
        if (mRms != null) {
            Core.putText(rgbaFrame, "Last calibration RMS: " + mRms, new Point(50, 150),
                Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255, 255, 0), 2);
        }
    }

    public boolean isCalibrated() {
        return mCalibrationResult != null;
    }

    public void setCalibrationResult(CameraCalibrationResult calibrationResult) {
        mCalibrationResult = calibrationResult;
        mCalibrationResult.getMat(mCameraMatrix, mDistortionCoefficients);
    }

    public CameraCalibrationResult getCalibrationResult() {
        return mCalibrationResult;
    }

}
