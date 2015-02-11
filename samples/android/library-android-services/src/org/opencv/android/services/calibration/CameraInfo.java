package org.opencv.android.services.calibration;

import org.opencv.android.CameraBridgeViewBase;

import android.os.Bundle;

public class CameraInfo {
    public static final int CAMERA_ID_ANY   = CameraBridgeViewBase.CAMERA_ID_ANY;
    public static final int CAMERA_ID_BACK  = CameraBridgeViewBase.CAMERA_ID_BACK;
    public static final int CAMERA_ID_FRONT = CameraBridgeViewBase.CAMERA_ID_FRONT;

    public int mCameraIndex;
    public int mWidth;
    public int mHeight;

    public CameraInfo() {
        mCameraIndex = CAMERA_ID_ANY;
        mWidth = -1;
        mHeight = -1;
    }

    public static final String BUNDLE_PARAMETER_CAMERA_ID = "camera_index";
    public static final String BUNDLE_PARAMETER_CAMERA_WIDTH = "camera_width";
    public static final String BUNDLE_PARAMETER_CAMERA_HEIGHT = "camera_height";

    public void readFromBundle(Bundle bundle) {
        mCameraIndex = bundle.getInt(BUNDLE_PARAMETER_CAMERA_ID, CAMERA_ID_ANY);
        mWidth = bundle.getInt(BUNDLE_PARAMETER_CAMERA_WIDTH, -1);
        mHeight = bundle.getInt(BUNDLE_PARAMETER_CAMERA_HEIGHT, -1);
        if (mWidth <= 0 || mHeight <= 0) {
            throw new AssertionError("Invalid camera size");
        }
    }

    public void saveToBundle(Bundle bundle) {
        bundle.putInt(BUNDLE_PARAMETER_CAMERA_ID, mCameraIndex);
        bundle.putInt(BUNDLE_PARAMETER_CAMERA_WIDTH, mWidth);
        bundle.putInt(BUNDLE_PARAMETER_CAMERA_HEIGHT, mHeight);
    }

    @Override
    public String toString() {
        return String.format("camera_%d:%dx%d", mCameraIndex, mWidth, mHeight);
    }
}
