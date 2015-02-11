package org.opencv.android.services;

import java.io.FileOutputStream;
import java.util.List;

import org.opencv.android.JavaCameraView;

import android.content.Context;
import android.hardware.Camera;
import android.hardware.Camera.PictureCallback;
import android.hardware.Camera.Size;
import android.util.AttributeSet;
import android.util.Log;

public class CameraView extends JavaCameraView implements PictureCallback {

    private static final String TAG = "CameraView";
    private String mPictureFileName;

    public CameraView(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    public List<String> getEffectList() {
        return mCamera.getParameters().getSupportedColorEffects();
    }

    public boolean isEffectSupported() {
        return (mCamera.getParameters().getColorEffect() != null);
    }

    public String getEffect() {
        return mCamera.getParameters().getColorEffect();
    }

    public void setEffect(String effect) {
        Camera.Parameters params = mCamera.getParameters();
        params.setColorEffect(effect);
        mCamera.setParameters(params);
    }

    public List<Size> getResolutionList() {
        return mCamera.getParameters().getSupportedPreviewSizes();
    }

    public void setResolution(int width, int height) {
        if (mCamera != null)
        {
            disconnectCamera();
            mMaxHeight = height;
            mMaxWidth = width;
            connectCamera(getWidth(), getHeight());
        } else {
            mMaxHeight = height;
            mMaxWidth = width;
        }
    }

    public Size getResolution() {
        return mCamera.getParameters().getPreviewSize();
    }

    public int getCameraIndex() {
        return mCameraIndex;
    }

    public void takePicture(final String fileName) {
        Log.i(TAG, "Taking picture");
        this.mPictureFileName = fileName;

        mCamera.setPreviewCallback(null);
        mCamera.takePicture(null, null, this);
    }

    @Override
    public void onPictureTaken(byte[] data, Camera camera) {
        Log.i(TAG, "Saving a bitmap to file");

        mCamera.startPreview();
        mCamera.setPreviewCallback(this);

        // Write the image in a file (in jpeg format)
        try {
            FileOutputStream fos = new FileOutputStream(mPictureFileName);

            fos.write(data);
            fos.close();

        } catch (java.io.IOException e) {
            Log.e(TAG, "Exception in photoCallback", e);
        }

    }

    @Override
    protected org.opencv.core.Size calculateCameraFrameSize(List<?> supportedSizes, ListItemAccessor accessor, int surfaceWidth, int surfaceHeight) {
        if (mMaxWidth > 0) {
            // force to use requested size
            return new org.opencv.core.Size(mMaxWidth, mMaxHeight);
        } else {
            return super.calculateCameraFrameSize(supportedSizes, accessor, surfaceWidth, surfaceHeight);
        }
    }
}
