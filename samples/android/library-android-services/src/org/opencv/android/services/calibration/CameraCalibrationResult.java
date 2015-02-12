package org.opencv.android.services.calibration;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.opencv.core.Mat;

import android.app.Activity;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.SharedPreferences;
import android.content.pm.PackageManager.NameNotFoundException;
import android.os.Bundle;
import android.util.Log;
import android.widget.Toast;

public class CameraCalibrationResult {
    private static final String TAG = "CalibrationResult";

    private static final int CAMERA_MATRIX_ROWS = 3;
    private static final int CAMERA_MATRIX_COLS = 3;
    private static final int DISTORTION_COEFFICIENTS_SIZE = 5;

    public CameraInfo mCameraInfo;
    public final double[] mCameraMatrixArray = new double[CAMERA_MATRIX_ROWS*CAMERA_MATRIX_COLS];
    public final double[] mDistortionCoefficientsArray = new double[DISTORTION_COEFFICIENTS_SIZE];

    public CameraCalibrationResult(CameraInfo cameraInfo) {
        mCameraInfo = cameraInfo;
    }

    public void init(Mat cameraMatrix, Mat distortionCoefficients) {
        cameraMatrix.get(0,  0, mCameraMatrixArray);
        distortionCoefficients.get(0, 0, mDistortionCoefficientsArray);
    }
    public void getMat(Mat cameraMatrix, Mat distortionCoefficients) {
        cameraMatrix.put(0, 0, mCameraMatrixArray);
        distortionCoefficients.put(0, 0, mDistortionCoefficientsArray);
    }

    public JSONObject getJSON() {
        try {
            JSONObject json = new JSONObject();
            JSONArray jsonCameraMatrix = new JSONArray();
            for (int i = 0; i < mCameraMatrixArray.length; i++)
                jsonCameraMatrix.put(mCameraMatrixArray[i]);
            JSONArray jsonDistortionCoefficients = new JSONArray();
            for (int i = 0; i < mDistortionCoefficientsArray.length; i++)
                jsonDistortionCoefficients.put(mDistortionCoefficientsArray[i]);
            json.put("cameraMatrix", jsonCameraMatrix);
            json.put("distortionCoefficients", jsonDistortionCoefficients);
            return json;
        } catch (JSONException e) {
            e.printStackTrace();
            throw new AssertionError("getJSON");
        }
    }

    public void initFromJSON(JSONObject json) {
        try {
            JSONArray jsonCameraMatrix = json.getJSONArray("cameraMatrix");
            if (jsonCameraMatrix.length() != mCameraMatrixArray.length)
                throw new AssertionError("Invalid camera matrix");
            for (int i = 0; i < CAMERA_MATRIX_ROWS*CAMERA_MATRIX_COLS; i++) {
                mCameraMatrixArray[i] = jsonCameraMatrix.getDouble(i);
            }
            JSONArray jsonDistortionCoefficients = json.getJSONArray("distortionCoefficients");
            if (jsonDistortionCoefficients.length() != mDistortionCoefficientsArray.length)
                throw new AssertionError("Invalid distortion coefficients matrix");
            for (int i = 0; i < DISTORTION_COEFFICIENTS_SIZE; i++) {
                mDistortionCoefficientsArray[i] = jsonDistortionCoefficients.getDouble(i);
            }
        } catch (JSONException e) {
            e.printStackTrace();
            throw new AssertionError("getJSON");
        }
    }

    public void save(SharedPreferences sharedPref) {
        SharedPreferences.Editor editor = sharedPref.edit();

        editor.putInt("cameraIndex", mCameraInfo.mCameraIndex);
        editor.putInt("width", mCameraInfo.mWidth);
        editor.putInt("height", mCameraInfo.mHeight);
        editor.putString("calibration", getJSON().toString());

        editor.commit();

        Log.i(TAG, "Saved calibration data: " + String.format("%dx%d", mCameraInfo.mWidth, mCameraInfo.mHeight));
    }

    public boolean tryLoad(Context context) {
        SharedPreferences sharedPref = getSharedPreferences(context);
        if (!sharedPref.contains("calibration")) {
            Log.i(TAG, "No previous calibration results found");
            return false;
        }

        try {
            initFromJSON(new JSONObject(sharedPref.getString("calibration", "")));
            Log.i(TAG, "Loaded calibration data: " + String.format("%dx%d", mCameraInfo.mWidth, mCameraInfo.mHeight));
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }


    @SuppressWarnings("deprecation")
    public SharedPreferences getSharedPreferences(Context context) {
        return context.getSharedPreferences("calibration_" + mCameraInfo.toString(), Context.MODE_WORLD_READABLE);
    }

    @SuppressWarnings("deprecation")
    public SharedPreferences getExternalSharedPreferences(Activity activity) {
        try {
            Context context = activity.createPackageContext("org.opencv.android.services.calibration.camera", Context.MODE_WORLD_READABLE);
            return context.getSharedPreferences("calibration_" + mCameraInfo.toString(), Context.MODE_WORLD_READABLE);
        } catch (NameNotFoundException e) {
            e.printStackTrace();
            return null;
        }
    }

    public static final String CALIBRATE_ACTION = "org.opencv.android.services.calibration.camera.calibrate";

    public interface RequestCallback {
        void onSuccess();
        void onFailure();
    }
    public void requestData(final Context context, final boolean force, final RequestCallback callbacks) {
        Toast.makeText(context, "Requesting camera calibration data", Toast.LENGTH_SHORT).show();
        final String responseAction = CALIBRATE_ACTION + "!" + mCameraInfo.toString();
        new Runnable() {
            private static final String TAG = "CalibrationResult::requestData";
            private final BroadcastReceiver myReceiver = new BroadcastReceiver() {

                @Override
                public void onReceive(Context context, Intent intent) {
                    context.unregisterReceiver(myReceiver);
                    Log.e(TAG, "BroadcastReceiver::onReceive: " + intent.getAction());
                    boolean success = false;
                    try {
                        String response = intent.getExtras().getString("response");
                        CameraCalibrationResult.this.initFromJSON(new JSONObject(response));
                        success = true;
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    Log.e(TAG, "success=" + success);
                    if (callbacks != null) {
                        if (success)
                            callbacks.onSuccess();
                        else
                            callbacks.onFailure();
                    }
                }
            };

            @Override
            public void run() {
                context.registerReceiver(myReceiver, new IntentFilter(responseAction));

                Intent intent = new Intent(CALIBRATE_ACTION);
                Bundle params = new Bundle();
                mCameraInfo.saveToBundle(params);
                intent.putExtras(params);
                intent.putExtra("responseAction", responseAction);
                if (force)
                    intent.putExtra("force", true);
                context.startActivity(intent);
                Log.d(TAG, "call calibration intent");
            }
        }.run();
    }
}
