package org.opencv.android.services.calibration;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.opencv.android.services.Utils;
import org.opencv.core.Mat;

import android.app.AlertDialog;
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
    private static final String TAG = "CameraCalibrationResult";

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
    public void getMatInvertFy(Mat cameraMatrix, Mat distortionCoefficients) {
        double[] cameraMatrixArray = mCameraMatrixArray.clone();
        cameraMatrixArray[4] = -Math.abs(cameraMatrixArray[4]);
        cameraMatrix.put(0, 0, cameraMatrixArray);
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
            Log.e(TAG, json.toString());
            e.printStackTrace();
            throw new AssertionError("initFromJSON");
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

    public void save(Context context) {
        SharedPreferences sharedPref = getSharedPreferences(context);
        save(sharedPref);
    }

    public boolean tryLoad(SharedPreferences sharedPref) {
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

    public boolean tryLoad(Context context) {
        SharedPreferences sharedPref = getSharedPreferences(context);
        return tryLoad(sharedPref);
    }

    @SuppressWarnings("deprecation")
    public SharedPreferences getSharedPreferences(Context context) {
        return context.getSharedPreferences("calibration_" + mCameraInfo.toString(), Context.MODE_WORLD_READABLE);
    }

    @SuppressWarnings("deprecation")
    public SharedPreferences getExternalSharedPreferences(Context context) {
        try {
            Context c = context.createPackageContext("org.opencv.android.services.calibration.camera", 0);
            return c.getSharedPreferences("calibration_" + mCameraInfo.toString(), Context.MODE_WORLD_READABLE);
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
//        try {
//            SharedPreferences sp = getExternalSharedPreferences(context);
//            if (sp != null) {
//                if (tryLoad(sp)) {
//                    Toast.makeText(context, "Camera calibration data loaded from shared preferences", Toast.LENGTH_SHORT).show();
//                    if (callbacks != null) {
//                        callbacks.onSuccess();
//                    }
//                    return;
//                }
//            }
//        } catch (Exception e) {
//            e.printStackTrace();
//        }
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
                    Bundle extras = intent.getExtras();
                    if (extras != null) {
                        try {
                            String response = extras.getString("response");
                            CameraCalibrationResult.this.initFromJSON(new JSONObject(response));
                            success = true;
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
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
                try {
                    context.startActivity(intent);
                    Log.d(TAG, "call calibration intent");
                } catch (android.content.ActivityNotFoundException e) {
                    AlertDialog ad = new AlertDialog.Builder(context).create();
                    ad.setCancelable(false); // This blocks the 'BACK' button
                    ad.setMessage("It seems that camera calibration service is not installed. Install it for proper application work");
                    ad.show();
                    if (callbacks != null)
                        callbacks.onFailure();
                }
            }
        }.run();
    }

    public void saveToStorage(Context notificationContext) {
        String storageBase = Utils.getStorageBase();
        try {
            String dstDir = storageBase + "/itseez/camera_calibration";
            if (!new File(dstDir).exists() && !new File(dstDir).mkdirs())
            {
                String msg = "Can't create storage dir: " + dstDir;
                Log.e(TAG, msg);
                if (notificationContext != null)
                    Toast.makeText(notificationContext, msg, Toast.LENGTH_LONG).show();
                return;
            }
            {
                String dstPath = dstDir + "/camera.intrinsics."+mCameraInfo.toString()+".txt";
                OutputStream os = new FileOutputStream(dstPath);
                String record = "";
                for (int i = 0; i < mCameraMatrixArray.length; i++)
                    record = record + (i==0?"":" ") + mCameraMatrixArray[i];
                record = record + "\n";
                for (int i = 0; i < mDistortionCoefficientsArray.length; i++)
                    record = record + (i==0?"":" ") + mDistortionCoefficientsArray[i];
                record = record + "\n";
                os.write(record.toString().getBytes());
                os.close();
                String msg = "Calibration data " + mCameraInfo.mWidth + "x" + mCameraInfo.mHeight + " saved:\n" + dstPath;
                Log.i(TAG, msg);
                if (notificationContext != null)
                    Toast.makeText(notificationContext, msg, Toast.LENGTH_LONG).show();
            }
        } catch (Exception e) {
            if (notificationContext != null)
                Toast.makeText(notificationContext, "Error", Toast.LENGTH_LONG).show();
            e.printStackTrace();
        }

    }
}
