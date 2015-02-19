package org.opencv.android.services.sensor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;

import org.json.JSONException;
import org.json.JSONObject;
import org.opencv.android.services.Utils;
import org.opencv.android.services.calibration.CameraInfo;

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

public class SensorCalibrationResult {
    private static final String TAG = "SensorCalibrationResult";

    public CameraInfo mCameraInfo;
    public double mPitchDiff = 0.0;
    public double mRollDiff = 0.0;

    public SensorCalibrationResult(CameraInfo cameraInfo) {
        mCameraInfo = cameraInfo;
    }

    public void init(double pitchDiff, double rollDiff) {
        mPitchDiff = pitchDiff;
        mRollDiff = rollDiff;
    }

    public JSONObject getJSON() {
        try {
            JSONObject json = new JSONObject();
            json.put("pitchDiff", mPitchDiff);
            json.put("rollDiff", mRollDiff);
            return json;
        } catch (JSONException e) {
            e.printStackTrace();
            throw new AssertionError("getJSON");
        }
    }

    public void initFromJSON(JSONObject json) throws Exception {
        try {
            mPitchDiff = json.getDouble("pitchDiff");
            mRollDiff = json.getDouble("rollDiff");
        } catch (JSONException e) {
            Log.e(TAG, json.toString());
            e.printStackTrace();
            throw new Exception("initFromJSON");
        }
    }

    public void save(SharedPreferences sharedPref) {
        SharedPreferences.Editor editor = sharedPref.edit();

        editor.putString("calibration", getJSON().toString());

        editor.commit();

        Log.i(TAG, "Saved sensor calibration data");
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
            Log.i(TAG, "Loaded sensor calibration data");
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
            Context c = context.createPackageContext("org.opencv.android.services.calibration.sensor", Context.CONTEXT_IGNORE_SECURITY|Context.CONTEXT_INCLUDE_CODE);
            return c.getSharedPreferences("calibration_" + mCameraInfo.toString(), Context.MODE_WORLD_READABLE);
        } catch (NameNotFoundException e) {
            e.printStackTrace();
            return null;
        }
    }

    public static final String CALIBRATE_ACTION = "org.opencv.android.services.calibration.sensor.calibrate";

    public interface RequestCallback {
        void onSuccess();
        void onFailure();
    }
    public void requestData(final Context context, final boolean force, final RequestCallback callbacks) {
//        try {
//            SharedPreferences sp = getExternalSharedPreferences(context);
//            if (sp != null) {
//                if (tryLoad(sp)) {
//                    Toast.makeText(context, "Sensor calibration data loaded from shared preferences", Toast.LENGTH_SHORT).show();
//                    if (callbacks != null) {
//                        callbacks.onSuccess();
//                    }
//                    return;
//                }
//            }
//        } catch (Exception e) {
//            e.printStackTrace();
//        }
        Toast.makeText(context, "Requesting sensor calibration data", Toast.LENGTH_SHORT).show();
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
                            SensorCalibrationResult.this.initFromJSON(new JSONObject(response));
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
                    ad.setMessage("It seems that sensor calibration service is not installed. Install it for proper application work");
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
                {
                    String dstPath = dstDir + "/calibration_pitch." + mCameraInfo.toString() + ".txt";
                    OutputStream os = new FileOutputStream(dstPath);
                    String record = "" + mPitchDiff;
                    os.write(record.getBytes());
                    os.close();
                }
                {
                    String dstPath = dstDir + "/calibration_roll." + mCameraInfo.toString() + ".txt";
                    OutputStream os = new FileOutputStream(dstPath);
                    String record = "" + mRollDiff;
                    os.write(record.getBytes());
                    os.close();
                }
                String msg = "Sensor calibration data " + mCameraInfo.mWidth + "x" + mCameraInfo.mHeight + " saved:\n" + dstDir;
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
