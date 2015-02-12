package org.opencv.android.services.sensor;

import java.io.FileOutputStream;
import java.io.OutputStream;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.util.Log;

public class SensorRecorder implements SensorEventListener {
    private SensorManager mSensorManager;
    private Sensor mRotationVectorSensor;
    private Sensor mGravitySensor;

    public final float[] mRotationMatrix = new float[16];
    public float[] mOrientation = new float[3];
    public float[] mGravity;

    public long mOrientationTimestamp = 0;
    public long mRotationMatrixTimestamp = 0;
    public long mGravityTimestamp = 0;

    private OutputStream osRotation;
    private OutputStream osGravity;
    private OutputStream osOrientation;

    public long timestampRecordStart = 0;

    public SensorRecorder(Context context) {
        mSensorManager = (SensorManager)context.getSystemService(Context.SENSOR_SERVICE);
        mRotationVectorSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR);
        mGravitySensor = mSensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY);
    }

    public void start() {
        mSensorManager.registerListener(this, mRotationVectorSensor, 10000);
        mSensorManager.registerListener(this, mGravitySensor, 10000);
    }

    public void stop() {
        mSensorManager.unregisterListener(this, mRotationVectorSensor);
        mSensorManager.unregisterListener(this, mGravitySensor);
    }

    public void prepare(String fileNameBase)
    {
        osRotation = null;
        try {
            if (fileNameBase != null) {
                osRotation = new FileOutputStream(fileNameBase + ".rotation.txt");
                osGravity = new FileOutputStream(fileNameBase + ".gravity.txt");
                osOrientation = new FileOutputStream(fileNameBase + ".orientation.txt");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void startRecord() {
        timestampRecordStart = System.nanoTime();
    }

    public void stopRecord()
    {
        try {
            if (osRotation != null)
                osRotation.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
        try {
            if (osGravity != null)
                osGravity.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
        try {
            if (osOrientation != null)
                osOrientation.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
        osRotation = null;
        osGravity = null;
        timestampRecordStart = 0;
    }

    public static float toDeg(float rad)
    {
        return (float)(rad * 180 / Math.PI);
    }

    @Override
    public void onSensorChanged(android.hardware.SensorEvent event) {
        if (event.sensor.getType() == Sensor.TYPE_ROTATION_VECTOR) {
            SensorManager.getRotationMatrixFromVector(mRotationMatrix, event.values);
            SensorManager.getOrientation(mRotationMatrix, mOrientation);
            mRotationMatrixTimestamp = event.timestamp;
            mOrientationTimestamp = event.timestamp;
            //Log.d("Sensor", "Rotation +" + String.format("%.3f", ts * 1e-9) + "s [" + event.values[0] + ", " + event.values[1] + ", " + event.values[2] + "]");
            //Log.d("Sensor", "Orientation [" + toDeg(mOrientation[0]) + ", " + toDeg(mOrientation[1]) + ", " + toDeg(mOrientation[2]) + "]");
            if (timestampRecordStart == 0)
                return;
            long ts = event.timestamp - timestampRecordStart;
            if (osRotation != null) {
                String record = String.format("%d", ts / 1000000) + " :";
                for (int i = 0; i < mRotationMatrix.length; i++)
                    record = record + " " + mRotationMatrix[i];
                record = record + "\n";
                try {
                    osRotation.write(record.getBytes("UTF-8"));
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
            if (osOrientation != null) {
                String record = String.format("%d", ts / 1000000) + " :";
                for (int i = 0; i < mOrientation.length; i++)
                    record = record + " " + mOrientation[i];
                record = record + "\n";
                try {
                    osOrientation.write(record.getBytes("UTF-8"));
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
        else if (event.sensor.getType() == Sensor.TYPE_GRAVITY) {
            mGravity = event.values.clone();
            mGravityTimestamp = event.timestamp;
            //Log.d("Sensor", "Gravity [" + event.values[0] + ", " + event.values[1] + ", " + event.values[2] + "]");
            if (timestampRecordStart == 0)
                return;
            long ts = event.timestamp - timestampRecordStart;
            if (osGravity != null) {
                String record = String.format("%d", ts / 1000000) + " :";
                for (int i = 0; i < mGravity.length; i++)
                    record = record + " " + mGravity[i];
                record = record + "\n";
                try {
                    osGravity.write(record.getBytes("UTF-8"));
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    }
    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        Log.d("Sensor", "onAccuracyChanged: " + sensor.getName() + " acc=" + accuracy);
    }
};
