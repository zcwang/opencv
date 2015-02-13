package org.opencv.android.services;

import java.io.File;

import android.os.Environment;


public class Utils {

    public static String getStorageBase() {
        String storageBase;
        File extSD = new File("/storage/removable/sdcard1/");
        if (extSD.exists()) {
            storageBase = extSD.getAbsolutePath();
        } else {
            extSD = new File("/storage/extSdCard");
            if (extSD.exists())
                storageBase = extSD.getAbsolutePath();
            else
                storageBase = Environment.getExternalStorageDirectory().getAbsolutePath();
        }
        return storageBase;
    }

    public static float toDeg(float rad) { return (float)(rad * 180.0 / Math.PI); }

    public static String toStringArray(float[] v) {
        StringBuffer sb = new StringBuffer();
        sb.append("[");
        for (int i = 0; i < v.length; i++) {
            if (i != 0)
              sb.append(" ");
            sb.append(String.format("%.2f", v[i]));
        }
        sb.append("]");
        return sb.toString();
    }
}
