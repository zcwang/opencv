package org.opencv.android.services;


public class Utils {

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
