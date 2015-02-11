package org.opencv.android.services.calibration;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;

public class CalibrationBoard {

    public final Size mPatternSize = new Size(4, 11);
    public final int mGridFlags = Calib3d.CALIB_CB_ASYMMETRIC_GRID;
    public final int mCornersSize = (int)(mPatternSize.width * mPatternSize.height);
    protected double mSquareSize = 0.0155;

    public void calcBoardCornerPositions(Mat corners) {
        final int cn = 3;
        float positions[] = new float[mCornersSize * cn];

        // 4 x 7
        // * * * *
        //  * * *
        // * * * *
        //  * * *
        // * * * *
        //  * * *
        // * * * *
        //  * * *
        for (int i = 0; i < mPatternSize.height; i++) {
            for (int j = 0; j < mPatternSize.width; j++) {
                positions[(int) (i * mPatternSize.width * cn + j * cn + 0)] =
                        - // OpenCV findCorners issue
                        i * (float) mSquareSize;
                positions[(int) (i * mPatternSize.width * cn + j * cn + 1)] =
                        (2 * j + i % 2) * (float) mSquareSize;
                positions[(int) (i * mPatternSize.width * cn + j * cn + 2)] = 0;
            }
        }
        corners.create(mCornersSize, 1, CvType.CV_32FC3);
        corners.put(0, 0, positions);
    }

}
