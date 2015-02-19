package org.opencv.android.services.calibration;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;

import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.text.InputType;
import android.view.Menu;
import android.view.MenuItem;
import android.view.SubMenu;
import android.widget.EditText;
import android.widget.Toast;

public class CalibrationBoard {

    public final Size mPatternSize = new Size(4, 11);
    public final int mGridFlags = Calib3d.CALIB_CB_ASYMMETRIC_GRID;
    public final int mCornersSize = (int)(mPatternSize.width * mPatternSize.height);
    protected double mSquareSize = 0.0155;

    public void calcBoardCornerPositions(Mat corners) {
        final int cn = 3;
        float positions[] = new float[mCornersSize * cn];

        // 4 x 11
        // *   *   *   *   *   *0  ^ Y
        //   *   *   *   *   *4    |
        // *   *   *   *   *   *1  |
        //   *   *   *   *   *5    |    X
        // *   *   *   *   *   *2  +---->
        //   *   *   *   *   *6
        // *   *   *   *   *   *3
        //   *   *   *   *   *7
        for (int i = 0; i < (int)mPatternSize.height; i++) {
            for (int j = 0; j < (int)mPatternSize.width; j++) {
                positions[(int) (i * mPatternSize.width * cn + j * cn + 0)] =
                        ((int)mPatternSize.height / 2) * (float)mSquareSize
                        - // OpenCV findCorners issue
                        i * (float) mSquareSize;
                positions[(int) (i * mPatternSize.width * cn + j * cn + 1)] =
                        ((int)mPatternSize.width) * (float)mSquareSize
                        - (2 * j + i % 2) * (float) mSquareSize;
                positions[(int) (i * mPatternSize.width * cn + j * cn + 2)] = 0;
            }
        }
        corners.create(mCornersSize, 1, CvType.CV_32FC3);
        corners.put(0, 0, positions);
    }

    private int mMenuGroupId = -1;
    private static final int mChangeSizeId = 0;
    public void onCreateMenu(Menu menu, int menuGroupId) {
        mMenuGroupId = menuGroupId;
        SubMenu mBoardMenu = menu.addSubMenu("Calibration board");
        mBoardMenu.add(mMenuGroupId, mChangeSizeId, Menu.NONE, "Change size...");
    }

    public boolean onMenuItemSelected(final Context context, MenuItem item) {
        if (item.getGroupId() == mMenuGroupId)
        {
            int id = item.getItemId();
            if (id == mChangeSizeId) {
                AlertDialog.Builder builder = new AlertDialog.Builder(context);
                builder.setTitle("Board size");

                final EditText input = new EditText(context);
                input.setInputType(InputType.TYPE_CLASS_NUMBER | InputType.TYPE_NUMBER_FLAG_SIGNED | InputType.TYPE_NUMBER_FLAG_DECIMAL);
                input.setText("" + mSquareSize);
                builder.setView(input);

                builder.setPositiveButton("OK", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        try {
                            Double boardSize = Double.valueOf(input.getText().toString());
                            mSquareSize = boardSize;
                            Toast.makeText(context, "New value saved: " + mSquareSize, Toast.LENGTH_LONG).show();
                            dialog.dismiss();
                        } catch (Exception e) {
                            e.printStackTrace();
                            Toast.makeText(context, "Can't update board size", Toast.LENGTH_LONG).show();
                        }
                    }
                });
                builder.setNegativeButton("Cancel", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        dialog.cancel();
                    }
                });
                builder.show();
                return true;
            }
        }
        return false;
    }
}
