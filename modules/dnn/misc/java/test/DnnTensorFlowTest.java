package org.opencv.test.dnn;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import org.junit.FixMethodOrder;
import org.junit.runners.MethodSorters;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.DictValue;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Importer;
import org.opencv.dnn.Layer;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.test.OpenCVTestCase;


@FixMethodOrder(MethodSorters.NAME_ASCENDING)
public class DnnTensorFlowTest extends OpenCVTestCase {

    String modelFileName = "";
    String sourceImageFile = "";

    Net net;

    @Override
    protected void setUp() throws Exception {
        super.setUp();

        Path path = Paths.get("");
        String currentRelativePath = path.toAbsolutePath().toString() + "/downloads/dnn/";
        modelFileName = currentRelativePath + "google/inception5/tensorflow_inception_graph.pb";
        sourceImageFile = currentRelativePath + "samples/space_shuttle.jpg";

        net = new Net();
        if(net.empty()) {
            Importer importer = Dnn.createTensorflowImporter(modelFileName);
            importer.populateNet(net);
        }

    }

    public void test30GetLayerTypes() {
        List<String> layertypes = new ArrayList();
        net.getLayerTypes(layertypes);

        assertFalse("No layer types returned!", layertypes.isEmpty());
    }

    public void test50GetLayer() {
        List<String> layernames = net.getLayerNames();

        assertFalse("Test net returned no layers!", layernames.isEmpty());

        String testLayerName = layernames.get(0);

        DictValue layerId = new DictValue(testLayerName);

        assertEquals("DictValue did not return the string, which was used in constructor!", testLayerName, layerId.getStringValue());

        Layer layer = net.getLayer(layerId);

        assertEquals("Layer name does not match the expected value!", testLayerName, layer.get_name());

    }

    public void test60LoadImage() {
        Mat rawImage = Imgcodecs.imread(sourceImageFile);

        assertNotNull("Loading image from file failed!", rawImage);

        Mat image = new Mat();
        Imgproc.resize(rawImage, image, new Size(224,224));

        Mat inputBlob = Dnn.blobFromImage(image);
        assertNotNull("Converting image to blob failed!", inputBlob);

        Mat inputBlobP = new Mat();
        Core.subtract(inputBlob, new Scalar(117.0), inputBlobP);

        net.setInput(inputBlobP, "input" );

        Mat result = net.forward();

        assertNotNull("Net returned no result!", result);

        Core.MinMaxLocResult minmax = Core.minMaxLoc(result.reshape(1, 1));

        assertTrue("No image recognized!", minmax.maxVal > 0.9);


    }

}
