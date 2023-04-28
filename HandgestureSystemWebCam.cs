using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;
using Unity.Barracuda;
using System.Linq;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityUtils;
using System.IO;
using System.Collections;
public class HandgestureSystemWebCam : MonoBehaviour
{
    public NNModel modelAsset;
    public string deviceName;
    public RawImage cameraImageRenderer; // Reference to the RawImage object in the scene
    public RawImage cameraImageRendererInput;
    public int handValence, handArousal;
    private Model m_RuntimeModel;
    private IWorker m_Worker;
  //  public OVRCameraRig cameraRig;
    public Camera leftCamera;
    public Camera rightCamera;
    // Set the device index to 0 to use the default webcam
    public int deviceIndex = 0;
    private WebCamTexture webcamTexture;
    private int counter;

    void Start()
    {
        // Load the PyTorch model
        m_RuntimeModel = ModelLoader.Load(modelAsset);
        m_Worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, m_RuntimeModel);

        // Start the webcam
        WebCamDevice[] devices = WebCamTexture.devices;
        deviceName = devices[0].name;
        Debug.Log("camname: " + deviceName);
        webcamTexture = new WebCamTexture(deviceName, 176, 144, 12);
        webcamTexture.requestedHeight = 144;
        webcamTexture.requestedWidth = 176;
        webcamTexture.Play();
    }

    void Update()
    {
        // Get the camera image
        Debug.Log("00WEBBB0width: " + webcamTexture.width + " height: " + webcamTexture.height);
        var cameraTexture2D = new Texture2D(webcamTexture.width, webcamTexture.height);
        cameraTexture2D.SetPixels(webcamTexture.GetPixels());
        //cameraTexture2D.Resize(240, 195);
        cameraTexture2D.Apply();
        Debug.Log("000width: " + cameraTexture2D.width + " height: " + cameraTexture2D.height);
        // Convert the color image to grayscale using the luminosity method
        var grayscaleTexture2D = new Texture2D(webcamTexture.width, webcamTexture.height);
        for (int i = 0; i < cameraTexture2D.width; i++)
        {
            for (int j = 0; j < cameraTexture2D.height; j++)
            {
                Color pixel = cameraTexture2D.GetPixel(i, j);
                float luminosity = 0.21f * pixel.r + 0.72f * pixel.g + 0.07f * pixel.b;
                grayscaleTexture2D.SetPixel(i, j, new Color(luminosity, luminosity, luminosity));
            }
        }
       // grayscaleTexture2D.Resize(240,195);
        grayscaleTexture2D.Apply();

        // Set the texture of the RawImage to the grayscaleTexture2D
        /*  RenderTexture rt = new RenderTexture(grayscaleTexture2D.width, grayscaleTexture2D.height, 0);
          Graphics.Blit(grayscaleTexture2D, rt);
          Material mat = new Material(Shader.Find("Custom/BlackAndWhite"));
          mat.mainTexture = rt;
          RenderTexture result = new RenderTexture(grayscaleTexture2D.width, grayscaleTexture2D.height, 0);
          RenderTexture.active = result;
          Graphics.Blit(rt, result, mat);
          Texture2D resultTexture = new Texture2D(grayscaleTexture2D.width, grayscaleTexture2D.height);
          resultTexture.ReadPixels(new Rect(0, 0, result.width, result.height), 0, 0);
          resultTexture.Apply();*/

       // Convert the webcam image to a Mat
        Mat img = new Mat(grayscaleTexture2D.height, grayscaleTexture2D.width, CvType.CV_8UC1);
        Utils.texture2DToMat(grayscaleTexture2D, img);

        // Apply binary thresholding
        Mat thresholded = new Mat();
        Imgproc.threshold(img, thresholded, 10, 255, Imgproc.THRESH_BINARY_INV | Imgproc.THRESH_OTSU);

        // Convert the thresholded image back to a Texture2D
        Texture2D thresholdedTexture = new Texture2D(thresholded.cols(), thresholded.rows(), TextureFormat.RGBA32, false);
        Utils.matToTexture2D(thresholded, thresholdedTexture);
      
        // Save the thresholded image to a file
      /*  string fileName = counter+".png";
        counter++;
        string filePath = Path.Combine( Application.dataPath + "/data/train/thumps_down/", fileName);
        File.WriteAllBytes(filePath, thresholdedTexture.EncodeToPNG());*/
       

        // Display the thresholded image
        // GetComponent<Renderer>().material.mainTexture = thresholdedTexture;

        cameraImageRenderer.texture = thresholdedTexture;

        var pixelCount = grayscaleTexture2D.width * grayscaleTexture2D.height;
        var data = new float[pixelCount];
        for (int i = 0; i < pixelCount; i++)
        {
            var x = i % grayscaleTexture2D.width;
            var y = i / grayscaleTexture2D.width;
            var color = grayscaleTexture2D.GetPixel(x, y);
            data[i] = color.r / 255.0f;
        }
      //  cameraImageRenderer.texture = grayscaleTexture2D;
        Debug.Log("width: "+ grayscaleTexture2D.width +" height: "+ grayscaleTexture2D.height);
        Debug.Log(data);
        // var cameraImage = new Tensor(1, 144, 175, 1, data);
        var cameraImage = new Tensor(grayscaleTexture2D, channels: 1);


        // Preprocess the image
        var input = cameraImage;
        input = input.Reshape(new TensorShape(1, 176, 144, 1));
      /*  var mean = input;
        for (int i = 0; i < input.length; i++)
        {
            input[i] -= mean[i % 1];
        }
        for (int i = 0; i < input.length; i++)
        {
            input[i] /= 0.5f;
        }*/
        // Run the model
        m_Worker.Execute(input);
        var output = m_Worker.PeekOutput();
        var probabilities = output.ToReadOnlyArray();

        // Get the predicted class
        var predictedClass = probabilities.ToList().IndexOf(probabilities.Max());
        // Do something with the predicted class
        if (predictedClass == 0)
        {
            Debug.Log("fist");
        } else
            if (predictedClass ==1)
        {
            Debug.Log("hand open");
        } else
            if (predictedClass ==2)
        {
            Debug.Log("thumps down");
        } else
            if (predictedClass ==3) {
            Debug.Log("thumps up");
        }
        else {
            Debug.Log("No gesture");
        }
       /* switch (predictedClass)
        {
            case 0:
                // Perform action for gesture 0
                Debug.Log("fist");
                break;
            case 1:
                // Perform action for gesture 1
                Debug.Log("hand open");
                break;
            case 2:
                // Perform action for gesture 2
                Debug.Log("thumps down");
                break;
            case 3:
                // Perform action for gesture 3
                Debug.Log("thumps up");
                break;
            default:
                // Perform action for other gestures or no gesture
                Debug.Log("No gesture");
                break;
        }*/
    }

    void OnDestroy()
    {
        // Release the resources used by the worker and the model
        m_Worker.Dispose();
       // m_RuntimeModel.Dispose();

        // Stop the webcam
        webcamTexture.Stop();
    }
}