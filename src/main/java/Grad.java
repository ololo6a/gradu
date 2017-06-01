import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_face;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.objdetect.CascadeClassifier;

import java.nio.DoubleBuffer;
import java.nio.IntBuffer;

import static org.bytedeco.javacpp.opencv_core.CV_32SC1;
import static org.bytedeco.javacpp.opencv_face.createLBPHFaceRecognizer;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.opencv.imgcodecs.Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;
/**
 * Created by Администратор on 01.06.2017.
 */
public class Grad {
    int N = 64;
    int sqrt = 8;
    String [] s = {"l","d","p","n"};
    int [] imagecount = {0,0,0,0};
    int [][] value = new int [4][N];
    int[] c = {5,6,2,4};
    String [] ss = {"1.","3.","2.","4."};
    opencv_face.FaceRecognizer[] faceRecognizer = new opencv_face.FaceRecognizer[N+1] ;

    void createTable(){
        int a = 0; // Начальное значение диапазона - "от"
        int b = 4; // Конечное значение диапазона - "до"

        for (int k = 0; k < s.length; k++) {
            for (int i = 0; i < N; i++) {
                value[k][i]= 0 + (int) (Math.random() * 4);
            }
        }


    }

    void createRecognizer(){

        opencv_core.MatVector images = new opencv_core.MatVector(s.length);
        org.bytedeco.javacpp.opencv_core.Mat labels = new org.bytedeco.javacpp.opencv_core.Mat(s.length, 1, CV_32SC1);
        IntBuffer labelsBuf = labels.createBuffer();
        for (int j = 1; j <= N; j++) {
        for (int i = 0; i < s.length; i++) {
                opencv_core.Mat img = imread("images\\" + s[i] +"\\ideal"+"\\"+ss[i]+ j + ".JPG",CV_LOAD_IMAGE_GRAYSCALE);
                images.put(i, img);
                labelsBuf.put(i, i);
            }
            faceRecognizer[j] = createLBPHFaceRecognizer();
            faceRecognizer[j].train(images, labels);
        }
    }
    void findFaces(){

        CascadeClassifier faceDetector = new CascadeClassifier("haarcascade_frontalface_alt.xml");
        for (int j = 0; j < c.length; j++) {
            int i=0;
            for (int k = 1; k <=c[j] ; k++) {

                Mat image = Imgcodecs.imread("images\\" + s[j] +"\\"+k+".jpg",CV_LOAD_IMAGE_GRAYSCALE);
                MatOfRect faceDetections = new MatOfRect();
                faceDetector.detectMultiScale(image, faceDetections);

                for (Rect rect : faceDetections.toArray()) {
                    i++;
                    //lets create separated face by 9 pieces
                    for (int n = 1; n <= N; n++) {
                        int e = n-1;
                         Rect rectCrop = new Rect(rect.x+e%sqrt*rect.width/sqrt, rect.y+e/sqrt*rect.height/sqrt, rect.width/sqrt, rect.height/sqrt);
                        Mat image_roi = new Mat (image,rectCrop);
                        Imgcodecs.imwrite("images\\" + s[j] +"\\sep"+"\\"+i+"."+ n + ".JPG",image_roi);
                    }
                }
            }
            imagecount[j]=i;
        }
    }

    void predict()
    {
        int ans[] = new int[sqrt];
        double conf[] = new double[sqrt];
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= 4 ; j++) {

                int T = 0;
                if (j==2) T++;
                if (j==3) T+=sqrt;
                if (j==4) T+=sqrt+1;


                if ((i+T)<1 || (i+T)>N) continue;
                IntPointer label = new IntPointer(1);

                DoublePointer confidence = new DoublePointer(1);
                IntBuffer b =  label.asBuffer();
                DoubleBuffer c = confidence.asBuffer();

                opencv_core.Mat testImage = imread("test\\"+ "2." + i +".jpg",CV_LOAD_IMAGE_GRAYSCALE);
                faceRecognizer[i+T].predict(testImage, b, c);
                ans[label.get(0)]++;
                conf[label.get(0)]+=c.get(0);
            }
        }
        for (int i = 0; i < 4; i++) {
            if (ans[i]!=0) System.out.println(ans[i] + "  " + conf[i]/ans[i]);
            else System.out.println(0);
        }
    }

}
