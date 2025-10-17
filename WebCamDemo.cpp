#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp> // For CascadeClassifier
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // Open the default webcam (index 0)
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open webcam." << endl;
        return -1;
    }

    // Load the Haar Cascade for face detection
    CascadeClassifier faceCascade;
    string cascadePath = "haarcascade_frontalface_default.xml";
    if (!faceCascade.load(cascadePath)) {
        cerr << "Error: Could not load Haar Cascade file: " << cascadePath << endl;
        return -1;
    }

    // Create a window to display the feed
    namedWindow("Webcam Feed", WINDOW_AUTOSIZE);

    Mat frame;
    while (true) {
        // Capture a frame from the webcam
        cap >> frame;
        if (frame.empty()) {
            cerr << "Error: Empty frame captured." << endl;
            break;
        }

        // Convert the frame to grayscale for face detection
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Detect faces
        vector<Rect> faces;
        faceCascade.detectMultiScale(
            gray,           // Input grayscale image
            faces,          // Output detected faces
            1.1,            // Scale factor
            3,              // Minimum neighbors
            0,              // Flags
            Size(30, 30)    // Minimum face size
        );

        // Draw rectangles around detected faces
        for (const auto& face : faces) {
            rectangle(
                frame,                      // Image to draw on
                face,                       // Rectangle coordinates
                Scalar(0, 255, 0),         // Green color (BGR)
                2                           // Thickness
            );
        }

        // Display the frame with detected faces
        imshow("Webcam Feed", frame);

        // Exit on ESC key (ASCII 27)
        if (waitKey(30) == 27) {
            break;
        }
    }

    // Clean up
    cap.release();
    destroyAllWindows();

    return 0;
}