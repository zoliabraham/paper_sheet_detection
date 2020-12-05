#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

#include "edge_detector.hpp"

using namespace cv;
using namespace std;

int main() {
    Mat image = imread("img1.png");
    EdgeDetector edgeDetector = EdgeDetector();
    //vector<cv::Point> points = EdgeDetector::detect_edges(image);

    edgeDetector.debug_squares(image);


    resize(image, image, Size(), 0.2, 0.2);
    imshow("original", image);
    waitKey(0);
    return 0;
}
