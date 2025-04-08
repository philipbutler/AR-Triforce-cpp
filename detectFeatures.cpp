/*
Phil Butler

CS 5330 Computer Vision
Spring 2022

Project 4 Task 7

*/
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <iostream>
#include <cstdarg>
#include <vector>

#include "csv_util.h"

/*
 Main function.
 */
int main(int argc, char *argv[]) {
    
    int device_ID = (argc > 1) ? std::stoi(argv[1]) : 0;
    cv::VideoCapture *capdev;

    // open the video device
    capdev = new cv::VideoCapture(device_ID);
    if( !capdev->isOpened() ) {
            printf("Unable to open video device\n");
            return(-1);
    }

    // get some properties of the image
    cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                   (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("\nExpected size: %d %d\n", refS.width, refS.height);

    cv::namedWindow("AR", 1);
    cv::Mat frame, gray, corners;
    int num_corners = 15;
    
    char key = cv::waitKey(10);
    while (key != 'q') {
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        
        if ( frame.empty() ) {
            printf("frame is empty\n");
            break;
        }
        
        // Find the features
        frame.copyTo(gray);
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::goodFeaturesToTrack(gray, corners, num_corners, 0.01, 10);
        
        // Draw the features
        for (int i = 0; i < num_corners; i++) {
            cv::circle(frame, corners.at<cv::Point2f>(i), 20, cv::Scalar(0, 255, 0), 5);
        }
        
        cv::imshow("AR", frame);
        key = cv::waitKey(10);
    }
    delete capdev;
    return 0;
}
