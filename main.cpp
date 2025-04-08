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
 Prints out a 2D vector. Used exclusively for inspecting the corner set list.
 */
int print2DVector(std::vector<std::vector<cv::Point2f> > v) {
    std::cout << "\n\nCorner set list:";
    for (std::vector<cv::Point2f> u : v) {
        for (cv::Point2f p : u) {
            std::cout << "\n\tx: " << p.x << "\ty: " << p.y;
        }
    }
    std::cout << "\n";
    return 0;
}

/*
 Prints out a 3D matrix. Used exclusively for inspecting the camera matric.
 */
void print3x3Mat(cv::Mat &camera_matrix) {
    std::cout << "Camera Matrix:\n";
    std::cout << "\n" << camera_matrix.at<double>(0, 0) << " ";
    std::cout << camera_matrix.at<double>(0, 1) << " ";
    std::cout << camera_matrix.at<double>(0, 2) << " ";
    
    std::cout << "\n" << camera_matrix.at<double>(1, 0) << " ";
    std::cout << camera_matrix.at<double>(1, 1) << " ";
    std::cout << camera_matrix.at<double>(1, 2) << " ";
    
    std::cout << "\n" << camera_matrix.at<double>(2, 0) << " ";
    std::cout << camera_matrix.at<double>(2, 1) << " ";
    std::cout << camera_matrix.at<double>(2, 2) << "\n\n";
}

/*
 Fills the point set with the 3D coordinates of what the inner chessboard corners correspond to.
 */
void fillPointSet(std::vector<cv::Vec3f> &point_set) {
    for (int y = 0; y < 9; y++) {
        for (int x = 0; x < 6; x++) {
            point_set.emplace_back(cv::Vec3f(x, -1*y, 0));
        }
    }
}

/*
 Indicates whether the camera is calibrated or not yet.
 */
enum Mode {
    CALIBRATED, UNCALIBRATED
};

/*
 The main function.
 
 By default, the default camera will be used i.e. cv::VideoCapture(0) .
 To specify that a different device should be used, enter the corresponding
 device ID (on my machine, specifying 1 will use the my webcam).
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
    cv::Mat frame;
    
    int chessBoardFlags = cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE;
    cv::Size patternSize(6, 9);
    char numCornersText[256];
    char pointText[256];
    char rotationText[256];
    char translationText[256];
    std::vector<cv::Vec3f> point_set;
    point_set.reserve(54);
    fillPointSet(point_set);
    
    //for (int i = 0; i < 54; i++) {
    //    std::cout << point_set.at(i) << "\n";
    //}
    
    char intrinsic_params_file[256];
    strcpy(intrinsic_params_file, "./iParams.csv");
    char camMatLabel[256];
    strcpy(camMatLabel, "Camera Matrix: ");
    char distCoefLabel[256];
    strcpy(distCoefLabel, "Distortion Coefficients: ");
    
    std::vector<std::vector<cv::Vec3f> > point_list;
    std::vector<std::vector<cv::Vec3f> > new_point_list;
    std::vector<cv::Point2f> corner_set;
    std::vector<std::vector<cv::Point2f> > corner_list;
    std::vector<float> distCoeffs;
    distCoeffs.reserve(14);
    cv::Mat rvecs;
    cv::Mat tvecs;
    float error;
    
    *capdev >> frame;
    
    // Initialize camera matrix
    cv::Mat camera_matrix(3, 3, CV_64FC1, cv::Scalar(0));
    camera_matrix.at<double>(0, 0) = 1;
    camera_matrix.at<double>(1, 1) = 1;
    camera_matrix.at<double>(2, 2) = 1;
    camera_matrix.at<double>(0, 2) = frame.cols / 2;
    camera_matrix.at<double>(1, 2) = frame.rows / 2;
    print3x3Mat(camera_matrix);
    std::vector<float> flatCamMat;
    flatCamMat.reserve(9);
    
    std::vector<std::vector<float>> data;
    
    Mode mode = UNCALIBRATED;
    
    // Try to load intrinsic Parameters.
    if (read_image_data_csv(intrinsic_params_file, data, 1) == 0) {
        // First row of data will be flattened camera matrix
        int k = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                camera_matrix.at<double>(i, j) = data.at(0).at(k);
                k++;
            }
        }
    
        // Second row of data will be distortion coefficients
        for (float n : data.at(1)) {
            distCoeffs.emplace_back(n);
        }
        mode = CALIBRATED;
    }
    
    int savedTextTimeLeft = 0;
    
    char key = cv::waitKey(10);
    while (key != 'q') {
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        
        if ( frame.empty() ) {
            printf("frame is empty\n");
            break;
        }
        
        // Find & Draw corners
        bool found = findChessboardCorners(frame, patternSize, corner_set, chessBoardFlags);
        cv::drawChessboardCorners(frame, patternSize, corner_set, found);
        
        // Display info about found corners
        std::sprintf(numCornersText, "Number of corners found: %lu", corner_set.size());
        cv::putText(frame, numCornersText, cv::Point(20, 30), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0,255,0));
        if (corner_set.size() >= 1) {
            std::sprintf(pointText, "First point coordinates: (%f, %f)", corner_set[0].x, corner_set[0].y);
            cv::putText(frame, pointText, cv::Point(20, 70), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0,255,0));
            cv::Mat viewGray;
            cvtColor(frame, viewGray, cv::COLOR_BGR2GRAY);
            cv::cornerSubPix(viewGray, corner_set, cv::Size(11, 11), cv::Size(-1,-1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
        } else {
            std::sprintf(pointText, "First point coordinates: (N/A, N/A)");
            cv::putText(frame, pointText, cv::Point(20, 70), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0,255,0));
        }
        
        // Save corners in this frame
        if (found and key == 's') {
            savedTextTimeLeft = 70;
            corner_list.push_back(corner_set);
            point_list.push_back(point_set);
            //print2DVector(corner_list);
            std::cout << "Corner List Size: " << corner_list.size() << '\n';
            
            // Do calibration
            // Note: Even if the intrinsic parameters were loaded from a file,
            // saving 5 images will recalibrate the camera
            if (corner_list.size() >= 5) {
                std::cout << "\nCalibrating Camera!\n";
                error = cv::calibrateCamera(point_list, corner_list, refS, camera_matrix, distCoeffs, rvecs, tvecs, cv::CALIB_FIX_ASPECT_RATIO);
                mode = CALIBRATED;
                print3x3Mat(camera_matrix);
                std::cout << "Error (RMS): " << error << "\n\nDistortion Coefficients:\n";
                for (int i = 0; i < distCoeffs.size(); i++) {
                    std::cout << distCoeffs.at(i) << " ";
                }
            }
        }
        
        // Write intrinsic parameters to file
        if (key == 'w') {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    flatCamMat.emplace_back(camera_matrix.at<double>(i, j));
                }
            }
            std::cout << "\nWriting intrinsic parameters to file.\n";
            append_image_data_csv(intrinsic_params_file, camMatLabel, flatCamMat);
            append_image_data_csv(intrinsic_params_file, distCoefLabel, distCoeffs);
        }
        
        // While time is left, will tell user that points have been saved
        if (savedTextTimeLeft > 0) {
            cv::putText(frame, "Corner points saved.", cv::Point(20, 110), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0,255,0));
            savedTextTimeLeft--;
        }
        
        // Inserting virtual objects into the scene
        if (found and mode == CALIBRATED) {
            cv::Mat rvec, tvec;
            //int r1, r2, r3, t1, t2, t3;
            cv::Mat imagePoints;
            imagePoints.reserve(3); // TO DO - necessary?
            
            // Calculate & display pose
            cv::solvePnP(point_set, corner_set, camera_matrix, distCoeffs, rvec, tvec);
            std::sprintf(rotationText, "Rotation: [%d, %d, %d]", (int) rvec.at<float>(0), (int) rvec.at<float>(1), (int) rvec.at<float>(2));
            std::sprintf(translationText, "Translation: [%d, %d, %d]", (int) tvec.at<float>(0), (int) tvec.at<float>(1), (int) tvec.at<float>(2));
            cv::putText(frame, rotationText, cv::Point(20, 150), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0,255,0));
            cv::putText(frame, translationText, cv::Point(20, 190), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0,255,0));
            
            // Project axes points
            double m[3][3] = {{10, 0, 0}, {0, 10, 0}, {0, 0, 10}};
            cv::Mat objectPoints = cv::Mat(3, 3, CV_64F, m);
            cv::projectPoints(objectPoints, rvec, tvec, camera_matrix, distCoeffs, imagePoints);
            
            // Draw axes
            cv::line(frame, corner_set.at(0), imagePoints.at<cv::Point2d>(0), cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
            cv::line(frame, corner_set.at(0), imagePoints.at<cv::Point2d>(1), cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            cv::line(frame, corner_set.at(0), imagePoints.at<cv::Point2d>(2), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            
            // Create & project triforce points (combine with axes points?)
            cv::Mat triforce2DPoints;
            triforce2DPoints.reserve(6);
            double t[6][3] = {{0, 1, 1}, {0, 2, 1}, {0, 3, 1}, {0, 1.5, 1.866025}, {0, 2.5, 1.866025}, {0, 2, 2.732051}};
            cv::Mat triforce3DPoints = cv::Mat(6, 3, CV_64F, t);
            cv::projectPoints(triforce3DPoints, rvec, tvec, camera_matrix, distCoeffs, triforce2DPoints);
            
            // Draw Triforce
            cv::line(frame, triforce2DPoints.at<cv::Point2d>(0), triforce2DPoints.at<cv::Point2d>(1), cv::Scalar(0, 234, 255), 5, cv::LINE_AA);
            cv::line(frame, triforce2DPoints.at<cv::Point2d>(1), triforce2DPoints.at<cv::Point2d>(2), cv::Scalar(0, 234, 255), 5, cv::LINE_AA);
            cv::line(frame, triforce2DPoints.at<cv::Point2d>(0), triforce2DPoints.at<cv::Point2d>(3), cv::Scalar(0, 234, 255), 5, cv::LINE_AA);
            cv::line(frame, triforce2DPoints.at<cv::Point2d>(1), triforce2DPoints.at<cv::Point2d>(3), cv::Scalar(0, 234, 255), 5, cv::LINE_AA);
            cv::line(frame, triforce2DPoints.at<cv::Point2d>(1), triforce2DPoints.at<cv::Point2d>(4), cv::Scalar(0, 234, 255), 5, cv::LINE_AA);
            cv::line(frame, triforce2DPoints.at<cv::Point2d>(2), triforce2DPoints.at<cv::Point2d>(4), cv::Scalar(0, 234, 255), 5, cv::LINE_AA);
            cv::line(frame, triforce2DPoints.at<cv::Point2d>(3), triforce2DPoints.at<cv::Point2d>(4), cv::Scalar(0, 234, 255), 5, cv::LINE_AA);
            cv::line(frame, triforce2DPoints.at<cv::Point2d>(3), triforce2DPoints.at<cv::Point2d>(5), cv::Scalar(0, 234, 255), 5, cv::LINE_AA);
            cv::line(frame, triforce2DPoints.at<cv::Point2d>(4), triforce2DPoints.at<cv::Point2d>(5), cv::Scalar(0, 234, 255), 5, cv::LINE_AA);
        } else {
            cv::putText(frame, "Rotation: (N/A)", cv::Point(20, 150), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0,255,0));
            cv::putText(frame, "Translation: (N/A)", cv::Point(20, 190), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0,255,0));
        }
        
        cv::imshow("AR", frame);
        key = cv::waitKey(10);
    }

    delete capdev;
    return(0);
}
