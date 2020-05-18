#include "bitplanes/demo_cv/demo.h"
#include "bitplanes/core/debug.h"
#include "bitplanes/core/homography.h"
#include "bitplanes/core/bitplanes_tracker_pyramid.h"
#include "bitplanes/core/viz.h"

#include "bitplanes/utils/config_file.h"
#include "bitplanes/utils/error.h"

#include "bitplanes/demo_cv/bounded_buffer.h"

#include <memory>
#include <string>
#include <thread>
#include <iostream>

#include <opencv2/highgui.hpp>
#include <bitplanes/utils/utils.h>

typedef std::unique_ptr<cv::Mat> ImagePointer;
typedef bp::BitPlanesTrackerPyramid<bp::Homography> TrackerType;

int main()
{
    std::cout << "Hello World!" << std::endl;

    bp::AlgorithmParameters params;
    params.num_levels = 2;
    params.max_iterations = 50;
    params.subsampling = 2;
    params.verbose = false;

    std::unique_ptr<TrackerType> _tracker;
    _tracker.reset(new TrackerType(params));

    cv::Mat template_image;

    template_image = cv::imread("template.jpg");
    cv::Mat template_image_gray(template_image.rows, template_image.cols, 0);
    cv::cvtColor(template_image, template_image_gray, cv::COLOR_BGR2GRAY);

    cv::Rect template_image_roi(259, 284, 360 - 259, 416 - 284);
    _tracker->setTemplate(template_image_gray, template_image_roi);
    
    cv::VideoCapture cap("video.mp4");
    bp::Result track_result;
    bp::Matrix33f tform(bp::Matrix33f::Identity());
    int frame_count = 0;

    while(true) {
        cv::Mat image;
        cv::Mat image_gray;
        cv::Mat det_image;
        cap >> image;
        if (image.empty())
            break;
        cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
        track_result = _tracker->track(image_gray, tform);
        tform = track_result.T;

        std::string det_image_path = "out_" + std::to_string(frame_count) + ".png";
        bp::DrawTrackingResult(det_image, image, template_image_roi, tform.data());
        
        if (frame_count % 2 == 0) {
            template_image_roi = bp::RectToROI(template_image_roi, tform.data());
            _tracker->setTemplate(image_gray, template_image_roi);
        }

        std::vector<int> compression_params;
        compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);
        cv::imwrite(det_image_path, det_image, compression_params);
        std::cout << "[SAVE] " + det_image_path << std::endl;
        frame_count++;
    }

    return 0;
}