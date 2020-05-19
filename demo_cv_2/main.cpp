#include "bitplanes/demo_cv/demo.h"
#include "bitplanes/core/debug.h"
#include "bitplanes/core/homography.h"
#include "bitplanes/core/translation.h"
#include "bitplanes/core/bitplanes_tracker_pyramid.h"
#include "bitplanes/core/viz.h"

#include "bitplanes/utils/config_file.h"
#include "bitplanes/utils/error.h"

#include "bitplanes/demo_cv/bounded_buffer.h"

#include <memory>
#include <string>
#include <thread>
#include <iostream>
#include <chrono>

#include <opencv2/highgui.hpp>
#include <bitplanes/utils/utils.h>

typedef std::unique_ptr<cv::Mat> ImagePointer;
typedef bp::BitPlanesTrackerPyramid<bp::Translation> TrackerType;

void get_optical_flow(TrackerType *tracker, cv::Mat *ref_frame, cv::Mat *new_frame) {
    int img_h = ref_frame->rows;
    int img_w = ref_frame->cols;
    int patch_size = 11;
    int stride = 21;
    float *flow_x = new float[img_w * img_h];
    float *flow_y = new float[img_w * img_h];

    // cv::Rect patch_roi(0, 0, patch_size, patch_size);
    // bp::Matrix33f tform(bp::Matrix33f::Identity());
    // tracker->setTemplate(*ref_frame, patch_roi);
    // bp::Result track_result = tracker->track(*new_frame, tform);

    for (int i = 2; i + patch_size < img_h - 1; i+=stride) {
        for (int j = 2; j + patch_size < img_w - 1; j+=stride)
        {   
            // std::cout << std::to_string(i) + ", " + std::to_string(j) << std::endl;
            cv::Rect patch_roi(j, i, patch_size, patch_size);
            bp::Matrix33f tform(bp::Matrix33f::Identity());

            tracker->setTemplate(*ref_frame, patch_roi);

            bp::Result track_result = tracker->track(*new_frame, tform);
        }
    }
}


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

    cv::Mat last_frame;
    while(true) {
        cv::Mat image;
        cv::Mat resize_image;
        cv::Mat image_gray;
        cv::Mat det_image;
        cap >> image;
        cv::resize(image, resize_image, cv::Size(256, 256));

        if (image.empty()) break;
        cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);

        // if (frame_count > 0) {
        //     using milli = std::chrono::milliseconds;
        //     auto start = std::chrono::high_resolution_clock::now();

        //     get_optical_flow(_tracker.get(), &last_frame, &image_gray);

        //     auto finish = std::chrono::high_resolution_clock::now();
        //     std::cout << "get_optical_flow() took "
        //               << std::chrono::duration_cast<milli>(finish - start).count()
        //               << " milliseconds\n";
        // }
        // last_frame = image_gray;
        // std::cout << "[DONE] " + std::to_string(frame_count) << std::endl;

        track_result = _tracker->track(image_gray, tform);
        tform = track_result.T;

        std::string det_image_path = "out_" + std::to_string(frame_count) + ".png";
        bp::DrawTrackingResult(det_image, image, template_image_roi, tform.data());
        
        if (frame_count % 1 == 0) {
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