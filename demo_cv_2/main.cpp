#include "bitplanes/demo_cv/demo.h"
#include "bitplanes/core/debug.h"
#include "bitplanes/core/homography.h"
#include "bitplanes/core/translation.h"
#include "bitplanes/core/affine.h"
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

int main(int argc, char *argv[])
{
    std::cout << std::string(argv[1]) + ", " + std::string(argv[2]) + ", " + std::string(argv[3]) + ", " + std::string(argv[4]) << std::endl;
    int roi_x = std::stoi(std::string(argv[2]));
    int roi_y = std::stoi(std::string(argv[3]));
    int roi_size = std::stoi(std::string(argv[4]));
    std::cout << "Hello World!" + std::to_string(roi_x) + ", " + std::to_string(roi_y) << std::endl;

    bp::AlgorithmParameters params;
    params.num_levels = 3;
    params.max_iterations = 32;
    params.subsampling = 1;
    params.verbose = false;
    params.function_tolerance= 1e-4;
    params.parameter_tolerance = 5e-5;
    params.sigma = 1.2;
    // params.linearizer = bp::AlgorithmParameters::LinearizerType::ForwardCompositional;

    std::unique_ptr<TrackerType> _tracker;
    _tracker.reset(new TrackerType(params));

    // // cv::Mat template_image;
    // // cv::Mat template_resize;

    // // template_image = cv::imread("template_dark.jpg");
    // // cv::resize(template_image, template_resize, cv::Size(256, 256));
    // // // template_resize = template_image;
    
    // // cv::Mat template_image_gray(template_resize.rows, template_resize.cols, 0);
    
    // cv::cvtColor(template_resize, template_image_gray, cv::COLOR_BGR2GRAY);

    cv::Rect template_image_roi(roi_x, roi_y, roi_size, roi_size);
    // _tracker->setTemplate(template_image_gray, template_image_roi);

    std::string input_video(argv[1]);
    cv::VideoCapture cap(input_video);
    bp::Result track_result;
    bp::Matrix33f tform(bp::Matrix33f::Identity());
    bp::Matrix33f mtxI(bp::Matrix33f::Identity());
    
    int frame_count = 0;
    const int image_size = 384;

    cv::Mat first_frame;
    cv::Mat last_frame;
    cap >> first_frame;
    cv::VideoWriter out_video(
        "outcpp.mp4",
        CV_FOURCC('M', 'P', '4', 'V'),
        10,
        cv::Size(image_size, image_size)
    );

    try {
        while(true) {
            cv::Mat image;
            cv::Mat source_image;
            cv::Mat image_gray;
            cv::Mat det_image;
            
            cap >> source_image;
            if (source_image.empty())
                break;
            
            cv::resize(source_image, image, cv::Size(image_size, image_size));
            // image = source_image;
            cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);

            if (frame_count % 2 != 0){
                frame_count++;
                continue;
            }

            if (frame_count == 0) {
                _tracker->setTemplate(image_gray, template_image_roi);
            }

            track_result = _tracker->track(image_gray, tform);
            tform = track_result.T;

            std::string det_image_path = "out_" + std::to_string(frame_count) + ".png";

            cv::Rect track_roi(
                template_image_roi.x + tform(0, 2),
                template_image_roi.y + tform(1, 2),
                roi_size,
                roi_size);
            // bp::DrawTrackingResult(det_image, image, track_roi, mtxI.data());
            bp::DrawTrackingResult(det_image, image, template_image_roi, tform.data());
            
            if (frame_count % 1 == 0) {
                std::cout << "transform mtx: " << tform << std::endl;
                template_image_roi = bp::RectToROI(template_image_roi, tform.data());
                tform = bp::Matrix33f::Identity();
                _tracker->setTemplate(image_gray, template_image_roi);
                
                // std::cout 
                //     << std::to_string(template_image_roi.x) << ", "
                //     << std::to_string(template_image_roi.y) << ", "
                //     << std::to_string(template_image_roi.width) << ", "
                //     << std::to_string(template_image_roi.height) << std::endl;


                if (template_image_roi.x < 0 || template_image_roi.y < 0 
                    || template_image_roi.x + template_image_roi.width >= image_size
                    || template_image_roi.y + template_image_roi.height >= image_size) {
                    break;
                }
            }

            if (frame_count % 10 == 0 || true) {
                cv::putText(
                    det_image, std::to_string(frame_count),
                    cv::Point(10, image_size - 10), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 0, 255), 3);
                std::vector<int> compression_params;
                compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
                compression_params.push_back(9);
                cv::imwrite(det_image_path, det_image, compression_params);

                out_video.write(det_image);

                std::cout << GREEN << "[SAVE] " + det_image_path << RESET << std::endl;
            }
            frame_count++;
        }
    }
    catch(bp::Error) {
        out_video.release();
        cap.release();
        return 0;
    }

    out_video.release();
    cap.release();
    return 0;
}