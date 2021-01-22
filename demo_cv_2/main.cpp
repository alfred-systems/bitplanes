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
#include <vector>

#include <opencv2/highgui.hpp>
#include <bitplanes/utils/utils.h>
#include <opencv2/optflow/rlofflow.hpp>

typedef std::unique_ptr<cv::Mat> ImagePointer;
typedef bp::BitPlanesTrackerPyramid<bp::Translation> TrackerType;
const int SKIP = 1;

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
    std::cout << std::string(argv[1]) + ", " + std::string(argv[2]) + ", " + std::string(argv[3]) << std::endl;
    int start_frame = std::stoi(std::string(argv[2]));
    int end_frame = std::stoi(std::string(argv[3]));
    std::cout << "Hello World!" + std::to_string(start_frame) + ", " + std::to_string(end_frame) << std::endl;

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

    // cv::Rect template_image_roi(roi_x, roi_y, roi_size, roi_size);
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
    cv::Mat last_frame_gray;
    cap >> first_frame;
    cv::VideoWriter out_video(
        "outcpp.mp4",
        cv::VideoWriter::fourcc('M', 'P', '4', 'V'),
        10,
        cv::Size(image_size, image_size));

    cv::Ptr<cv::optflow::RLOFOpticalFlowParameter> rlof_param = cv::optflow::RLOFOpticalFlowParameter::create();
    rlof_param->setSupportRegionType(cv::optflow::SR_FIXED);
    cv::Ptr<cv::optflow::SparseRLOFOpticalFlow>
        rlof = cv::optflow::SparseRLOFOpticalFlow::create(rlof_param);
    // auto rlof = cv::optflow::createOptFlow_SparseRLOF();

    std::vector<cv::Point2f> p0, p1;
    for(float y = 0; y < image_size; y+=15) {
        for(float x = 0; x < image_size; x+=15) {
            p0.push_back(cv::Point2f(x, y));
        }
    }

    cv::Mat mask = cv::Mat::zeros(cv::Size(image_size, image_size), CV_8UC3);
    std::vector<cv::Scalar> colors;
    cv::RNG rng;

    for (int i = 0; i < 100; i++)
    {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.push_back(cv::Scalar(r, g, b));
    }

    try {
        while(true) {
            cv::Mat image;
            cv::Mat source_image;
            cv::Mat image_gray;
            cv::Mat det_image;
            cv::Mat dot_image;

            std::vector<uchar> status;
            std::vector<float> err;

            cap >> source_image;
            if (source_image.empty())
                break;

            std::cout << BLUE << "[READ FRAME] " << RESET << std::to_string(frame_count) << std::endl;
            cv::resize(source_image, image, cv::Size(image_size, image_size));
            cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);

            if (frame_count == 0) {
                last_frame = image;
                last_frame_gray = image_gray;
            }
            else if (frame_count % SKIP != 0 || frame_count < start_frame)
            {
                last_frame = image;
                last_frame_gray = image_gray;
                frame_count++;
                continue;
            }
            else if (frame_count > end_frame) {
                break;
            }
            else {
                // rlof->calc(last_frame_gray, image_gray, p0, p1, status, err);
                cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);
                cv::calcOpticalFlowPyrLK(last_frame_gray, image_gray, p0, p1, status, err, cv::Size(15, 15), 2, criteria);

                det_image = dot_image = last_frame = image;
                last_frame_gray = image_gray;

                std::vector<cv::Point2f> good_new;
                

                for (uint i = 0; i < p0.size(); i++)
                {
                    // Select good points
                    if (status[i] == 1)
                    {
                        if (good_new.size() == 0) {
                            std::cout << "(" + std::to_string(p1[i].x) + ", " + std::to_string(p1[i].y) + ")" << std::endl;
                        }
                        
                        good_new.push_back(p1[i]);
                        // draw the tracks
                        cv::line(mask, p1[i], p0[i], colors[i % 100], 2);
                        cv::circle(dot_image, p1[i], 3, colors[i % 100], -1);
                        cv::add(dot_image, mask, det_image);
                    }
                }
                std::cout << BLUE << "[TRACK] " << RESET << std::to_string(good_new.size()) << std::endl;
                p0 = good_new;

                std::string det_image_path = "out_" + std::to_string(frame_count) + ".png";

                if (frame_count % SKIP == 0 || true) {
                    cv::putText(
                        det_image, std::to_string(frame_count),
                        cv::Point(10, image_size - 10), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 0, 255), 3);
                    std::vector<int> compression_params;
                    compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
                    compression_params.push_back(9);
                    // cv::imwrite(det_image_path, det_image, compression_params);

                    out_video.write(det_image);

                    std::cout << GREEN << "[SAVE] " + det_image_path << RESET << std::endl;
                }

                if (good_new.size() == 0){
                    p0.clear();
                    for (float y = 0; y < image_size; y += 10)
                    {
                        for (float x = 0; x < image_size; x += 10)
                        {
                            p0.push_back(cv::Point2f(x, y));
                        }
                    }
                    // break;
                }
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