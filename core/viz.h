/*
  This file is part of bitplanes.

  bitplanes is free software: you can redistribute it and/or modify
  it under the terms of the Lesser GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  bitplanes is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  Lesser GNU General Public License for more details.

  You should have received a copy of the Lesser GNU General Public License
  along with bitplanes.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef BITPLANES_CORE_VIZ_H
#define BITPLANES_CORE_VIZ_H

#include <array>
#include "bitplanes/core/internal/cvfwd.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace bp {

enum class ColorByName
{
  Red,
  Green,
  Blue,
  Black,
  White,
  Yellow
}; // ColorByName

/**
 * Draws the result of tracking
 *
 * \param dst output destination
 * \param src the input image
 * \param bbox bounding box of the template in the first frame
 * \param H    poitner to the tranform result (3x3 matrix col-major order)
 */
void DrawTrackingResult(cv::Mat& dst, const cv::Mat& src, const cv::Rect&,
                        const float* H, ColorByName = ColorByName::Yellow,
                        int line_thickness = 4, int line_type = 16,
                        int line_shift = 0);

std::array<cv::Point2f, 4> RectToPoints(const cv::Rect &r, const float *H_ptr);
cv::Rect RectToROI(const cv::Rect &r, const float *H_ptr);

}; // bp

#endif // BITPLANES_CORE_VIZ_H
