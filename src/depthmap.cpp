// This file is part of REMODE - REgularized MOnocular Depth Estimation.
//
// Copyright (C) 2014 Matia Pizzoli <matia dot pizzoli at gmail dot com>
// Robotics and Perception Group, University of Zurich, Switzerland
// http://rpg.ifi.uzh.ch
//
// REMODE is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// REMODE is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <quadmap/depthmap.h>

#include <utility>
quadmap::Depthmap::Depthmap(size_t  width,
                            size_t  height,
                            float   fx,
                            float   cx,
                            float   fy,
                            float   cy,
                            cv::Mat remap_1,
                            cv::Mat remap_2,
                            int     semi2dense_ratio)
    : width_(width), height_(height),
      seeds_(width, height, quadmap::PinholeCamera(fx, fy, cx, cy)),
      fx_(fx), fy_(fy), cx_(cx), cy_(cy) {
    seeds_.set_remap(std::move(remap_1), std::move(remap_2));
    seeds_.set_semi2dense_ratio(semi2dense_ratio);

    // printf("inremap_2itial the seed (%zu x %zu) fx: %f, fy: %f, cx: %f, cy: %f.\n", width, height, fx, fy, cx, cy);
}
quadmap::Depthmap::Depthmap(const QParam& p, int semi2dense_ratio)
    : width_(p.width), height_(p.height),
      seeds_(p.width, p.height, quadmap::PinholeCamera(p.fx, p.fy, p.cx, p.cy)),
      fx_(p.fx), fy_(p.fy), cx_(p.cx), cy_(p.cy) {

    float downsample_factor = 1;

    // initial the remap mat, it is used for undistort and also resive the image
    cv::Mat input_K = (cv::Mat_<float>(3, 3) << p.fx, 0.0F, p.cx, 0.0F, p.fy, p.cy, 0.0F, 0.0F, 1.0F);
    cv::Mat input_D = (cv::Mat_<float>(1, 5) << p.k1, p.k2, p.p1, p.p2, p.k3);
    // cv::Mat input_D = (cv::Mat_<float>(1, 5) << 0, 0, 0, 0, 0); // #todo: 不添加畸变系数点云效果看起来更好为什么?

    float   resize_fx        = p.fx * downsample_factor;
    float   resize_fy        = p.fy * downsample_factor;
    float   resize_cx        = p.cx * downsample_factor;
    float   resize_cy        = p.cy * downsample_factor;
    cv::Mat resize_K         = (cv::Mat_<float>(3, 3) << resize_fx, 0.0F, resize_cx, 0.0F, resize_fy, resize_cy, 0.0F, 0.0F, 1.0F);
    resize_K.at<float>(2, 2) = 1.0F;
    int resize_width         = p.width * downsample_factor;  // NOLINT
    int resize_height        = p.height * downsample_factor; // NOLINT

    cv::Mat undist_map1, undist_map2;
    cv::initUndistortRectifyMap(
        input_K,
        input_D,
        cv::Mat_<double>::eye(3, 3),
        resize_K,
        cv::Size(resize_width, resize_height),
        CV_32FC1,
        undist_map1, undist_map2);

    seeds_.set_remap(undist_map1, undist_map2);
    seeds_.set_semi2dense_ratio(semi2dense_ratio);

    // printf("[quadtree map]inremap_2itial the seed (%d x %d) fx: %f, fy: %f, cx: %f, cy: %f.\n", p.width, p.height, p.fx, p.fy, p.cx, p.cy);
}

bool quadmap::Depthmap::add_frames(const cv::Mat&    img_curr,
                                   const SE3<float>& T_curr_world) {
    std::lock_guard<std::mutex> lock(update_mutex_);
    cv::Mat                     img_gray;
    cv::cvtColor(img_curr, img_gray, cv::COLOR_RGB2GRAY);
    bool has_result = seeds_.input_raw(img_gray, T_curr_world);

    if (has_result) {
        seeds_.get_result(depth_out, debug_out, reference_out);
        T_world_ref = T_curr_world.inv();

        /// 提取并存储点云
        // const cv::Mat depth = depth_out;
        const cv::Mat depth = debug_out;

        // 图像边缘提取
        cv::Mat edges;
        cv::Canny(img_gray, edges, 50, 40);
        // pts_.clear();
        for (int y = 0; y < depth.rows; ++y) {
            for (int x = 0; x < depth.cols; ++x) {
                float depth_value = depth.at<float>(y, x);
                if (depth_value < 0.1 || depth_value > 10.0 || edges.at<uchar>(y, x) == 0) { continue; }
                const float3 f   = make_float3((x - cx_) / fx_, (y - cy_) / fy_, 1.0F); // NOLINT
                const float3 xyz = T_world_ref * (f * depth_value);

                const uint8_t intensity = reference_out.at<uint8_t>(y, x);
                if (intensity < kMinIntensity) { continue; }
                auto id = xyz2UniqeID(xyz);
                if (id_freq_map_[id]++ == 10) {
                    std::get<0>(pts_colors_).emplace_back(xyz.x);
                    std::get<0>(pts_colors_).emplace_back(xyz.y);
                    std::get<0>(pts_colors_).emplace_back(xyz.z);
                    auto c = img_curr.at<cv::Vec3b>(y, x);
                    std::get<1>(pts_colors_).emplace_back(c[0] / 255.F); // NOLINT
                    std::get<1>(pts_colors_).emplace_back(c[1] / 255.F); // NOLINT
                    std::get<1>(pts_colors_).emplace_back(c[2] / 255.F); // NOLINT
                };
            }
        }
    }

    return has_result;
}

// 给定一个三维坐标，返回一个唯一的ID
// x, y, z 分别乘以 kRVoxel(如100)后取整，保留后21位，再组合成一个64位的整数
uint64_t quadmap::Depthmap::xyz2UniqeID(const float3& xyz) const {
    const uint64_t last21bit = 0x00000000001FFFFF; // 保留后21位 0000 0000 001F FFFF

    auto id = (static_cast<uint64_t>(std::lround(xyz.x * kRVoxel)) & last21bit) << kMoveBits * 2
              | (static_cast<uint64_t>(std::lround(xyz.y * kRVoxel)) & last21bit) << kMoveBits
              | (static_cast<uint64_t>(std::lround(xyz.z * kRVoxel)) & last21bit);
    return id;
}

cv::Mat quadmap::Depthmap::getDepthmap() const {
    return depth_out;
}
cv::Mat quadmap::Depthmap::getDebugmap() const {
    return debug_out;
}
cv::Mat quadmap::Depthmap::getReferenceImage() const {
    return reference_out;
}
