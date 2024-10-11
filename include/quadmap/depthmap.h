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

#pragma once

#include <mutex>
#include <unordered_map>
#include <vector>

#include <Eigen/Eigen>
#include <opencv2/imgproc/imgproc.hpp>

#include <quadmap/se3.cuh>
#include <quadmap/seed_matrix.cuh>
namespace quadmap {

class Depthmap {
public:
    Depthmap(
        size_t  width,
        size_t  height,
        float   fx,
        float   cx,
        float   fy,
        float   cy,
        cv::Mat remap_1,
        cv::Mat remap_2,
        int     semi2dense_ratio);

    bool add_frames(const cv::Mat&    img_curr,
                    const SE3<float>& T_curr_world);

    const cv::Mat_<float> getDepthmap() const;
    const cv::Mat_<float> getDebugmap() const;
    const cv::Mat         getReferenceImage() const;

    float getFx() const { return fx_; }

    float getFy() const { return fy_; }

    float getCx() const { return cx_; }

    float getCy() const { return cy_; }

    std::mutex& getUpdateMutex() { return update_mutex_; }

    size_t getWidth() const { return width_; }

    size_t getHeight() const { return height_; }

    SE3<float> getT_world_ref() const { return T_world_ref; }

    std::vector<float3> getPtsFreq(); // 获取频率大于阈值的3d点

private:
    SeedMatrix seeds_;
    size_t     width_;
    size_t     height_;
    float      fx_, fy_, cx_, cy_;

    std::mutex update_mutex_;

    SE3<float> T_world_ref;
    cv::Mat    depth_out;
    cv::Mat    reference_out;
    cv::Mat    debug_out;
    cv::Mat    current_img;

    const int   kMoveBits = 21;                       // 3d点xyz坐标表示的位数
    uint64_t    xyz2UniqeID(const float3& xyz) const; // 给每一个3d点创建一个唯一的id，3d点坐标范围有限制
    const float kRVoxel       = 50;                   // 体素尺寸的倒数
    const float kVoxelSize    = 1.F / kRVoxel;        // 体素尺寸
    const int   kMinIntensity = 50;                   // quadtree map最小强度阈值

    std::unordered_map<uint64_t, uint16_t> id_freq_map_; // 3d点id与频率的映射
    std::unordered_map<uint64_t, float3>   id_pts_map_;  // 3d点id与3d点坐标的映射
};

} // namespace quadmap