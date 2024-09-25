#include <fstream>

#include <quadmap/depthmap.h>
#include <quadmap/check_cuda_device.cuh>

void savePoints2File(const std::vector<Eigen::Vector4f>& pc, const std::string& filename)
{
    std::ofstream outFile(filename); // 打开一个文件流用于写入
    for (const auto& p : pc) {
        outFile << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << std::endl;
    }
    outFile.close(); // 关闭文件流
    std::cout << "数据写入完成。" << std::endl;
}

int main(int argc, char **argv)
{
    Eigen::Matrix3f K;
    K << 517.306408, 0, 318.643040,
        0, 516.469215, 255.313989,
        0, 0, 1;
    int cam_width = 640;
    int cam_height = 480;
    float cam_fx = K(0,0);
    float cam_fy = K(1,1);
    float cam_cx = K(0,2);
    float cam_cy = K(1,2);
    double downsample_factor = 1;
    int semi2dense_ratio = 1;
    printf("read : width %d height %d\n", cam_width, cam_height);

    float k1, k2, r1, r2;
    k1 = k2 = r1 = r2 = 0.0;

    // initial the remap mat, it is used for undistort and also resive the image
    cv::Mat input_K = (cv::Mat_<float>(3, 3) << cam_fx, 0.0f, cam_cx, 0.0f, cam_fy, cam_cy, 0.0f, 0.0f, 1.0f);
    cv::Mat input_D = (cv::Mat_<float>(1, 4) << k1, k2, r1, r2);

    float resize_fx, resize_fy, resize_cx, resize_cy;
    resize_fx = cam_fx * downsample_factor;
    resize_fy = cam_fy * downsample_factor;
    resize_cx = cam_cx * downsample_factor;
    resize_cy = cam_cy * downsample_factor;
    cv::Mat resize_K = (cv::Mat_<float>(3, 3) << resize_fx, 0.0f, resize_cx, 0.0f, resize_fy, resize_cy, 0.0f, 0.0f, 1.0f);
    resize_K.at<float>(2, 2) = 1.0f;
    int resize_width = cam_width * downsample_factor;
    int resize_height = cam_height * downsample_factor;

    cv::Mat undist_map1, undist_map2;
    cv::initUndistortRectifyMap(
        input_K,
        input_D,
        cv::Mat_<double>::eye(3, 3),
        resize_K,
        cv::Size(resize_width, resize_height),
        CV_32FC1,
        undist_map1, undist_map2);

    auto depthmap_ = std::make_shared<quadmap::Depthmap>(
        resize_width, resize_height, resize_fx, resize_cx, resize_fy, resize_cy,
        undist_map1, undist_map2, semi2dense_ratio);

    // 打开文件
    std::string path = "../datasets/TUM/rgbd_dataset_freiburg1_desk/";
    std::ifstream file("data/pos_and_img_rgbd_dataset_freiburg1_desk.txt");
    std::string line;
    std::string imagePath;
    std::vector<quadmap::SE3<float>> Ts;
    std::vector<cv::Mat> imgs;
    if (file.is_open()) {
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            double timestamp, x, y, z, qx, qy, qz, qw;
            if (iss >> timestamp >> x >> y >> z >> qx >> qy >> qz >> qw >> imagePath) {
                quadmap::SE3<float> T_world_curr(qw, qx, qy, qz, x, y, z);
                cv::Mat img = cv::imread(path + imagePath, cv::IMREAD_GRAYSCALE);
                imgs.push_back(img);
                Ts.push_back(T_world_curr);
            }
        }
        file.close();
    } else {
        std::cerr << "Unable to open file";
    }

    bool has_result;
    std::vector<Eigen::Vector4f> pc;
    for(int i = 1; i < imgs.size(); i++) {
        has_result = depthmap_->add_frames(imgs[i], Ts[i].inv());
        if(!has_result) continue;
        {
            std::lock_guard<std::mutex> lock(depthmap_->getRefImgMutex());

            const cv::Mat depth = depthmap_->getDepthmap();
            // const cv::Mat depth = depthmap_->getDebugmap();

            const cv::Mat ref_img = depthmap_->getReferenceImage();
            const quadmap::SE3<float> T_world_ref = depthmap_->getT_world_ref();

            const float fx = depthmap_->getFx();
            const float fy = depthmap_->getFy();
            const float cx = depthmap_->getCx();
            const float cy = depthmap_->getCy();
            pc.clear();

            for (int y = 0; y < depth.rows; ++y) {
                for (int x = 0; x < depth.cols; ++x) {
                    float depth_value = depth.at<float>(y, x);
                    if (depth_value < 0.1) continue;
                    const float3 f = normalize(make_float3((x - cx) / fx, (y - cy) / fy, 1.0f));
                    const float3 xyz = T_world_ref * (f * depth_value);

                    const uint8_t intensity = ref_img.at<uint8_t>(y, x);
                    Eigen::Vector4f p(xyz.x, xyz.y, xyz.z, intensity);
                    pc.push_back(p);
                }
            }
        }
        std::cout << i << " number of points: " << pc.size() << std::endl;
        // if (i % 200 == 0) {
        //     savePoints2File(pc, "points" + std::to_string(i) + ".txt");
        // }
    }

    {
        std::lock_guard<std::mutex> lock(depthmap_->getRefImgMutex());

        const cv::Mat depth = depthmap_->getDepthmap();
        // const cv::Mat depth = depthmap_->getDebugmap();

        const cv::Mat ref_img = depthmap_->getReferenceImage();
        const quadmap::SE3<float> T_world_ref = depthmap_->getT_world_ref();

        const float fx = depthmap_->getFx();
        const float fy = depthmap_->getFy();
        const float cx = depthmap_->getCx();
        const float cy = depthmap_->getCy();
        pc.clear();

        for (int y = 0; y < depth.rows; ++y) {
            for (int x = 0; x < depth.cols; ++x) {
                float depth_value = depth.at<float>(y, x);
                if (depth_value < 0.1)
                    continue;
                const float3 f = normalize(make_float3((x - cx) / fx, (y - cy) / fy, 1.0f));
                const float3 xyz = T_world_ref * (f * depth_value);

                const uint8_t intensity = ref_img.at<uint8_t>(y, x);
                Eigen::Vector4f p(xyz.x, xyz.y, xyz.z, intensity);
                pc.push_back(p);
            }
        }
        savePoints2File(pc, "data/points.txt");

        //  ///*for debug*/
        // for(int y=0; y<depth.rows; ++y)
        // {
        //   for(int x=0; x<depth.cols; ++x)
        //   {
        //     float depth_value = depth.at<float>(y, x);
        //     if(depth_value < 0.1)
        //       continue;
        //     const float3 f = make_float3((x-cx)/fx, (y-cy)/fy, 1.0f);
        //     const float3 xyz = T_world_ref * ( f * depth_value );

        //     PointType p;
        //     p.x = x / 100.0;
        //     p.y = depth_value;
        //     p.z = - y / 100.0;
        //     const uint8_t intensity = ref_img.at<uint8_t>(y, x);
        //     p.intensity = intensity;
        //     pc->push_back(p);
        //   }
        // }
    }

    return 0;
}
