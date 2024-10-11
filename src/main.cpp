#include <fstream>

#include <quadmap/check_cuda_device.cuh>
#include <quadmap/depthmap.h>
#include <string>

void savePoints2File(const std::vector<Eigen::Vector4f>& pc, const std::string& filename) {
    std::ofstream outFile(filename); // 打开一个文件流用于写入
    for (const auto& p : pc) {
        outFile << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << std::endl;
    }
    outFile.close(); // 关闭文件流
    std::cout << "数据写入完成。" << std::endl;
}

void saveColors2File(const std::vector<cv::Vec3b>& vc, const std::string& filename) {
    std::ofstream outFile(filename); // 打开一个文件流用于写入
    for (const auto& p : vc) {
        outFile << static_cast<float>(p[0]) / 255 << " " << static_cast<float>(p[1]) / 255 << " " << static_cast<float>(p[2]) / 255 << std::endl;
    }
    outFile.close(); // 关闭文件流
    std::cout << "数据已写入 " << filename << std::endl;
}

int main(int argc, char** argv) {
    Eigen::Matrix3f K;
    K << 517.306408, 0, 318.643040,
        0, 516.469215, 255.313989,
        0, 0, 1;
    int    cam_width         = 640;
    int    cam_height        = 480;
    float  cam_fx            = K(0, 0);
    float  cam_fy            = K(1, 1);
    float  cam_cx            = K(0, 2);
    float  cam_cy            = K(1, 2);
    double downsample_factor = 1;
    int    semi2dense_ratio  = 1;
    printf("read : width %d height %d\n", cam_width, cam_height);

    float k1, k2, r1, r2;
    k1 = k2 = r1 = r2 = 0.0;

    // initial the remap mat, it is used for undistort and also resive the image
    cv::Mat input_K = (cv::Mat_<float>(3, 3) << cam_fx, 0.0f, cam_cx, 0.0f, cam_fy, cam_cy, 0.0f, 0.0f, 1.0f);
    cv::Mat input_D = (cv::Mat_<float>(1, 4) << k1, k2, r1, r2);

    float resize_fx, resize_fy, resize_cx, resize_cy;
    resize_fx                = cam_fx * downsample_factor;
    resize_fy                = cam_fy * downsample_factor;
    resize_cx                = cam_cx * downsample_factor;
    resize_cy                = cam_cy * downsample_factor;
    cv::Mat resize_K         = (cv::Mat_<float>(3, 3) << resize_fx, 0.0f, resize_cx, 0.0f, resize_fy, resize_cy, 0.0f, 0.0f, 1.0f);
    resize_K.at<float>(2, 2) = 1.0f;
    int resize_width         = cam_width * downsample_factor;
    int resize_height        = cam_height * downsample_factor;

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
    std::string                      path = "../datasets/TUM/rgbd_dataset_freiburg2_xyz/";
    std::ifstream                    file("data/photoslam_pose_img_rgbd_dataset_freiburg2.txt");
    std::string                      line;
    std::string                      imagePath;
    std::vector<quadmap::SE3<float>> Twcs;
    std::vector<cv::Mat>             imgs;
    std::vector<std::string>         img_paths;
    if (file.is_open()) {
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            double             timestamp, x, y, z, qx, qy, qz, qw;
            if (iss >> timestamp >> x >> y >> z >> qx >> qy >> qz >> qw >> imagePath) {
                quadmap::SE3<float> Twc(qw, qx, qy, qz, x, y, z);
                // cv::Mat img = cv::imread(path + imagePath, cv::IMREAD_COLOR);
                // imgs.push_back(img);
                img_paths.push_back(path + imagePath);
                Twcs.push_back(Twc);
            }
        }
        file.close();
    } else {
        std::cerr << "Unable to open file";
    }

    std::cout << "data loaded..." << std::endl;
    // for(int i = 1; i < imgs.size(); i++) {
    for (int i = 1; i < img_paths.size(); i += 10) {
        // for(int i = 1; i < 250; i++) {
        cv::Mat img = cv::imread(img_paths[i], cv::IMREAD_COLOR);
        bool has_result  = depthmap_->add_frames(img, Twcs[i].inv());
        if (!has_result) {
            std::cout << "not initialized..." << std::endl;
            continue;
        }
        // std::cout << i << std::endl;
        std::cout << i << " number of points: " << depthmap_->getPtsFreq().size() << std::endl;
        // if (i % 200 == 0) {
        //     savePoints2File(pc, "points" + std::to_string(i) + ".txt");
        // }
    }

    std::ofstream outFile("data/pointsall.txt"); // 打开一个文件流用于写入
    for (const auto& p : depthmap_->getPtsFreq()) {
        outFile << p.x << " " << p.y << " " << p.z << std::endl;
    }
    outFile.close(); // 关闭文件流
    std::cout << "数据写入完成。" << std::endl;

    return 0;
}
