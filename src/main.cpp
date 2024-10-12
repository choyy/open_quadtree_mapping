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
    quadmap::QParam params("../cfg/ORB_SLAM3/Monocular/TUM/tum_freiburg2_xyz.yaml");
    auto depthmap_ = std::make_shared<quadmap::Depthmap>(params);

    // 打开文件
    std::string                      path = "../../datasets/TUM/rgbd_dataset_freiburg2_xyz/";
    std::ifstream                    file("data/debug_photoslam_pose_img_rgbd_dataset_freiburg2.txt");
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
    for (int i = 1; i < img_paths.size(); i += 1) {
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
