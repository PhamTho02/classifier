#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

void k_means()
{
    RNG rng(12345);
    int k, clusterCount = 3;
    int i, sampleCount = rng.uniform(300, 500);
    cout << "Sample count: " << sampleCount << endl;
    Mat points(sampleCount, 1, CV_32FC2), labels;
    clusterCount = MIN(clusterCount, sampleCount);
    std::vector<Point2f> centers;

    // Print the number of centers
    cout << "Number of centers: " << clusterCount << endl;

    /* generate random sample from multigaussian distribution */
    for (k = 0; k < clusterCount; k++)
    {
        Point center;
        center.x = rng.uniform(300, 500);
        center.y = rng.uniform(300, 500);
        cout << "center real: " << center << endl;
        Mat pointChunk = points.rowRange(k * sampleCount / clusterCount,
            k == clusterCount - 1 ? sampleCount :
            (k + 1) * sampleCount / clusterCount);
        rng.fill(pointChunk, RNG::NORMAL, Scalar(center.x, center.y), Scalar(30, 30));
    }
    //random center
    for (k = 0; k < clusterCount; k++)
    {
        Point center;
        center.x = rng.uniform(300, 500);
        center.y = rng.uniform(300, 500);
        cout << "Generated center before kmeans: " << center << endl;
    }
    randShuffle(points, 1, &rng);
    double compactness = kmeans(points, clusterCount, labels,
        TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 1000, 0.001),
        3, KMEANS_PP_CENTERS, centers);

    // Print the centers after kmeans
    cout << "Centers after kmeans:" << endl;
    for (const auto& center : centers)
    {
        cout << center << endl;
    }


}

void application_k_means()
{
    // Đọc ảnh
    Mat img = imread(R"(..\test_data\fire2.png)");
    if (img.empty()) {
        cout << "Error loading image!" << endl;
    }
    imshow("Original Image", img);
    cv::waitKey(0);
    // Chuyển đổi ảnh thành ma trận 2 chiều
    Mat X = img.reshape(1, img.total());
    X.convertTo(X, CV_32F); // Chuyển đổi kiểu dữ liệu về CV_32F

    // Các giá trị K cần phân cụm
    int K = 2; // Số cụm mong muốn

    // Áp dụng thuật toán KMeans
    Mat labels, centers;
    kmeans(X, K, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.2), 3, KMEANS_RANDOM_CENTERS, centers);

    // Tạo một ma trận mới để lưu ảnh kết quả
    Mat imgResult = Mat::zeros(img.size(), img.type());

    // Thay thế mỗi pixel bằng trung tâm của cụm mà nó thuộc về
    for (int i = 0; i < X.rows; i++) {
        int clusterIdx = labels.at<int>(i); // Lấy nhãn cụm của pixel
        imgResult.at<Vec3b>(i / img.cols, i % img.cols) = centers.at<Vec3f>(clusterIdx); // Gán màu tương ứng
    }

    // Hiển thị ảnh gốc

    // Hiển thị ảnh sau khi phân cụm
    imshow("KMeans Result", imgResult);

    // Chờ phím nhấn
    waitKey(0);
}


int main()
{
    //k_means();
    application_k_means();

}