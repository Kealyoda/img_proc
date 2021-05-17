#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <fstream>

using json = nlohmann::json;
using Frame = cv::Mat;

int MIN_MATCH_COUNT = 10;

std::vector<cv::Mat> startUp() {
    std::cout << "Input video location." << std::endl;
    std::string path;
    path = "C:/Users/romas/OneDrive/Desktop/cv_videos/test_list.mp4";
    //std::cin >> path;
    cv::VideoCapture cap(path);
    std::vector<cv::Mat> res;
    if (!cap.isOpened())
    {
        std::cout << "fgwsawdafv" << std::endl;
        return {};
    }

    while (true) {
        cv::Mat frm;
        cap >> frm;
        if (frm.empty()) {
            break;
        }
        cv::Mat gray_curr;
        gray_curr.convertTo(gray_curr, CV_8UC3);
        cv::cvtColor(frm, gray_curr, cv::COLOR_RGB2GRAY);
        res.push_back(gray_curr);
    }
    return res;
}
double d(double h11, double h12, double h13, double h21, double h22, double h23, double h31, double h32, double h33, double x, double y) {
    double c = h31 * x + h32 * y + h33;
    double res = sqrt((x - (h11 * x + h12 * y + h13) / c) * (x - (h11 * x + h12 * y + h13) / c) + (y - (h21 * x + h22 * y + h23) / c) * (y - (h21 * x + h22 * y + h23) / c));
    return res;
}
std::vector<cv::Point> readMarkUp() {
    //clockwise, starting from left top corner
    std::cout << "Input path to first frame mark up." << std::endl;
    std::string path_mark_up;
    path_mark_up = "C:/Users/romas/OneDrive/Desktop/cv_videos/test_list_frame.json";
    //std::cin >> path_mark_up;
    std::string photo_path = path_mark_up;
    std::ifstream s(photo_path);
    json j;
    s >> j;
    std::string metadata_index = j["_via_image_id_list"][0];
    std::vector<cv::Point> poly;
    for (int i = 0; i < j["_via_img_metadata"][metadata_index]["regions"].size(); i++) {
        for (int k = 0; k < j["_via_img_metadata"][metadata_index]["regions"][i]["shape_attributes"]["all_points_x"].size(); k++) {
            int x = j["_via_img_metadata"][metadata_index]["regions"][i]["shape_attributes"]["all_points_x"][k];
            int y = j["_via_img_metadata"][metadata_index]["regions"][i]["shape_attributes"]["all_points_y"][k];
            poly.push_back(cv::Point(x, y));
        }
    }
    return poly;
}

void featureMatching(std::vector<cv::Mat> frames, std::vector<cv::Point> polygon) {
    //prepare polygon;
    /*double alpha = 2.2;
    double b = 0;
    for (int y = 0; y < frames[0].rows; y++) {
        for (int x = 0; x < frames[0].cols; x++) {
            for (int c = 0; c < frames[0].channels(); c++) {
                frames[0].at<cv::Vec3b>(y, x)[c] =
                    cv::saturate_cast<uchar>(alpha * frames[0].at<cv::Vec3b>(y, x)[c] + b);
            }
        }
    }*/
    cv::cvtColor(frames[0], frames[0], CV_32FC1);
    cv::normalize(frames[0], frames[0], 1.0, 0.0, cv::NORM_MINMAX, CV_32FC1);
    pow(frames[0], 2.2, frames[0]);
    cv::normalize(frames[0], frames[0], 255, 0, cv::NORM_MINMAX, CV_8U);
    cv::Mat mask = cv::Mat::zeros(frames[0].rows, frames[0].cols, CV_8U);
    cv::fillPoly(mask, polygon, 255, 8);
    cv::Mat prev_mask = cv::Mat::zeros(frames[0].rows, frames[0].cols, CV_8U);
    cv::fillPoly(prev_mask, polygon, 255, 8);
    cv::resize(mask, mask, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
    cv::resize(prev_mask, prev_mask, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
    cv::imwrite("C:/Users/romas/OneDrive/Desktop/course_img/coursework/coursework/data/zmask_1.png", mask);
    cv::threshold(mask, mask, 100, 255, cv::THRESH_BINARY);
    cv::threshold(prev_mask, prev_mask, 100, 255, cv::THRESH_BINARY);
    //get image itself
    cv::Ptr<cv::SiftFeatureDetector> det = cv::SIFT::create();
    std::vector<cv::KeyPoint> kp_first_frame;
    cv::Mat descriptor_first_frame;
    GaussianBlur(frames[0], frames[0], cv::Size(5,5), 0, 0);
    cv::resize(frames[0], frames[0], cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
    det->detectAndCompute(frames[0], mask, kp_first_frame, descriptor_first_frame);
    cv::Mat out;
    cv::drawKeypoints(frames[0], kp_first_frame, out);
    //cv::imshow("|sgwdfsdax", out);
    //cv::waitKey(0);
    int FLAN_INDEX_KDTREE = 1;
    std::vector<cv::Mat> vid_output;
    //cv::Mat descriptor_prev_frame = descriptor_first_frame;
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    //std::vector<cv::KeyPoint> kp_prev_frame = kp_first_frame;
    std::vector<cv::Point> rect_p1;
    cv::RotatedRect minRectMask;
    minRectMask = cv::minAreaRect(polygon);
    cv::Point2f rect_mask[4];
    std::vector<cv::Point> rect_mask1;
    minRectMask.points(rect_mask);
    for (int j = 0; j < 4; ++j) {
        rect_mask1.push_back(rect_mask[j]);
        polygon[j].x *= 0.5;
        polygon[j].y *= 0.5;
    }
    for (int i = 1; i < frames.size(); i++) {
        /*double alpha = 2.2;
        double b = 0;
        for (int y = 0; y < frames[i].rows; y++) {
            for (int x = 0; x < frames[i].cols; x++) {
                for (int c = 0; c < frames[i].channels(); c++) {
                    frames[i].at<cv::Vec3b>(y, x)[c] =
                        cv::saturate_cast<uchar>(alpha * frames[i].at<cv::Vec3b>(y, x)[c] + b);
                }
            }
        } */
        cv::cvtColor(frames[i], frames[i], CV_32FC1);
        cv::normalize(frames[i], frames[i], 1.0, 0.0, cv::NORM_MINMAX, CV_32FC1);
        pow(frames[i], 2.2, frames[i]);
        cv::normalize(frames[i], frames[i], 255, 0, cv::NORM_MINMAX, CV_8U);
        std::string pa = "C:/Users/romas/OneDrive/Desktop/course_img/coursework/coursework/data/zlist_" + std::to_string(i) + ".png";
        std::string pam = "C:/Users/romas/OneDrive/Desktop/course_img/coursework/coursework/data/zmask_1.png";
        cv::Mat prev_mask = cv::imread(pam, cv::IMREAD_GRAYSCALE);
        std::vector<cv::KeyPoint> kp_prev_frame;
        cv::Mat descriptor_prev_frame;
        std::vector<cv::KeyPoint> kp_curr_frame;
        cv::Mat descriptor_curr_frame;
        GaussianBlur(frames[i], frames[i], cv::Size(5,5), 0, 0);

        cv::resize(frames[i], frames[i], cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
        det->detectAndCompute(frames[i-1], prev_mask, kp_prev_frame, descriptor_prev_frame);
        det->detectAndCompute(frames[i], cv::noArray(), kp_curr_frame, descriptor_curr_frame);
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descriptor_prev_frame, descriptor_curr_frame, knn_matches, 2);
        const float ration_thresh = 0.73f;
        std::vector<cv::DMatch> good_matches = {};
        for (int j = 0; j < knn_matches.size(); j++) {
            if (knn_matches[j][0].distance < ration_thresh * knn_matches[j][1].distance) {
                good_matches.push_back(knn_matches[j][0]);
            }
        }
        /*for (int i = 0; i < good_matches.size(); ++i) {
            std::cout << kp_curr_frame[good_matches[i].trainIdx-1].pt << std::endl;
        }*/
        cv::Mat img_matches;
        cv::Mat markers = cv::Mat::zeros(frames[i].size(), CV_8UC1);
        //std::cout << markers.cols << " " << markers.rows << std::endl;
        //std::cout << good_matches.size() << std::endl;
        for (int j = 0; j < good_matches.size(); ++j) {
            cv::circle(markers, kp_curr_frame[good_matches[j].trainIdx].pt, 2, 255, cv::FILLED, 8, 0);
        }
        int dilation_elem = 0;
        int dilation_size = 20;
        int const max_elem = 2;
        int const max_kernel_size = 21;
        int dilation_type = cv::MORPH_RECT; 
        cv::Mat element = cv::getStructuringElement(dilation_type,
            cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
            cv::Point(dilation_size, dilation_size));
        dilate(markers, markers, element);
        dilation_size = 3;
        element = cv::getStructuringElement(dilation_type,
            cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
            cv::Point(dilation_size, dilation_size));
        dilate(markers, markers, element);
        int erosion_elem = 0;
        int erosion_size = 22;
        int erosion_type = cv::MORPH_RECT;
        element = cv::getStructuringElement(erosion_type,
            cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
            cv::Point(erosion_size, erosion_size));
        erode(markers, markers, element);
        int thresh = 100;
        cv::RNG rng(12345);
        cv::Mat canny_output;
        cv::Canny(markers, canny_output, thresh, thresh * 2);
        std::vector<std::vector<cv::Point> > contours;
        findContours(canny_output, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
        std::vector<cv::RotatedRect> minRect(contours.size());
        std::vector<cv::RotatedRect> minEllipse(contours.size());
        for (size_t t = 0; t < contours.size(); t++)
        {
            minRect[t] = minAreaRect(contours[t]);
           
        } 
        
        //cv::imwrite("dhfusfh.png", markers);
        std::vector<cv::KeyPoint> kp_first_frame1;
        std::vector<cv::KeyPoint> kp_curr_frame1;
        std::vector<cv::DMatch> good_matches1 = {};
        cv::drawMatches(frames[i-1], kp_prev_frame, frames[i], kp_curr_frame, good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);


        double ma = 0;
        int ima = 0;
        //std::cout << minRect.size() << std::endl;
        prev_mask = cv::Mat::zeros(frames[0].rows, frames[0].cols, CV_8U);
        double ma1 = 0;
        double ima1 = 0;
        if (minRect.size() != 0) {
            
            for (size_t t = 0; t < contours.size(); t++)
            {
                // rotated rectangle
                cv::Point2f rect_points[4];
                minRect[t].points(rect_points);
                double area = abs((rect_points[3].x - rect_points[0].x) * (rect_points[1].y - rect_points[0].y) - (rect_points[1].x - rect_points[0].x) * (rect_points[3].y - rect_points[0].y));
                if (area > ma) {
                    ima = t;
                    ma = area;
                }

            }
            cv::Scalar color = cv::Scalar(0, 1, 0);
            cv::Point2f rect_points[4];
            std::vector<cv::Point> rect_p1;
            minRect[ima].points(rect_points);
            for (int j = 0; j < 4; j++)
            {
                rect_points[j].x += frames[i].cols;
                rect_p1.push_back(rect_points[j]);
                rect_p1[rect_p1.size() - 1].x -= frames[i].cols;
            }
            for (int j = 0; j < 4; j++)
            {
                line(img_matches, rect_points[j], rect_points[(j + 1) % 4], (0, 0, 255, 0), 1);
                line(prev_mask, rect_p1[j], rect_p1[(j + 1) % 4], cv::Scalar(0, 255, 0));
                
                //std::cout << rect_p1[j].x << " " << rect_p1[j].y << " ";
            }
            //std::cout << std::endl;
            
            cv::fillConvexPoly(prev_mask,
                rect_p1,
                255, 8);
            cv::imwrite(pam, prev_mask);
            cv::Mat H = findHomography(rect_p1, polygon);
            cv::Mat hom;
            cv::warpPerspective(frames[i], hom, H, cv::Size(frames[i].cols, frames[i].rows));
            cv::Mat hom_mask;
            cv::warpPerspective(prev_mask, hom_mask, H, cv::Size(frames[i].cols, frames[i].rows));
            std::string pah = "C:/Users/romas/OneDrive/Desktop/course_img/coursework/coursework/data/hom_zlist/frame_h_z_" + std::to_string(i) + ".png";
            
            cv::Mat new_img = cv::Mat::zeros(cv::Size(frames[i].cols, frames[i].rows), CV_8UC1);
            //std::cout << frames[i].cols << " " << frames[i].rows << std::endl;
            for (int n = 0; n < new_img.cols; ++n) {
                for (int m = 0; m < new_img.rows; ++m) {
                    if (hom_mask.at<uchar>(m,n) > 0) {   
                        new_img.at<uchar>(m,n) = hom.at<uchar>(m,n);
                    }
                }
            }
            cv::imwrite(pah, hom);
            double ma = 0;
            double x0, x1, y0, y1;
            if (polygon[1].x > polygon[0].x) {
                 x0 = polygon[0].x;
                 x1 = polygon[1].x;
            }
            else {
                 x1 = polygon[0].x;
                 x0 = polygon[1].x;
            }
            if (polygon[1].y > polygon[0].y) {
                 y0 = polygon[0].y;
                 y1 = polygon[1].y;
            }
            else {
                 y1 = polygon[0].y;
                 y0 = polygon[1].y;
            }
            
            for (double st = 0; st < 1; st += 0.05) {
                double x = x0 + (x1 - x0) * st;
                double y = y0 + (y1 - y0) * st;
                double res = d(H.at<double>(0, 0), H.at<double>(0, 1), H.at<double>(0, 2), H.at<double>(1, 0), H.at<double>(1, 1), H.at<double>(1, 2), H.at<double>(2, 0), H.at<double>(2, 1), H.at<double>(2, 2), x, y);
                if (res > ma) {
                    ma = res;
                }
            }
            if (polygon[2].x > polygon[1].x) {
                x0 = polygon[1].x;
                x1 = polygon[2].x;
            }
            else {
                x1 = polygon[1].x;
                x0 = polygon[2].x;
            }
            if (polygon[2].y > polygon[1].y) {
                y0 = polygon[1].y;
                y1 = polygon[2].y;
            }
            else {
                y1 = polygon[1].y;
                y0 = polygon[2].y;
            }

            for (double st = 0; st < 1; st += 0.05) {
                double x = x0 + (x1 - x0) * st;
                double y = y0 + (y1 - y0) * st;
                double res = d(H.at<double>(0, 0), H.at<double>(0, 1), H.at<double>(0, 2), H.at<double>(1, 0), H.at<double>(1, 1), H.at<double>(1, 2), H.at<double>(2, 0), H.at<double>(2, 1), H.at<double>(2, 2), x, y);
                if (res > ma) {
                    ma = res;
                }
            }
            if (polygon[3].x > polygon[2].x) {
                x0 = polygon[2].x;
                x1 = polygon[3].x;
            }
            else {
                x1 = polygon[2].x;
                x0 = polygon[3].x;
            }
            if (polygon[3].y > polygon[2].y) {
                y0 = polygon[2].y;
                y1 = polygon[3].y;
            }
            else {
                y1 = polygon[2].y;
                y0 = polygon[3].y;
            }

            for (double st = 0; st < 1; st += 0.05) {
                double x = x0 + (x1 - x0) * st;
                double y = y0 + (y1 - y0) * st;
                double res = d(H.at<double>(0, 0), H.at<double>(0, 1), H.at<double>(0, 2), H.at<double>(1, 0), H.at<double>(1, 1), H.at<double>(1, 2), H.at<double>(2, 0), H.at<double>(2, 1), H.at<double>(2, 2), x, y);
                if (res > ma) {
                    ma = res;
                }
            }
            if (polygon[0].x > polygon[3].x) {
                x0 = polygon[3].x;
                x1 = polygon[0].x;
            }
            else {
                x1 = polygon[3].x;
                x0 = polygon[0].x;
            }
            if (polygon[0].y > polygon[3].y) {
                y0 = polygon[3].y;
                y1 = polygon[0].y;
            }
            else {
                y1 = polygon[3].y;
                y0 = polygon[0].y;
            }

            for (double st = 0; st < 1; st += 0.05) {
                double x = x0 + (x1 - x0) * st;
                double y = y0 + (y1 - y0) * st;
                double res = d(H.at<double>(0, 0), H.at<double>(0, 1), H.at<double>(0, 2), H.at<double>(1, 0), H.at<double>(1, 1), H.at<double>(1, 2), H.at<double>(2, 0), H.at<double>(2, 1), H.at<double>(2, 2), x, y);
                if (res > ma) {
                    ma = res;
                }
            }
            std::cout << i << " " << ma << std::endl;
        } 
        
        cv::imwrite(pa, img_matches);
        
        vid_output.push_back(img_matches);
    }
    auto out_vid_final = cv::VideoWriter("sift8.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(vid_output[0].cols, vid_output[0].rows));
    for (int i = 0; i < vid_output.size(); i++) {
        out_vid_final.write(vid_output[i]);
    }
    out.release();
}


int main() {
    std::vector<cv::Mat> input_sequence = startUp();
    std::vector<cv::Point> input_poly = readMarkUp();
    featureMatching(input_sequence, input_poly);
}
