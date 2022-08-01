// Source codes for EE_576 Project 3
// Mehmet Yiğit Avcı
// Bogazici University, 2022


// including necessary libraries and initialization of variables

#ifndef HEADER_H
#define HEADER_H
#include <QCoreApplication>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/opencv.hpp"
#include <map>
using namespace std;
using namespace cv;

std::list<int> lab_list;
string img_name;
Mat img_hsv;
std::map<int, std::list<int>> label_equivalency2;
int label=2;
double m20_arr[10];
double m02_arr[10];
double area_arr[10];


Mat random_rgb(Mat label_matrix,int width,int height);
int weighted_euclidan_distance(Vec3b pixel1,Vec3b pixel2);
int geodesic(Vec3b pixel1,Vec3b pixel2);
int l1_distance(Vec3b pixel1,Vec3b pixel2);
int l2_distance(Vec3b pixel1,Vec3b pixel2);
double dist(Vec3b pixel1,Vec3b pixel2);
float distance(Vec3b pixel1,Vec3b pixel2);
Mat find_labels(Mat img,int th,std::map<int, std::list<int>> label_equivalency2,int width,int height,int *label);
Mat reduce_label_size(Mat label_matrix,int width,int height, int label_size,int label_arr[]);
Mat separate_HSV_obtain_H(Mat img);
int getMaxAreaContourId(vector <vector<cv::Point>> contours);
void find_sift(Mat img,string img_name,string y);
double avg(double numbers[], int count );
Mat draw_ellipse(Mat img,Mat drawed_img,double m20_arr[],double m02_arr[],double area_arr[],int count);

#endif // HEADER_H
