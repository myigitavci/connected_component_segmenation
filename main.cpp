// Source codes for EE_576 Project 3
// Mehmet Yiğit Avcı
// Bogazici University, 2022

// necessary declarations
#include <header.h>
#include "header.cpp"
#include <map>



int main(int argc, char *argv[])
{
    //reading and displaying the image
    cout << "Enter name of the image. ";
    cin >> img_name;

    //loading images and displaying
    Mat img  = imread("../576_final_project/SRIN-DATASET_R1/SRIN-DOORWAY/original_raw_data_doorway/close_door/84317a50-6142-4579-8bef-9bd56d07f6fc.JPG");
    imshow("Original Image", img);
    waitKey(1);

    // setting threshold
    int th=12;

    // separating hsv channels and displaying and obtaining hsv image
    cvtColor(img, img_hsv, COLOR_RGB2HSV);

    // obtaining width and height of the image
    int width=img.size().width;
    int height=img.size().height;

    // finding labels
    Mat label_matrix=find_labels(img_hsv,th,label_equivalency2,width,height,&label);
    //relabeling
    for (int k = label_equivalency2.size(); k >=0; k--) {
        lab_list=label_equivalency2[k];
        for (auto it = lab_list.end(); it != lab_list.begin(); it++)
        {

            label_matrix.setTo(k, label_matrix == *it);
        }

    }
    Mat first = random_rgb(label_matrix, width, height);
    imwrite("../576_project3/full_label_"+img_name+".bmp", first);

    // label size is reduced to 10
    int label_num=10;
    int label_arr[10];
    label_matrix= reduce_label_size( label_matrix, width, height,label_num,label_arr);

    // for each segment draw ellipses and extract sift features and calculate moments
    int count=0;
    cv::Mat img_segmented(img.size(), CV_8UC3);
    Mat label_matrix2 = label_matrix.clone();
    Mat drawed= Mat::zeros( width, height, CV_8UC3 );
    for(int y = 0; y <label_num ; y++){
        Mat label_mask = label_matrix.clone();
        label_mask.setTo(0, label_mask != label_arr[y]);
        label_mask.setTo(1, label_mask == label_arr[y]);

        label_mask.convertTo(label_mask, CV_8U);
        img.copyTo(img_segmented, label_mask);
        imwrite("../576_project3/mask.bmp", label_mask*255);

        string y_ = to_string(y);

        // get sift features and keypoints and write them on txt file for each segment
        find_sift(img_segmented,img_name,y_);
        // find moments and draw ellipses
       drawed= draw_ellipse(img_segmented,drawed,m20_arr,m02_arr,area_arr,count);
        count++;
    }

    //finding average values
     double m20_avg= avg(m20_arr, label_num);
     double m02_avg= avg(m02_arr, label_num);
     double area_avg= avg(area_arr, label_num);

    std::cout << " m20 avg is "  <<m20_avg  << " m02 avg is "  <<m02_avg  << " area avg is "  <<area_avg  <<'\n';

    // show the last segmented image
    Mat last = random_rgb(label_matrix2, width, height);
    //cout << last;
    imshow("a",last);
    waitKey(0);

    imwrite("../576_project3/last_"+img_name+".bmp", last);

    return 0;

}
