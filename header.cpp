// Function codes for EE_576 Project 3
// Mehmet Yiğit Avcı
// Bogazici University, 2022


#include<header.h>

// different distance metric functions are below this line
int weighted_euclidan_distance(Vec3b pixel1,Vec3b pixel2)
{
    int diff,h_diff;
    if(pixel1[0]-pixel2[0]<180)
    {
        h_diff=pixel1[0]-pixel2[0];
    }
    else{
        h_diff=360-(pixel1[0]-pixel2[0]);
    }
    diff=sqrt((pixel1[2]-pixel2[2])^2+pixel1[1]^2+pixel2[1]^2-2*pixel1[1]*pixel2[1]*h_diff);
    return diff;
}
int geodesic(Vec3b pixel1,Vec3b pixel2)
{
    int diff,h_diff;
    if(pixel1[0]-pixel2[0]<180)
    {
        h_diff=pixel1[0]-pixel2[0];
    }
    else{
        h_diff=360-(pixel1[0]-pixel2[0]);
    }
    diff=sqrt((pixel1[2]-pixel2[2])^2+(pixel1[1]+pixel2[1])^2+ (pixel1[1]^2)*h_diff);
    return diff;
}
int l1_distance(Vec3b pixel1,Vec3b pixel2)
{

    int diff=(abs(pixel1[0]-pixel2[0])+abs(pixel1[1]-pixel2[1])+abs(pixel1[2]-pixel2[2]));
    return diff;
}
int l2_distance(Vec3b pixel1,Vec3b pixel2)
{

    int diff=norm(pixel1-pixel2);
    return diff;
}
double dist(Vec3b pixel1,Vec3b pixel2)
{

    double dh = 2*min(abs(pixel1[0]-pixel2[0]), 180-abs(pixel1[0]-pixel2[0])) ;
    double ds = abs(pixel1[1]-pixel1[1]);
    double dv = abs(pixel1[2]-pixel1[2]);
    double diff = sqrt(dh*dh+ds*ds+dv*dv);
    return diff;
}
float distance(Vec3b pixel1,Vec3b pixel2)
{
    float dh = min(abs(pixel1[0]-pixel2[0]), 360-abs(pixel1[0]-pixel2[0])) / 180.0;
    float ds = abs(pixel1[1]-pixel1[1]);
    float dv = abs(pixel1[2]-pixel1[2]) / 255.0;
    float diff = sqrt(dh*dh+ds*ds+dv*dv);
    return diff;
}
// distance metric functions end

// generate random colors for each segment
Mat random_rgb(Mat label_matrix,int width,int height){
    Mat last = Mat::zeros(label_matrix.size(), CV_8UC3);
    map<int, Vec3b> color_map;
    int label2;
    for (int k = 0; k <= label; k++) {

        Vec3b color = Vec3b((uchar)255 * rand() / RAND_MAX, (uchar)255 * rand() / RAND_MAX, (uchar)255 * rand() / RAND_MAX);
        color_map[k] = color;

    }
    for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
            label2=label_matrix.at<int>(x, y);
            last.at<Vec3b>(x,y) = color_map.at(label2);
            //cout<< last.at<Vec3b>(x,y);

        }
    }


    return last;
}

// find the largest area contour
int getMaxAreaContourId(vector <vector<cv::Point>> contours) {
    double maxArea = 0;
    int maxAreaContourId = -1;
    for (int j = 0; j < contours.size(); j++) {
        double newArea = cv::contourArea(contours.at(j));
        if (newArea > maxArea) {
            maxArea = newArea;
            maxAreaContourId = j;
        } // End if
    } // End for
    return maxAreaContourId;
} // End function

// this function tries to but cannot draw ellipses, but calculates the moments and returns them
Mat draw_ellipse(Mat img,Mat drawed_img,double m20_arr[],double m02_arr[],double area_arr[],int count)
{   Mat gray;
    cvtColor(img, gray, COLOR_RGB2GRAY);
    RNG rng(12345);
    Mat canny_output;
    Canny( gray, canny_output, 100, 150 );
    vector<vector<Point> > contours;
    findContours( canny_output, contours, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );
    Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
    Moments mu =  moments(contours[getMaxAreaContourId(contours)], false);
    Point2d mc(mu.m10/(mu.m00+0.00001), mu.m01/(mu.m00+0.00001));
    double m20 = mu.m20 / (mu.m00+0.00001);
    double m02 = mu.m02 / (mu.m00+0.00001);
    m02_arr[count]=m02;
    m20_arr[count]=m20;
    area_arr[count]=contourArea(contours[getMaxAreaContourId(contours)]);
   // cout << "m20: " <<m20<< " m02: " <<m02<< "area: " <<contourArea(contours[getMaxAreaContourId(contours)])<< endl;;
    RotatedRect ell;
    ell = fitEllipse( contours[getMaxAreaContourId(contours)] );
    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
        ellipse( drawed_img, ell, color, 1 );

        imwrite("../576_project3/as.bmp", drawed_img);
     return drawed_img;
}

// average of array
double avg(double numbers[], int count )
{
    double sum = 0;         //sum is used to add all the values in the array
    double average;

    for (int i= 0 ; i < count; i++)
           sum += numbers[i];

    average = sum /(double)count;

    return average;
}

// this function finds SIFT keypoints and descriptors and writes them on a txt file
void find_sift(Mat img,string img_name,string y)
{
    ofstream myfile;
    myfile.open ("../576_project3/txts/keypoints_"+img_name+"segment"+y+".txt");
    auto detector = cv::SiftFeatureDetector::create();
    vector<KeyPoint>keypoints;
    detector->detect(img, keypoints);

    Mat img_w_keypoints,img_w_keypoints2;
    drawKeypoints(img, keypoints, img_w_keypoints);
    //drawKeypoints(sift_img, keypoints, img_w_keypoints2);

    vector<KeyPoint>::const_iterator it = keypoints.begin(), end = keypoints.end();

    myfile << "num key_pts: " << keypoints.size() << "\n";

    for( ; it != end; ++it ) {
        myfile << "  kpt x " << it->pt.x << ", y " << it->pt.y << "\n"; //<< ", size " << it->size << "\n";
    }

    // extract the sift descriptors from image
    auto extractor = cv::SiftDescriptorExtractor::create();

    Mat descriptors;
    extractor->compute(img, keypoints, descriptors);

    myfile << "\n\n\n\n";
    myfile << "descriptor size: " << descriptors.size() << "\n";
    myfile << descriptors;
    myfile.close();

    //imshow("sa",img_w_keypoints);
    imwrite("../576_project3/sa.bmp", img_w_keypoints);

}
// Connected component algorithm
Mat find_labels(Mat img,int th,std::map<int, std::list<int>> label_equivalency2,int width,int height,int *label)
{
    Mat label_matrix = Mat::ones(Size(width,height), CV_32SC1);
    Vec3b pixel,pixel_up;
    Vec3b pixel_left;
    double diff_left,diff_up;
    label_matrix.at<int>(0, 0)=*label;
    std::map<int, int> label_equivalency;
    Mat lab_eq = Mat::ones(Size(2,width*height), CV_32SC1);
    Mat lab_eq2 = Mat::zeros(Size(width,height), CV_32SC1);
    std::list<int> lab_list;
    int count=0;
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            pixel = img.at<Vec3b>(y, x);//storing value of (i,j) pixel in variable//
            //cout << "Value of pixel" << "(" << y << "," << x << ")" << "=" << pixel << endl;//showing the values in console window//
            if(x==0 && y==0)
            {
                label_matrix.at<int>(0, 0)=*label;

            }
            else
            {
                if (x!=0) // if we are not at first column
                {
                    pixel_left=img.at<Vec3b>(y, x-1);
                    diff_left=dist(pixel,pixel_left);
                    //  cout << diff_left<<  " "<<pixel[0]<<" " <<pixel[1] <<" "<<pixel[2]<<endl;
                    if (y==0) // if we are in first row it should not look at upper pixel
                    {
                        if (diff_left<th){
                            label_matrix.at<int>(y, x)=label_matrix.at<int>(y, x-1);

                        }
                        else
                        {
                            *label=*label+1;
                            label_matrix.at<int>(y, x)=*label;

                        }
                    }
                    else // if we are not in first row, we have to look at upper pixel
                    {
                        pixel_up=img.at<Vec3b>(y-1, x);
                        diff_up=dist(pixel,pixel_up);
                        //cout <<"up"<< diff_up << endl;
                        //cout <<"left"<< diff_left << endl;

                        if (diff_up<=th && diff_left<= th) // both left and upper pixels have same label
                        {

                            label_matrix.at<int>(y, x)= min(label_matrix.at<int>(y-1, x),label_matrix.at<int>(y, x-1));
                            if(label_matrix.at<int>(y-1, x)!=label_matrix.at<int>(y, x-1))
                            {
                                lab_list=label_equivalency2[min(label_matrix.at<int>(y-1, x),label_matrix.at<int>(y, x-1))];
                                lab_list.push_back(max(label_matrix.at<int>(y-1, x),label_matrix.at<int>(y, x-1)));
                                label_equivalency2[min(label_matrix.at<int>(y-1, x),label_matrix.at<int>(y, x-1))]=lab_list;
                                //                                for (auto v : lab_list)
                                //                                    std::cout << v << "\n";
                                //                                cout << "a"<< endl;

                                label_equivalency[max(label_matrix.at<int>(y-1, x),label_matrix.at<int>(y, x-1))]=min(label_matrix.at<int>(y-1, x),label_matrix.at<int>(y, x-1));
                                lab_eq.at<int>(count,0)=min(label_matrix.at<int>(y-1, x),label_matrix.at<int>(y, x-1));
                                lab_eq.at<int>(count,1)=max(label_matrix.at<int>(y-1, x),label_matrix.at<int>(y, x-1));
                                count++;
                            }
                        }
                        else if(diff_up<=th && diff_left> th)
                        {
                            label_matrix.at<int>(y, x)=label_matrix.at<int>(y-1, x);
                            //cout <<"2"<< endl;

                        }
                        else if(diff_up>th && diff_left<= th)
                        {
                            label_matrix.at<int>(y, x)=label_matrix.at<int>(y, x-1);
                            //cout <<"3"<< endl;

                        }
                        else if(diff_up>th && diff_left>th)
                        {
                            * label=*label+1;
                            label_matrix.at<int>(y, x)=*label;
                            //cout <<"4"<< endl;


                        }
                    }




                }
                else // if we are in first column, we should not look at left pixel
                {
                    pixel_up=img.at<Vec3b>(y-1, x);
                    diff_up=dist(pixel,pixel_up);
                    if (diff_up<th){
                        label_matrix.at<int>(y, x)=label_matrix.at<int>(y-1, x);
                    }
                    else
                    {
                        *label=*label+1;
                        label_matrix.at<int>(y, x)=*label;

                    }
                }
            }
        }

    }
    return label_matrix;
}

// since CCL algorithm gives too much label, I have reduced the number of labels
Mat reduce_label_size(Mat label_matrix,int width,int height,int label_size,int label_arr[])
{
    // 10 segment
    double minVal;
    double maxVal,maxVal2;
    Point minLoc;
    Point maxLoc;
    int l;
    minMaxLoc( label_matrix, &minVal, &maxVal, &minLoc, &maxLoc );
    Mat count_vec = Mat::zeros(Size(1,maxVal), CV_32SC1);
    Mat lab_max_count = Mat::zeros(Size(1,label_size), CV_32SC1);

    for (int x = 0; x < height; x++) {
        for (int y = 0; y < width; y++) {
            l =label_matrix.at<int>(x, y);
            count_vec.at<int>(0,l)=count_vec.at<int>(0,l)+1;
        }}
    for (int x = 0; x < label_size; x++)
    {
        minMaxLoc( count_vec, &minVal, &maxVal2, &minLoc, &maxLoc );
        lab_max_count.at<int>(0,x)=maxLoc.y;
        label_arr[x]=maxLoc.y  ;
        //cout <<maxLoc.y;
        count_vec.setTo(0, count_vec == maxVal2);
    }
    label_arr[label_size-1]=1;
    Mat label_mask = label_matrix.clone();
    for (int x = 0; x < label_size; x++)
    {

        label_mask.setTo(1, label_mask == lab_max_count.at<int>(0,x));

    }
    label_mask.setTo(0, label_mask != 1);
    // imshow("ssa",label_mask);
    // waitKey(0);
    // cout << label_mask;

    label_matrix = label_matrix.mul(label_mask);
    label_matrix.setTo(1, label_matrix == 0);

            return label_matrix;
}
