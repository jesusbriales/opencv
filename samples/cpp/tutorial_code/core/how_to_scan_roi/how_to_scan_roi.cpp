#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <sstream>

using namespace std;
using namespace cv;

static void help()
{
    cout
        << "\n--------------------------------------------------------------------------"     << endl
        << "This program shows how to scan a ROI in image objects in OpenCV (cv::Mat)."
        << "As use case we take an input image and a random valid ROI, "                      << endl
        << "and compute the gradient as central difference in the ROI."                       << endl
        << "Shows C operator[] method and iterators for on-the-fly item address calculation." << endl
        << "Usage:"                                                                           << endl
        << "./howToScanRoi imageNameToUse [G]"                                                << endl
        << "if you add a G parameter the image is processed in gray scale"                    << endl
        << "--------------------------------------------------------------------------"       << endl
        << endl;
}

void ScanRoiAndGradientC(Mat& roi, Mat& dx, Mat& dy);
void ScanRoiAndGradientIterator(Mat& roi, Mat& dx, Mat& dy);
//void ScanRoiAndGradientRandomAccess(const Mat& roi, Mat& dx, Mat& dy);

int main( int argc, char* argv[])
{
    help();
    if (argc < 2)
    {
        cout << "Not enough parameters" << endl;
        return -1;
    }

    Mat img, roi, dx, dy;
    if( argc == 3 && !strcmp(argv[2],"G") )
        img = imread(argv[1], IMREAD_GRAYSCALE);
    else
        img = imread(argv[1], IMREAD_COLOR);

    if (img.empty())
    {
        cout << "The image" << argv[1] << " could not be loaded." << endl;
        return -1;
    }

    // set random ROI inside the input image
    RNG rng;
    Point2i ptUL( rng.uniform(0,img.cols), rng.uniform(0,img.rows) );
    Point2i ptBR( rng.uniform(ptUL.x+1,img.cols), rng.uniform(ptUL.y+1,img.rows) );
    Rect rect(ptUL,ptBR);
    // create ROI image (reference copy of a block in the original image)
    roi = Mat(img,rect);
    // preallocate gradient images
    dx = Mat(roi.rows,roi.cols, CV_32F);
    dy = Mat(roi.rows,roi.cols, CV_32F);

    const int times = 100;
    double t;

    t = (double)getTickCount();

    for (int i = 0; i < times; ++i)
    {
        ScanRoiAndGradientC(roi, dx,dy);
    }

    t = 1000*((double)getTickCount() - t)/getTickFrequency();
    t /= times;

    cout << "Time of reducing with the C operator [] (averaged for "
         << times << " runs): " << t << " milliseconds."<< endl;

    t = (double)getTickCount();

    for (int i = 0; i < times; ++i)
    {
        ScanRoiAndGradientIterator(roi, dx,dy);
    }

    t = 1000*((double)getTickCount() - t)/getTickFrequency();
    t /= times;

    cout << "Time of reducing with the iterator (averaged for "
        << times << " runs): " << t << " milliseconds."<< endl;

//    t = (double)getTickCount();

//    for (int i = 0; i < times; ++i)
//    {
//        ScanRoiAndGradientRandomAccess(roi, dx,dy);
//    }

//    t = 1000*((double)getTickCount() - t)/getTickFrequency();
//    t /= times;

//    cout << "Time of reducing with the on-the-fly address generation - at function (averaged for "
//        << times << " runs): " << t << " milliseconds."<< endl;

    // show input image, ROI and gradient images of the ROI
    namedWindow( "Input image", WINDOW_AUTOSIZE );
    rectangle( img, rect, Scalar( 50, 100, 255 ), 10 );
    imshow( "Input image", img );
    namedWindow( "ROI in image", WINDOW_AUTOSIZE );
    imshow( "ROI in image", roi );
    namedWindow( "Gradient in x", WINDOW_AUTOSIZE );
    imshow( "Gradient in x", dx );
    namedWindow( "Gradient in y", WINDOW_AUTOSIZE );
    imshow( "Gradient in y", dy );
    waitKey(0);                                          // Wait for a keystroke in the window

    return 0;
}

//! [scan-c]
void ScanRoiAndGradientC(Mat& roi, Mat& dx, Mat& dy)
{
    int channels = roi.channels();

    int nRows = roi.rows;
    int nCols = roi.cols * channels;

    if (roi.isContinuous())
    {
        nCols *= nRows;
        nRows = 1;
    }

    int i,j;
    const uchar* p;
    float* dxPtr = reinterpret_cast<float*>(dx.data);
    float* dyPtr = reinterpret_cast<float*>(dy.data);
    const int stride = roi.step[0];
    for( i = 0; i < nRows; ++i)
    {
        p = roi.ptr<uchar>(i);
        for ( j = 0; j < nCols; ++j, ++p, ++dxPtr, ++dyPtr )
        {
            *dxPtr = 0.5f * ( p[+1]-p[-1] );
            *dyPtr = 0.5f * ( p[+stride]-p[-stride] );
        }
    }
}
//! [scan-c]

//! [scan-iterator]
void ScanRoiAndGradientIterator(Mat& roi, Mat& dx, Mat& dy)
{
    const int channels = roi.channels();
    switch(channels)
    {
    case 1:
        {
            float* dxPtr = reinterpret_cast<float*>(dx.data);
            float* dyPtr = reinterpret_cast<float*>(dy.data);
            MatIterator_<uchar> it, end;
            const int stride = roi.step[0];
            for( it = roi.begin<uchar>(), end = roi.end<uchar>();
                 it != end; ++it, ++dxPtr, ++dyPtr)
            {
                uchar* imPtr = &(*it);
                *dxPtr = 0.5f * ( imPtr[+1]-imPtr[-1] );
                *dyPtr = 0.5f * ( imPtr[+stride]-imPtr[-stride] );
            }
            break;
        }
//    case 3:
//        {
//            MatIterator_<Vec3b> it, end;
//            for( it = roi.begin<Vec3b>(), end = roi.end<Vec3b>(); it != end; ++it)
//            {
//                (*it)[0] = table[(*it)[0]];
//                (*it)[1] = table[(*it)[1]];
//                (*it)[2] = table[(*it)[2]];
//            }
//        }
    }
}
//! [scan-iterator]

////! [scan-random]
//void ScanRoiAndGradientRandomAccess(const Mat& roi, Mat& dx, Mat& dy)
//{
//    const int channels = roi.channels();
//    switch(channels)
//    {
//    case 1:
//        {
//            for( int i = 0; i < roi.rows; ++i)
//                for( int j = 0; j < roi.cols; ++j )
//                    roi.at<uchar>(i,j) = table[roi.at<uchar>(i,j)];
//            break;
//        }
//    case 3:
//        {
//         Mat_<Vec3b> _I = roi;

//         for( int i = 0; i < roi.rows; ++i)
//            for( int j = 0; j < roi.cols; ++j )
//               {
//                   _I(i,j)[0] = table[_I(i,j)[0]];
//                   _I(i,j)[1] = table[_I(i,j)[1]];
//                   _I(i,j)[2] = table[_I(i,j)[2]];
//            }
//         roi = _I;
//         break;
//        }
//    }
//}
////! [scan-random]
