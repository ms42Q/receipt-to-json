#include <iostream>
#include <opencv2/core/core.hpp>     
#include <opencv2/highgui/highgui.hpp>     
#include <opencv2/imgproc/imgproc.hpp> 
#include <vector>
#include "receipt.h"
#include <json/value.h>
#include <json/writer.h>

using namespace std;
using namespace cv;
using namespace receipt;

int main(int argc, char *argv[]){
    std::vector<Receipt> receipts;
    
    for(int i = 1; i < argc ; i++){
        receipts.push_back( Receipt(argv[1]));
        receipts[i-1].improve_and_recognize();
        
        //Display original image
        cv::namedWindow("Improved Image", 3);
        cv::Mat temp;
        cv::resize(receipts[i-1].get_image_receipt(), temp, 
                Size(cvRound(0.3*receipts[i-1].get_image_receipt().cols),
                     cvRound(0.3*receipts[i-1].get_image_receipt().rows)));
        cv::imshow("Improved Image", temp);

        //Display improved image
        cv::resize(receipts[i-1].get_image_original(), temp, 
                Size(cvRound(0.3*receipts[i-1].get_image_original().cols),
                     cvRound(0.3*receipts[i-1].get_image_original().rows)));
        cv::namedWindow("Original Image", 3);
        cv::imshow("Original Image", temp);
        
        //Print resulting Json-dictionary to stdout
        Json::Value dict = receipts[i-1].get_receipt_json();
        Json::StyledWriter json_wr;
        std::cout << json_wr.write(dict);
        
        cv::waitKey(0);

    }
    return 0;
    
}
