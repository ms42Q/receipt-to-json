#pragma once
#include <opencv2/core/core.hpp>     
#include <opencv2/highgui/highgui.hpp>     
#include <opencv2/imgproc/imgproc.hpp> 
#include <tesseract/baseapi.h>
#include <string>
#include <vector>
#include <regex>
#include <json/value.h>
#include <json/writer.h>
#define M_PI 3.14159265358979323846

namespace receipt{

    class Receipt{

        private:
            cv::Mat image_original;
            cv::Mat image_receipt;
            cv::Mat image_mask;
            cv::Mat image_mask_words;
            cv::Mat image_borders;
            std::vector <cv::Mat> image_segments;
            std::vector <Json::Value> items;
            std::string date;
            Json::Value receipt_json;

            void remove_borders();
            void dilate();
            void histogrammStretching();
            void remove_skew();
            void find_segments();
            void interp_segments();
            void build_receipt();
            std::string recognize_image( cv::Mat input );
            Json::Value interp_segment_text(std::string segment_text);

        public:
            Receipt();
            Receipt(char * filename);
            cv::Mat marked_word_segments;

            cv::Mat get_image_original();
            cv::Mat get_image_receipt();
            Json::Value get_receipt_json();
            
            cv::Mat set_image_original(char * filename);
            
            void improve_and_recognize();

    };


}
