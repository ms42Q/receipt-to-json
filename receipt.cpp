#include "receipt.h"
#include <iostream>


namespace receipt {

    Receipt::Receipt(){
        date = "";
    }

    Receipt::Receipt(char * filename){
        this->image_original = cv::imread(filename);
        image_borders = cv::Mat( image_original.size(), CV_8U, cv::Scalar(0) );
    }


    cv::Mat Receipt::get_image_original(){
        return this->image_original;
    }

    cv::Mat Receipt::get_image_receipt(){
        return this->image_receipt;
    }

    void Receipt::remove_borders(){

        cv::Mat tempimage;
        cv::Mat bwmask(image_original.size(), CV_8U, cv::Scalar(0));
        std::vector<cv::Mat> hsv;
        
        cv::cvtColor(image_original, tempimage, cv::COLOR_RGB2HSV);
        cv::split(tempimage, hsv);
        hsv[1] = cv::Mat::zeros(hsv[1].rows, hsv[1].cols, hsv[1].type());
        cv::merge(hsv,tempimage);
        cv::cvtColor(tempimage, tempimage, cv::COLOR_HSV2RGB);

        cv::Mat mask(tempimage.size(), CV_32S, cv::Scalar(0));
        /// Draw a line from theupper left to the lower left (Background marker)
        cv::line(mask, cv::Point(5, mask.rows - 5),
                cv::Point(5,5), cv::Scalar(255),3);
        /// Draw a line from the upper right to the lower right (Background marker)
        cv::line(mask, cv::Point(mask.cols-5, 5),
                cv::Point(mask.cols-5, mask.rows-5), 
                cv::Scalar(255),3);
        /// Draw a line in the center (Object marker)
        cv::line(mask, cv::Point(mask.cols/2, mask.rows/2 - 50),
                cv::Point(mask.cols/2, mask.rows/2 + 50),
                cv::Scalar(100),10);
        cv::watershed(tempimage, mask);
        
        cv::Point2i upper_left(0,tempimage.rows);
        cv::Point2i bottom_right(0,0);
        for(int col = 0; col < mask.cols; col++){
            for(int row = 0; row < mask.rows ; row++){
                if(mask.at<int>(row,col) == 100){
                    bwmask.at<uchar>(row,col) = 255; 
                }
                else if(mask.at<int>(row,col) == -1 && 
                        row != 0 && col != 0 &&
                        row != mask.rows-1 && 
                        col != mask.cols-1){
                    if (upper_left.x == 0) upper_left.x = col;
                    if (upper_left.y > row) upper_left.y = row;
                    bottom_right.x = col;
                    if (bottom_right.y < row) bottom_right.y = row;
                    image_borders.at<uchar>(row,col) = 255;
                }
            }
        }

        bwmask.copyTo(image_mask);
        tempimage.copyTo(image_receipt, image_mask);

        cv::Mat image_mask_not(image_original.size(), CV_8U, cv::Scalar(0));
        cv::bitwise_not(image_mask, image_mask_not);
        cv::cvtColor(this->image_receipt, this->image_receipt, cv::COLOR_RGB2GRAY);
        cv::bitwise_not(image_receipt,image_receipt,image_mask_not);
        
        std::vector<cv::Point2i> points;
        points.push_back(upper_left);
        points.push_back(bottom_right);
        cv::Rect bound_rect = cv::boundingRect(points);
        cv::Mat segment_image(bound_rect.height, bound_rect.width, CV_8U, cv::Scalar(255));
        for(int row = 0; row < segment_image.rows; row++){
            int *rowptr = (int*)segment_image.ptr(row);
            for(int col = 0; col < segment_image.cols; col++){
                segment_image.at<uchar>(row,col) = image_receipt.at<uchar>(bound_rect.tl().y+row, bound_rect.tl().x+col);
            }
        }
        segment_image.copyTo(image_receipt);
    }

    void Receipt::histogrammStretching(){
        int wmin = 0;
        int wmax = 255;
        int brightest = 0;
        int darkest = 255;

        // Find darkest and brightest pixelvalue
        for(int row=0; row < image_receipt.rows; row++){
            for(int col=0; col < image_receipt.cols; col++){
                if(image_receipt.at<uchar>(row,col) < darkest &&
                        image_mask.at<uchar>(row,col) != 0){
                    darkest = image_receipt.at<uchar>(row,col);
                }
                if(image_receipt.at<uchar>(row,col) > brightest &&
                        image_mask.at<uchar>(row,col) != 0){ 
                    brightest = image_receipt.at<uchar>(row,col);
                }
            }
        }

        
        for(int row=0; row < image_receipt.rows; row++){
            for(int col=0; col < image_receipt.cols; col++){
                double gnew = image_receipt.at<uchar>(row,col) - darkest;
                gnew = gnew * (((double)wmax-(double)wmin)/((double)brightest-(double)darkest)) + wmin;
                if(gnew > 255) gnew = 255;
                else if(gnew < 0) gnew = 0;
                image_receipt.at<uchar>(row,col) = cvRound(gnew);
            }
        }
    }


    void Receipt::improve_and_recognize(){
        this->remove_borders();
        this->histogrammStretching();
        cv::threshold(image_receipt, image_receipt, 115 , 255, cv::THRESH_BINARY);
        remove_skew();
        dilate();
        find_segments();
        this->interp_segments();
        this->build_receipt();
    }

    void Receipt::remove_skew(){
        double roh = 1;
        double thetha = M_PI/180.0;
        float angle;
        int threshold = 60; 
        int top_x;
        int bot_x;
        cv::Mat skewedimage;
        cv::Mat rotation;
        std::vector<cv::Vec2f> lines;
        cv::Point2f center(skewedimage.cols*0.5, skewedimage.rows*0.5);
        
        image_borders.copyTo(skewedimage);
        
        cv::HoughLines(skewedimage, lines, roh, thetha, threshold);
        std::vector<cv::Vec2f>::const_iterator iterator = lines.begin();
        while(iterator != lines.end()){
            float rho = (*iterator)[0];
            float theta = (*iterator)[1];
            angle = (theta/M_PI)*180;
            if(theta < M_PI / 4.0 || theta > 3.0*M_PI/4.0){
                //vertical line
                cv::line(skewedimage, 
                        cv::Point(rho/std::cos(theta),0),
                        cv::Point((rho-skewedimage.rows*std::sin(theta))/
                            std::cos(theta), skewedimage.rows),
                        cv::Scalar(180),
                        11);
                top_x = cvRound(rho/std::cos(theta));
                bot_x = (rho-skewedimage.rows*std::sin(theta)) /
                    std::cos(theta);
                break; //Stop when the first horizontal line was found
            }
        }
        if(top_x < bot_x) angle =(180-angle)*-1;
        rotation = cv::getRotationMatrix2D(center,angle,1);
        cv::warpAffine(image_receipt, image_receipt, rotation, 
                       cv::Size(image_receipt.cols, image_receipt.rows),
                       cv::INTER_CUBIC, cv::BORDER_CONSTANT, 255);
    }


    void Receipt::dilate(){
        double kernelcols = image_receipt.cols * 0.05;
        cv::Mat char_mask;
        cv::Mat kernel(cv::Size(cvRound(kernelcols),1), CV_8U, cv::Scalar(1));
        cv::Mat kernel_closing(cv::Size(7,19), CV_8U, cv::Scalar(1));
        cv::Mat kernel_closing_2(cv::Size(13,1), CV_8U, cv::Scalar(1));
        std::vector<cv::Point> seed_points;
        cv::bitwise_not(image_receipt, char_mask);
        
        //Fuse words together by "smearing" each pixel to the left and to the right
        cv::dilate(char_mask, char_mask, kernel, cv::Point(kernelcols-(kernelcols/3.0),0));

        //Remove any particle-groups that are to thin to be recognized as a character (remove noise)
        cv::erode(char_mask, char_mask, kernel_closing, cv::Point(3,9));
        cv::dilate(char_mask, char_mask, kernel_closing, cv::Point(3,9));
        
        cv::threshold(char_mask, char_mask, 100, 255, cv::THRESH_BINARY);
        
        //close horizontal holes
        cv::dilate(char_mask, char_mask, kernel_closing_2, cv::Point(6,0));
        cv::erode(char_mask, char_mask, kernel_closing_2, cv::Point(6,0));
        
        char_mask.copyTo(this->image_mask_words);
    }
            

    void Receipt::find_segments(){
        cv::Mat label_image;
        cv::threshold(image_mask_words, label_image, 2,1,cv::THRESH_BINARY); //255 -> 1 Rest 0
        label_image.convertTo(label_image, CV_32S);
        
        int label_index = 2;
        for(int row = 0; row < label_image.rows; row++){
            int *rowptr = (int*)label_image.ptr(row);
            for(int col = 0; col < label_image.cols; col++){
                if(label_image.at<int>(row,col) != 1) continue;
                
                //Object found. Mark area with current index
                std::vector <cv::Point2i> segment;
                cv::Rect bounding_rect;
                cv::floodFill(label_image, cv::Point(col,row), label_index, &bounding_rect, 0, 0, 4);
                
                segment.push_back(bounding_rect.tl());
                segment.push_back(bounding_rect.br());

                //Search for horizontal neighbours of this object and add them
                int object_vcenter = (bounding_rect.height/2) + bounding_rect.tl().y;
                for(int i=0; i<label_image.cols; i++){
                    if(label_image.at<int>(object_vcenter, i) !=1) continue;
                    cv::Rect temp_rect;
                    cv::floodFill(label_image, cv::Point(i,object_vcenter), label_index, &temp_rect, 0, 0, 4);
                    segment.push_back(temp_rect.tl());
                    segment.push_back(temp_rect.br());
                    } 
                bounding_rect = cv::boundingRect(segment);
                cv::Mat segment_image(bounding_rect.height, bounding_rect.width, CV_8U, cv::Scalar(255));
                //Extract objects
                for(int seg_row = 0; seg_row < segment_image.rows; seg_row++){
                    int *seg_rowptr = (int*)segment_image.ptr(row);
                    for(int seg_col = 0; seg_col < segment_image.cols; seg_col++){
                        if(label_image.at<int>(bounding_rect.tl().y+seg_row, bounding_rect.tl().x+seg_col) != label_index) 
                            continue;
                            segment_image.at<uchar>(seg_row,seg_col) = image_receipt.at<uchar>(bounding_rect.tl().y+seg_row, 
                                                                       bounding_rect.tl().x+seg_col);
                        }
                    }
                image_segments.push_back(segment_image);
                label_index++;
                }
            }
        }

    void Receipt::interp_segments(){
        Json::Value item;
        for(int i=0; i < image_segments.size(); i++){
            std::string segment_text = recognize_image(image_segments[i]);
            item = interp_segment_text(segment_text);
            if(item == 0) continue;
            else if(item.isMember("price")) items.push_back(item);
        }
    }

    std::string Receipt::recognize_image( cv::Mat image ){
        std::string text;
        tesseract::TessBaseAPI tess;
        tess.Init("/usr/share/tesseract-ocr/tessdata/","deu");
        tess.SetImage((uchar*)image.data, image.size().width, image.size().height, image.channels(), image.step1());
        tess.Recognize(0);
        text  = tess.GetUTF8Text();
        return text;
    }


    Json::Value Receipt::interp_segment_text(std::string segment_text){
        std::string item;
        std::string price;
        std::string segment_text_ascii;
        std::string filtered_price;
        std::string::size_type pos=0; 
        std::string::size_type pos2=0; 

        //Remove utf-8 encoded characters which are not ascii
        for(int i = 0; i < segment_text.size(); i++){
            int utf8_val = (int)segment_text[i];
            if(utf8_val < 0){
                if( !(utf8_val & 0b01000000) ) continue;
                else segment_text_ascii.push_back('?'); //MSB of utf-8 char
                }
            else 
                segment_text_ascii.push_back(segment_text[i]);
            }

        std::regex date_exp("(([0-2][0-9])|(3[10]))[.,](0[1-9]|1[0-2])[.,]((2[0-1])?[0-9]{2})");
        std::smatch date_match;
        std::regex item_price_exp( "([^0-9A-z]?\\s*[0-9oO]+\\s?\\D\\s?[0-9oO\\s]+(\\s?\\D+)?)$");
        std::smatch item_price_match;
        if ((pos = segment_text_ascii.find("\n",0)) != std::string::npos ) // Only keep first line of image
            segment_text_ascii = segment_text_ascii.substr(0,pos);
        if (std::regex_search(segment_text_ascii,item_price_match,item_price_exp)){ // contains price
            pos = segment_text_ascii.find(item_price_match[1],0);
            item = segment_text_ascii.substr(0,pos);
            bool decimal_numbers = false;
            int decimal_precision = 2;
            price = segment_text_ascii.substr(pos,segment_text_ascii.npos);
            for(int i = 0; i < price.length(); i++){
                if(((int)price[i] > 47 && (int)price[i]<58) && (decimal_numbers == false || decimal_precision > 0)){
                    if(price[i] == 'o' || price[i] == 'O')
                        filtered_price.push_back('0');
                    if(decimal_numbers) decimal_precision--;
                    filtered_price.push_back(price[i]);    
                }
                else if(i==0 && price[i] != ' '){
                    filtered_price.push_back('-');    
                }
                else if( price[i]!=' ' && decimal_numbers == false){
                    filtered_price.push_back('.');
                    decimal_numbers = true;
                }
                else continue; 
                }
            price = filtered_price;
            }
        if(std::regex_search(segment_text_ascii,date_match,date_exp)){
            date = date_match[0];
            }
        else if(item != ""){ // text contains price and item
            Json::Value root;
            root["price"] = ::atof(price.c_str());
            root["item"] = item;
            return root;
        }

        return 0;
    }


    void Receipt::build_receipt(){
        Json::Value itemlist;
        itemlist.append(Json::Value::null);
        itemlist.clear();
        std::regex multip_exp("^(([0-9]){1,3}\\s?\\D)$");
        std::string total = "SUMME"; 
        std::string is_total_str = "^";
        for(int i = 0; i<=total.size(); i++ ){
            is_total_str.push_back('('); 
            for(int j = 0; j<total.size(); j++){
               is_total_str.push_back('['); 
               if(i==j) is_total_str.push_back('^');
               is_total_str.push_back(total[j]); 
               is_total_str.push_back(']'); 
               is_total_str.push_back('\\'); 
               is_total_str.push_back('s'); 
               is_total_str.push_back('*'); 
            }
            is_total_str.push_back(')');
            is_total_str.push_back('|');
        }
        is_total_str.pop_back();
        is_total_str.push_back('.');
        is_total_str.push_back('*');
        std::regex is_total(is_total_str.c_str(),std::regex_constants::ECMAScript | std::regex_constants::icase);
        
        bool found_total;
        std::string amount = "1";
        for(int i = 0 ; i < items.size(); i++){
            std::string item = items[i]["item"].asString();
            double price = items[i]["price"].asDouble();
            if(std::regex_match(item,is_total)){ // Total/Sum found
                receipt_json["total"] = std::abs( price);
                found_total = true;
                break;
            }
            else if(std::regex_match(item, multip_exp)){
                std::smatch amount_match;
                std::regex_search(item, amount_match, std::regex("^[0-9]+"));
                amount = amount_match[0];
            }
            else{
                items[i]["amount"] = ::atof(amount.c_str());
                itemlist.append(items[i]);
                amount = "1";
            }
        }
        receipt_json["items"] = itemlist;
        receipt_json["positions"] = itemlist.size();
        if(date != "") receipt_json["date"] = date;
    }

    Json::Value Receipt::get_receipt_json(){
        return receipt_json;
    }

}
