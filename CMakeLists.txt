cmake_minimum_required(VERSION 2.8)
project(Receipt.json)
find_package(OpenCV REQUIRED)

include_directories(/usr/include/tesseract/)
include_directories(/usr/include/jsoncpp/)

add_library(Receipt receipt.cpp)
add_executable(example example.cpp)
target_link_libraries(Receipt ${OpenCV_LIBS})
target_link_libraries(example Receipt tesseract jsoncpp)  
