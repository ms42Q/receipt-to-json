# Receipt to JSON

## Description

Paper receipts you get from grocery stores are an insult to modern technology. 
If your local store does not provide an API via which you can access your spending history, you have to collect all your bills and
manually copy all positions to your budgeting app.

With "Receipt to JSON" you have to struggle no more. Just take a photo of your receipt, pour it into the receipt engine
and receive a dictionary containing all the information you need to process your spendings further.

## Usage

Receipt to json relies on OpenCV2, tesseract-ocr and jsoncpp. In case you use GNU/Linux, you may install those
dependencies via the package management. 

```
# apt-get isntall tesseract-ocr-all libjsoncpp-dev libopencv-dev opencv-data
# cmake ./
# make
# ./example "/path/to/your/receipt/image.jpg"
```


## Features

* Input images are improved for OCR. There is a high chance "sloppy" photos will work fine.
* Output format: JSON
* Output contains:
  * Total amount
  * Date of purchase
  * Positions (amount, price, name)
* Free and OpenSource

## About
This Project was started at the FH-Aachen - University of Applied Sciences, in line with the computer vision lecture of
Prof. Scholl in 2016/2017.
