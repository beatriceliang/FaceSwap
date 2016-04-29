/* Itrat Ahmed Akhter
 * CS 365
 * Project 5
 * idBagVideoHaar.cpp
 * April 19, 2016
 * Program that uses the HAAR cascade classifier to detect
 * Wendy the badger in a live stream video 
 */


#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include <math.h>
#include "opencv2/opencv.hpp"


/*Main function*/
int main(int argc, char *argv[]) {
	char *filename = "faces/org/img7.jpg";
	cv::Mat swapping;
	swapping = cv::imread(filename);
	cv::VideoCapture *capdev;
	cv::CascadeClassifier cascade;
	cv::CascadeClassifier cascade2;
	if(!cascade.load("faces/haarcascade_frontalface_alt.xml")){
		printf("Can't load cascade properly\n");
		return(1);
	}
	if(!cascade2.load("faces/haarcascade_frontalface_alt.xml")){
		printf("Can't load cascade properly\n");
		return(1);
	}
	std::vector<cv::Rect> objects;
	std::vector<cv::Rect> swapObjects;

	cascade2.detectMultiScale(swapping,swapObjects,1.1,1,0,cv::Size(30,30),cv::Size(swapping.size().width/2,swapping.size().height/2));

	capdev = new cv::VideoCapture(0);
	if( !capdev->isOpened() ) {
		printf("Unable to open video device\n");
		return(-1);
	}

	cv::namedWindow("Video", 1);

	for(;;) {
		char keyPressed = cv::waitKey(10);

		cv::Mat frame,gray;
		
		*capdev >> frame; // get a new frame from the camera, treat as a stream
		resize(frame, frame, cv::Size(640, 360), 0, 0, cv::INTER_CUBIC);
		cv::cvtColor(frame,gray,CV_BGR2GRAY);
		cascade.detectMultiScale(frame,objects,1.1,1,0,cv::Size(30,30),cv::Size(frame.size().width/2,frame.size().height/2));
		if(objects.size()==1){
			printf("found face\n");
			rectangle(frame,cv::Point(objects[0].x,objects[0].y),cv::Point(objects[0].x+objects[0].width,objects[0].y+objects[0].height),cv::Scalar(0,0,255));
		}
		cv::imshow("MyFace", frame);
		
		cv::Mat replacingFace;
		replacingFace = swapping(cv::Rect(swapObjects[0].x,swapObjects[0].y,swapObjects[0].width,swapObjects[0].height));
		resize(replacingFace,replacingFace,cv::Size(objects[0].width,objects[0].height),0,0,cv::INTER_CUBIC);
		replacingFace.copyTo(frame(objects[0]));

		cv::imshow("Beyonce", frame);

		//press q to quit
		if(keyPressed==113)
			break;
	}

 

	// get rid of the window
	cv::destroyWindow("test");
 
	// terminate the program
	printf("Terminating\n");
	delete capdev;

	return(0);
}
