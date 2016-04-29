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

	//detect face in the image
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
		//detect face in the video
		cascade.detectMultiScale(frame,objects,1.1,1,0,cv::Size(30,30),cv::Size(frame.size().width/2,frame.size().height/2));
		if(objects.size()==1){
			printf("found face\n");
			//rectangle(frame,cv::Point(objects[0].x,objects[0].y),cv::Point(objects[0].x+objects[0].width,objects[0].y+objects[0].height),cv::Scalar(0,0,255));
		}
		cv::imshow("MyFace", frame);

		//replace the face in the image with the face in the video
		cv::Mat replacingFace;
		replacingFace = swapping(cv::Rect(swapObjects[0].x,swapObjects[0].y,swapObjects[0].width,swapObjects[0].height));
		
		//need to resize the face in image to face in video
		resize(replacingFace,replacingFace,cv::Size(objects[0].width,objects[0].height),0,0,cv::INTER_CUBIC);
		int faceHeight = objects[0].height;
		int faceWidth = objects[0].width;

		//copy ellipse around face in image to the face in video
		cv::Mat mask = cv::Mat::zeros(faceHeight,faceWidth,CV_8UC1);
		cv::ellipse(mask,cv::Point(faceWidth/2,faceHeight/2),cv::Size(faceWidth/2,faceHeight/2),0,0,360,cv::Scalar(255),CV_FILLED,8,0);
		replacingFace.copyTo(frame(objects[0]),mask);

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
