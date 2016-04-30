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

 void replaceFace(cv::Mat frame, cv::Mat swapping, std::vector<cv::Rect> objects, std::vector<cv::Rect> swapObjects, float theta){
 	float thetaDegrees = (theta)*(180/M_PI);
 	printf("theta %f\n",thetaDegrees);
 	int i,j;
 	//replace the face in the image with the face in the video
	cv::Mat replacingFace;
	replacingFace = swapping(cv::Rect(swapObjects[0].x,swapObjects[0].y,swapObjects[0].width,swapObjects[0].height));
	
	//need to resize the face in image to face in video
	resize(replacingFace,replacingFace,cv::Size(objects[0].width,objects[0].height),0,0,cv::INTER_CUBIC);
	int faceHeight = objects[0].height;
	int faceWidth = objects[0].width;
	int centerx = faceWidth/2;
	int centery = faceHeight/2;

	//taken from stackoverflow
	//get rotation matrix for rotating image around its center
	cv::Mat rot = cv::getRotationMatrix2D(cv::Point(centerx,centery),thetaDegrees,1.0);
	//determine bounding rectangle
	cv::Rect bbox = cv::RotatedRect(cv::Point(centerx,centery),replacingFace.size(),thetaDegrees).boundingRect();
	//adjust transformation matrix
	rot.at<double>(0,2) += bbox.width/2.0 - centerx;
    rot.at<double>(1,2) += bbox.height/2.0 - centery;
    //finally rotate image
    cv::Mat rotatedFace;
    cv::warpAffine(replacingFace,rotatedFace,rot,bbox.size());
    resize(rotatedFace,rotatedFace,cv::Size(objects[0].width,objects[0].height),0,0,cv::INTER_CUBIC);

	//copy ellipse around face in image to the face in video
	cv::Mat mask = cv::Mat::zeros(faceHeight,faceWidth,CV_8UC1);
	cv::ellipse(mask,cv::Point(faceWidth/2,faceHeight/2),cv::Size(faceWidth/2,faceHeight/2),0,0,360,cv::Scalar(255),CV_FILLED,8,0);
	//rotate mask as well
	cv::Mat rotatedMask;
	cv::warpAffine(mask,rotatedMask,rot,bbox.size());
	resize(rotatedMask,rotatedMask,cv::Size(objects[0].width,objects[0].height),0,0,cv::INTER_CUBIC);

	//copy rotated face onto video with the help of rotated mask
	rotatedFace.copyTo(frame(objects[0]),rotatedMask);
 }

 /* Return the angle made by the line between center of two eyes
  * with the x axis
  */
 float findAngle(cv::Mat frame, std::vector<cv::Rect> eyeObjects){
 	int eye1x = eyeObjects[0].x;
	int eye1y = eyeObjects[0].y;
	int eye1Width = eyeObjects[0].width;
	int eye1Height = eyeObjects[0].height;
	int eye2x = eyeObjects[1].x;
	int eye2y = eyeObjects[1].y;
	int eye2Width = eyeObjects[1].width;
	int eye2Height = eyeObjects[1].height;
	int centerEye1x = eye1x + eye1Width/2;
	int centerEye1y = eye1y + eye1Height/2;
	int centerEye2x = eye2x + eye2Width/2;
	int centerEye2y = eye2y + eye2Height/2;
	float eyeVectorx = abs(centerEye2x - centerEye1x);
	float eyeVectory;
	if(centerEye1x<centerEye2x){
	 	eyeVectory = centerEye1y - centerEye2y;
	}
	else{
		eyeVectory = centerEye2y - centerEye1y;
	}
	//printf("eyeVectorx %f eyeVectory %f\n",eyeVectorx,eyeVectory);
	float theta = atan2(eyeVectory,eyeVectorx);

	rectangle(frame,cv::Point(eye1x,eye1y),cv::Point(eye1x+eye1Width,eye1y+eye1Height),cv::Scalar(0,0,255));
	line(frame,cv::Point(centerEye1x,centerEye1y),cv::Point(centerEye2x,centerEye2y),cv::Scalar(0,0,255));
 	return theta;
 }


/*Main function*/
int main(int argc, char *argv[]) {
	int i;
	char *filename = "faces/org/img7.jpg";
	cv::Mat swapping;
	swapping = cv::imread(filename);

	cv::VideoCapture *capdev;
	cv::CascadeClassifier cascade; //for video
	cv::CascadeClassifier cascade2; //for image
	cv::CascadeClassifier eyeCascade; //for eye

	if(!cascade.load("faces/haarcascade_frontalface_alt.xml")){
		printf("Can't load cascade1 properly\n");
		return(1);
	}
	if(!cascade2.load("faces/haarcascade_frontalface_alt.xml")){
		printf("Can't load cascade2 properly\n");
		return(1);
	}
	if(!eyeCascade.load("faces/haarcascade_eye.xml")){
		printf("Can't load  eye cascade properly\n");
		return(1);
	}

	std::vector<cv::Rect> objects; //for video 
	std::vector<cv::Rect> swapObjects; //for image
	std::vector<cv::Rect> eyeObjects; //for eyes

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

		cv::Mat frame,frame2;
		
		*capdev >> frame; // get a new frame from the camera, treat as a stream
		*capdev >> frame2;
		resize(frame, frame, cv::Size(640, 360), 0, 0, cv::INTER_CUBIC);
		resize(frame2, frame2, cv::Size(640, 360), 0, 0, cv::INTER_CUBIC);

		//detect face in the video
		cascade.detectMultiScale(frame,objects,1.1,1,0,cv::Size(100,100),cv::Size(frame.size().width/2,frame.size().height/2));
		//detect eyes in the video
		eyeCascade.detectMultiScale(frame,eyeObjects,1.1,1,0,cv::Size(30,30),cv::Size(frame.size().width/2,frame.size().height/2));
		float theta =0;
		if(objects.size()==1){
			//printf("found face\n");
			rectangle(frame2,cv::Point(objects[0].x,objects[0].y),cv::Point(objects[0].x+objects[0].width,objects[0].y+objects[0].height),cv::Scalar(0,0,255));
		}

		//find angle of face with the help of eyes
		if(eyeObjects.size()==2){
			printf("found eyes\n");
			theta = findAngle(frame2,eyeObjects);
		}

		//replace face in video with face in image
		cv::imshow("MyFace", frame2);
		if(objects.size()==1 && swapObjects.size()==1){
			replaceFace(frame,swapping,objects,swapObjects,theta);
		}
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
