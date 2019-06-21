// KinectSekeletonOpenGL.cpp : Definiert den Einstiegspunkt für die Konsolenanwendung.
//



/*
_________________________________________________________________________________________________________________________________
|																																|
|									 Cologne University of Applied Sciences- Campus Gummersbach							        |
|																																|
|																																|
|											        3D Skeleton Visualisation									                |
|													using OpenGL, OpenNI & NiTE											        |
|																																|
|												Orginated for Vertiefung Programmieren											|
|												Lecturer: Prof. Dr. Elena Algorri									            |
|																																|
|																																|
|													Created on January 2015											            |
|																																|
|										Author: Christian Breiderhoff, MatrNo: 11090370								            |
|_______________________________________________________________________________________________________________________________|
*/


#include "stdafx.h"
#include <iostream>
#include <string>
#include <Windows.h>
#include "Dependencies\glew\glew.h"
#include "Dependencies\freeglut\freeglut.h"
#include <gl\GL.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <OpenNI.h>
#include <NiTE.h>


using namespace cv;
using namespace std;
using namespace openni;
using namespace nite;


nite::Point3f one;
nite::Point3f two;


#define MAX_USERS 1
#define CHANNEL 3
#define M_PI           3.14159265358979323846  
const string windowName = "Camera";

const string windowName2 = "Thresholded Image";

const string trackbarWindowName = "HSV ColorPicker";
const int font = (int)GLUT_BITMAP_TIMES_ROMAN_24;
const int font1 = (int)GLUT_BITMAP_9_BY_15;
const int font2 = (int)GLUT_BITMAP_HELVETICA_18;
//for Camaera
float _angle = 0.0f;
float _cameraAngle = 0.0f;

float lx = 0.0f, lz = -5.0f;
float x = 2.0f, z = 2.5f;

float angle = 0.0f;



//Userdata
bool g_visibleUsers[MAX_USERS] = { false };
nite::SkeletonState g_skeletonStates[MAX_USERS] = { nite::SKELETON_NONE };
double winkel_degree;
double winkel2_degree;

//for Objecttracking
float posX = 0; 
 float posY = 0;
 Point2f Object = { 0, 0 };
 Mat cameraFeed(cv::Size(640, 480), CV_8UC3, NULL);
Mat HSV(cv::Size(640, 480), CV_8UC3, NULL);
Mat thresholdMat(cv::Size(640, 480), CV_8UC1, NULL);
cv::Mat depthcv1(cv::Size(640, 480), CV_16UC1, NULL);
IplImage*  Contours = cvCreateImage(cvSize(640, 480), IPL_DEPTH_8U, 1);;
IplImage*  drawingIpl = cvCreateImage(cvSize(640, 480), IPL_DEPTH_8U, 1);;

static DepthPixel* depth1;
int H_MIN = 0;
int H_MAX = 256;
int S_MIN = 0;
int S_MAX = 256;
int V_MIN = 0;
int V_MAX =256;



const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;
bool objectTracked = false;
bool enableObjectTracking = true;
bool stateoftracking = true;
bool contact = false;
bool contact_right=false;
bool contact_left=false;
bool drink = false;


//static float objectX, objectY, objectZ;

//for Init
Device device;
openni::VideoStream depth, color;
openni::VideoStream** stream;
openni::VideoFrameRef depthFrame, colorFrame;
 nite::UserTracker userTracker;
nite::Status niteRc;
nite::UserTrackerFrameRef userTrackerFrame;
openni::Status Init(int argc, char** argv);
openni::Status InitOpenGL(int argc, char** argv);


//Skeletonjoints
static float coordinates_head[2];
static float coordinates_limb1[2];
static float coordinates_limb2[2];
static float coordinates_joint[2];
static float usercenter[1];

//for HUD
char buffer2[50];
char buffer1[50];
bool rAngle = false;
bool lAngle = false;
bool rDrinking = false;
bool lDrinking = false;
bool rContact = false;
bool lContact = false;



//Functions
void Run();
void MainLoop();
void DrawLimb( const nite::SkeletonJoint &joint1, const nite::SkeletonJoint &joint2);
void DrawJoint(const nite::SkeletonJoint &joint);
void DrawHead(const nite::SkeletonJoint &head);
void DrawButton();
void DrawButton( const float y);
void changeSize(int w, int h);
void handleKeypress(int key, int x, int y);
void processNormalKeys(unsigned char key, int x, int y);
void myIdle();
void ShutDown();
void Angle_and_contact(const nite::SkeletonJoint &RShoulder, const nite::SkeletonJoint &RElbow, 
	const nite::SkeletonJoint &RHand, const nite::SkeletonJoint &LShoulder,
	const nite::SkeletonJoint &LElbow, const nite::SkeletonJoint &LHand, const nite::SkeletonJoint &Head);
void Objecttracking();
void createTrackbars();
void on_trackbar(int, void*);
void drawObject(int x, int y, Mat &frame);
//void morphOps(Mat &thresh);
void renderBitmapString(float x, float y, void *font, char *string, int color, int size);

int i = 0;

int main(int argc, char** argv)
{
	std::cout << "**********Cologne University of Applied Sciences - Campus Gummersbach*****" << endl;
	std::cout << "************3D Skeleton Visualisation using OpenGL, OpenNI & NiTE*********" << endl;	
	std::cout << "************************Created on January 2014**************************" << endl;
	std::cout << "*********************Author: Christian Breiderhoff************************" << endl;
	std::cout << "\n \n" << endl;


	openni::Status rc = openni::STATUS_OK;


	rc = Init(argc, argv);
	if (rc != openni::STATUS_OK)
	{
		while (1)
		{
			int key = waitKey(30);
			if (key == 27);
			{
				ShutDown();
			}

		}
		return 1;
	}
	
		Run();
		return 0;
}

void ShutDown()

{
	
	//&userTracker.destroy;
	depth.stop();
	depth.destroy();
	color.stop();
	color.destroy();
	device.close();
	nite::NiTE::shutdown();
	openni::OpenNI::shutdown();
	exit(0);
}

void Run()
{
	glutMainLoop();
}

void myIdle()
{
	glutPostRedisplay();
}

openni::Status Init(int argc, char** argv)
{
	NiTE::initialize();
	OpenNI::initialize();

	std::cout << "Kinect initialization..." << endl;
	char buffer3[50];
	char buffer4[50];
	sprintf_s(buffer3, "OpenNi Version %d.%d.%d.%d  ", openni::OpenNI::getVersion().major, openni::OpenNI::getVersion().minor, 
		openni::OpenNI::getVersion().maintenance, openni::OpenNI::getVersion().build);
	sprintf_s(buffer4, "NiTE Version %d.%d.%d.%d  ", nite::NiTE::getVersion().major, nite::NiTE::getVersion().minor,
		nite::NiTE::getVersion().maintenance, nite::NiTE::getVersion().build);
	

	std::cout << buffer3 << endl;
	std::cout << buffer4 << endl;
	
	if (device.open(openni::ANY_DEVICE) != 0)
	{
		std::cout << "Kinect not found !" << endl;
		openni::OpenNI::getExtendedError();
		std::cout << "Press ESC to exit" << endl;
		
	}

	device.setImageRegistrationMode(IMAGE_REGISTRATION_DEPTH_TO_COLOR);
	device.setDepthColorSyncEnabled(true);

	std::cout << "Kinect opened" << endl;
	
	color.create(device, SENSOR_COLOR);
	color.start();
	std::cout << "Camera ok" << endl;
	depth.create(device, SENSOR_DEPTH);
	depth.start();
	std::cout << "Depth sensor ok" << endl;
	openni::VideoMode video;
	video.setResolution(640, 480);
	video.setFps(30);
	video.setPixelFormat(PIXEL_FORMAT_DEPTH_100_UM);
	depth.setVideoMode(video);
	video.setPixelFormat(PIXEL_FORMAT_RGB888);
	color.setVideoMode(video);


	//device.setDepthColorSyncEnabled(true);

	
	niteRc = userTracker.create(&device);

	if (niteRc != nite::STATUS_OK)
	{
		printf("Couldn't create user tracker\n");


	}

	
	return InitOpenGL( argc, argv);
	
	
}




openni::Status InitOpenGL(int argc, char** argv)
{
	
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(1280, 680); 

	cvNamedWindow("Camera", CV_WINDOW_AUTOSIZE);
	cvMoveWindow("Camera", 100, 100);
	cvNamedWindow("Thresholded Image", CV_WINDOW_AUTOSIZE);
	createTrackbars();
	glutCreateWindow("Kinect 3DSkeleton");
	glEnable(GL_DEPTH_TEST);

	
	glutReshapeFunc(changeSize);
	glutKeyboardFunc(processNormalKeys);
	glutSpecialFunc(handleKeypress);
	
	glutDisplayFunc(MainLoop);
	glutIdleFunc(myIdle);

	return openni::STATUS_OK;
	

}

void processNormalKeys(unsigned char key, int x, int y)
{

	if (key == 27)
		ShutDown();
}

void handleKeypress(int key, int x, int y)
{


	float fraction = 0.1f;
	switch (key) {
	case 27:
		exit(0);

	case GLUT_KEY_LEFT:
		angle -= 0.01f;
		lx = sin(angle);
		lz = -cos(angle);
		break;
	case GLUT_KEY_RIGHT:
		angle += 0.01f;
		lx = sin(angle);
		lz = -cos(angle);
		break;
	case GLUT_KEY_UP:
		x += lx * fraction;
		z += lz * fraction;
		break;
	case GLUT_KEY_DOWN:
		x -= lx * fraction;
		z -= lz * fraction;
		break;

		//glutPostRedisplay();
	}
}




void changeSize(int w, int h)
{
	if (h == 0)
		h = 1;
	float ratio = w * 1.0 / h;
	
	glMatrixMode(GL_PROJECTION);	
	glLoadIdentity();
	glViewport(1, 1, w, h);
	gluPerspective(60.0f, ratio, 0.1f, 100.0f);
	glMatrixMode(GL_MODELVIEW);
}


void MainLoop()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(1, 1, 1, 0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();


	//camera
	gluLookAt(x, -2.0f, z,
		x + lx, -2.0f, z + lz,
		0.0f, 1.0f, 0.0f);
	
	color.readFrame(&colorFrame);
	depth.readFrame(&depthFrame);
	cameraFeed.data = (uchar*)colorFrame.getData();
	
	cvtColor(cameraFeed, cameraFeed, CV_BGR2RGB);
	cvtColor(cameraFeed, HSV, COLOR_BGR2HSV);
	inRange(HSV, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), thresholdMat);
	imshow("RGB", cameraFeed);
	imshow("HSV", HSV);
	Objecttracking();

	


	

	CvScalar Color1;
	cv::Vec3b Color2;
	string status;
	if (enableObjectTracking)
	{
		status = "Disable";
		Color1 = cvScalar(0, 0, 255);
		 Color2 = cv::Vec3b(0, 255, 0);

	}
	else
	{
		status = "Enable";
		Color1 = cvScalar(0, 255, 0);
		Color2 = cv::Vec3b(0, 0, 255);
	}
	for (int y = 0; y < 100; y++)
	{
		for (int x = 540; x < 640; x++)
		{
			cameraFeed.at<Vec3b>(y ,x) = Color2;
		}
	}

	for(int y = 0; y < 100; y++)
	{
		for (int x = 0; x < 100; x++)
		{
			cameraFeed.at<Vec3b>(y, x) = { 0, 0, 255 };
		}
	}
	putText(cameraFeed, status, Point(550, 20),FONT_HERSHEY_SIMPLEX, 0.5, Color1, 1.5);
	putText(cameraFeed, "Object", Point(550, 45), FONT_HERSHEY_SIMPLEX, 0.5, Color1, 1.5);
	putText(cameraFeed, "Tracking", Point(550, 60), FONT_HERSHEY_SIMPLEX, 0.5, Color1, 1.5);
	
	putText(cameraFeed, "Reset", Point(10, 40), FONT_HERSHEY_SIMPLEX, 0.8, cvScalar(0, 255, 0), 1.5);

	depth1 = (DepthPixel*)depthFrame.getData();
	


	//HUD
	
	//renderBitmapString(350, 520, (void *)font, "ObjectTracking:", 4, 1);
	if (enableObjectTracking)
	{
		renderBitmapString(550, 80, (void *)font, "ObjectTracking Enabled", 3, 1);
		if (objectTracked)
		{
			renderBitmapString(550, 110, (void *)font, "Object Tracked", 3, 1);
		}
		else
		{
			renderBitmapString(550, 110, (void *)font, "No Object Tracked", 1, 1);
		}
	}
	else
	{
		renderBitmapString(550, 80, (void *)font, "ObjectTracking Disabled", 1, 1);
	}
	if (drink)
	{

		renderBitmapString(550, 200, (void *)font, " Intake detected", 3, 1);
	}
	
	renderBitmapString(800, 20, (void *)font1, "Cologne University of Applied Sciences - Campus Gummersbach", 4, 2);
	//renderBitmapString(800, 40, (void *)font1, "Kinect 3D Motion-, Pose- & ObjectCapturing", 5);
	
	

	//renderBitmapString(350, 400, (void *)font, "Arm Angles:", 4);
	

	if (lAngle)
	{
		renderBitmapString(70, 430, (void *)font2, buffer2, 2, 1);
	}
	if (rAngle)
	{
		renderBitmapString(70, 460, (void *)font2, buffer1, 2, 1);
	}


	//renderBitmapString(70,400, (void *)font, "Hand Position:", 4);
	if (!lContact)
	{
		renderBitmapString(70, 490, (void *)font2, "Pose Left Arm not detected", 1, 1);
	}
	else
	{
		renderBitmapString(70, 490, (void *)font2, "Pose Left Arm detected", 3, 1);
	}
	if (!rContact)
	{
		renderBitmapString(70, 520, (void *)font2, "Pose Right Arm not detected", 1, 1);
	}
	else
	{
		renderBitmapString(70, 520, (void *)font2, "Pose Right Arm detected", 3, 1);
	}

	//renderBitmapString(70, 520, (void *)font, "Object-Hand Contact:", 4);
	if (contact_right)
	{
		renderBitmapString(70, 550, (void *)font2, "Right Hand Object have Contact", 3, 1);
	}
	else
	{
		
		renderBitmapString(70, 550, (void *)font2, "Right Hand Object have no Contact", 1, 1);
	}
	if (contact_left)
	{
		renderBitmapString(70, 580, (void *)font2, "Left Hand Object have Contact", 3, 1);
	}
	else
	{
		renderBitmapString(70, 580, (void *)font2, "Left Hand Object have no Contact", 1, 1);
		
	}
	if (lDrinking)
	{
		renderBitmapString(70, 610, (void *)font2, "Left Intake", 3, 1);
	}
	if (rDrinking)
	{
		renderBitmapString(70, 640, (void *)font2, " Right Intake", 3, 1);
	}

	nite::Status niteRc1;
	niteRc1 = userTracker.readFrame(&userTrackerFrame);
	if (niteRc1 != nite::STATUS_OK)
	{
		printf("Couldn't read Frame");
	}


	nite::Point3f Floorpoint;
	Floorpoint.y = userTrackerFrame.getFloor().point.y;

	if (userTrackerFrame.getFloorConfidence()> 0.8f)
	{
		DrawButton(Floorpoint.y);
	}
	else
	{
		DrawButton();
	}



	const nite::Array<nite::UserData>& users = userTrackerFrame.getUsers();
	/*if(users.getSize() ==0)
	{
		renderBitmapString(150, 80, (void *)font, "No User ", 1);
	}*/
	

	for (int i = 0; i < users.getSize(); ++i)
	{
		const nite::UserData& user = users[i];



		if (user.isNew())
			std::cout << "New" << endl;
		else if (user.isVisible() && !g_visibleUsers[user.getId()])
			renderBitmapString(150, 80, (void *)font, "User Visible", 3, 1);
		else if (!user.isVisible() && g_visibleUsers[user.getId()])
		{

			renderBitmapString(150, 80, (void *)font, "No User ", 1, 1);

		}
		else if (user.isLost())
			std::cout << "Lost" << endl;

		g_visibleUsers[user.getId()] = user.isVisible();


		if (g_skeletonStates[user.getId()] != user.getSkeleton().getState())
		{
			switch (g_skeletonStates[user.getId()] = user.getSkeleton().getState())
			{
			case nite::SKELETON_NONE:
				std::cout << "Stopped tracking." << endl;
				
				break;
			case nite::SKELETON_CALIBRATING:
				std::cout << "Calibrating..." << endl;
				
				break;
			case nite::SKELETON_TRACKED:
				std::cout << "Tracking!" << endl;
				
				break;
			case nite::SKELETON_CALIBRATION_ERROR_NOT_IN_POSE:
			case nite::SKELETON_CALIBRATION_ERROR_HANDS:
			case nite::SKELETON_CALIBRATION_ERROR_LEGS:
			case nite::SKELETON_CALIBRATION_ERROR_HEAD:
			case nite::SKELETON_CALIBRATION_ERROR_TORSO:
				std::cout << "Calibration Failed... " << endl;
				
				break;
			}
		}





		//Skeletonstream
		if (user.isNew())
		{
			userTracker.startSkeletonTracking(user.getId());
		}
		else if (user.getSkeleton().getState() == nite::SKELETON_TRACKED)
		{
			renderBitmapString(600, 50, (void *)font, "User Visible", 3, 1);
			if (objectTracked)
			{
				cv::Point2f RightHand1;
				userTracker.convertJointCoordinatesToDepth(user.getSkeleton().getJoint(nite::JOINT_RIGHT_HAND).getPosition().x,
					user.getSkeleton().getJoint(nite::JOINT_RIGHT_HAND).getPosition().y,
					user.getSkeleton().getJoint(nite::JOINT_RIGHT_HAND).getPosition().z, &RightHand1.x, &RightHand1.y);
				cv::Point2f LeftHand1;
				userTracker.convertJointCoordinatesToDepth(user.getSkeleton().getJoint(nite::JOINT_LEFT_HAND).getPosition().x,
					user.getSkeleton().getJoint(nite::JOINT_LEFT_HAND).getPosition().y,
					user.getSkeleton().getJoint(nite::JOINT_LEFT_HAND).getPosition().z, &LeftHand1.x, &LeftHand1.y);

				cv::Vec2f Object_to_righthand((RightHand1.x -Object.x), (RightHand1.y - Object.y));
				cv::Vec2f Object_to_lefthand((LeftHand1.x - Object.x), (LeftHand1.y - Object.y));

				double distance_Object_right_Hand = sqrt(pow(Object_to_righthand[0], 2) + pow(Object_to_righthand[1], 2));
				double distance_Object_left_Hand = sqrt(pow(Object_to_lefthand[0], 2) + pow(Object_to_lefthand[1], 2));
				
				if (distance_Object_right_Hand < 70.0f)
				{

					contact_right = true;

				}
				else
				{
					contact_right = false;

				}

				if (distance_Object_left_Hand < 70.0f)
				{

					contact_left = true;

				}

				else
				{
					contact_left = false;

				}



			}
			else
			{
				contact_right = false;
				contact_left = false;


			}
	

			DrawLimb(user.getSkeleton().getJoint(nite::JOINT_HEAD), user.getSkeleton().getJoint(nite::JOINT_NECK));

			DrawLimb(user.getSkeleton().getJoint(nite::JOINT_LEFT_SHOULDER), user.getSkeleton().getJoint(nite::JOINT_LEFT_ELBOW));
			DrawLimb(user.getSkeleton().getJoint(nite::JOINT_LEFT_ELBOW), user.getSkeleton().getJoint(nite::JOINT_LEFT_HAND));

			DrawLimb(user.getSkeleton().getJoint(nite::JOINT_RIGHT_SHOULDER), user.getSkeleton().getJoint(nite::JOINT_RIGHT_ELBOW));
			DrawLimb(user.getSkeleton().getJoint(nite::JOINT_RIGHT_ELBOW), user.getSkeleton().getJoint(nite::JOINT_RIGHT_HAND));

			DrawLimb(user.getSkeleton().getJoint(nite::JOINT_LEFT_SHOULDER), user.getSkeleton().getJoint(nite::JOINT_RIGHT_SHOULDER));

			DrawLimb(user.getSkeleton().getJoint(nite::JOINT_LEFT_SHOULDER), user.getSkeleton().getJoint(nite::JOINT_TORSO));
			DrawLimb(user.getSkeleton().getJoint(nite::JOINT_RIGHT_SHOULDER), user.getSkeleton().getJoint(nite::JOINT_TORSO));

			DrawLimb(user.getSkeleton().getJoint(nite::JOINT_TORSO), user.getSkeleton().getJoint(nite::JOINT_LEFT_HIP));
			DrawLimb(user.getSkeleton().getJoint(nite::JOINT_TORSO), user.getSkeleton().getJoint(nite::JOINT_RIGHT_HIP));

			DrawLimb(user.getSkeleton().getJoint(nite::JOINT_LEFT_HIP), user.getSkeleton().getJoint(nite::JOINT_RIGHT_HIP));


			DrawLimb(user.getSkeleton().getJoint(nite::JOINT_LEFT_HIP), user.getSkeleton().getJoint(nite::JOINT_LEFT_KNEE));
			DrawLimb(user.getSkeleton().getJoint(nite::JOINT_LEFT_KNEE), user.getSkeleton().getJoint(nite::JOINT_LEFT_FOOT));

			DrawLimb(user.getSkeleton().getJoint(nite::JOINT_RIGHT_HIP), user.getSkeleton().getJoint(nite::JOINT_RIGHT_KNEE));
			DrawLimb(user.getSkeleton().getJoint(nite::JOINT_RIGHT_KNEE), user.getSkeleton().getJoint(nite::JOINT_RIGHT_FOOT));



			DrawJoint(user.getSkeleton().getJoint(nite::JOINT_NECK));
			DrawJoint(user.getSkeleton().getJoint(nite::JOINT_LEFT_SHOULDER));
			DrawJoint(user.getSkeleton().getJoint(nite::JOINT_RIGHT_SHOULDER));
			DrawJoint(user.getSkeleton().getJoint(nite::JOINT_LEFT_ELBOW));
			DrawJoint(user.getSkeleton().getJoint(nite::JOINT_RIGHT_ELBOW));
			DrawJoint(user.getSkeleton().getJoint(nite::JOINT_TORSO));
			DrawJoint(user.getSkeleton().getJoint(nite::JOINT_LEFT_HAND));
			DrawJoint(user.getSkeleton().getJoint(nite::JOINT_RIGHT_HAND));
			DrawJoint(user.getSkeleton().getJoint(nite::JOINT_LEFT_HIP));
			DrawJoint(user.getSkeleton().getJoint(nite::JOINT_RIGHT_HIP));
			DrawJoint(user.getSkeleton().getJoint(nite::JOINT_LEFT_KNEE));
			DrawJoint(user.getSkeleton().getJoint(nite::JOINT_RIGHT_KNEE));
			DrawJoint(user.getSkeleton().getJoint(nite::JOINT_LEFT_FOOT));
			DrawJoint(user.getSkeleton().getJoint(nite::JOINT_RIGHT_FOOT));

			DrawHead(user.getSkeleton().getJoint(nite::JOINT_HEAD));

			Angle_and_contact(user.getSkeleton().getJoint(nite::JOINT_RIGHT_SHOULDER), user.getSkeleton().getJoint(nite::JOINT_RIGHT_ELBOW),
				user.getSkeleton().getJoint(nite::JOINT_RIGHT_HAND), user.getSkeleton().getJoint(nite::JOINT_LEFT_SHOULDER),
				user.getSkeleton().getJoint(nite::JOINT_LEFT_ELBOW),
				user.getSkeleton().getJoint(nite::JOINT_LEFT_HAND), user.getSkeleton().getJoint(nite::JOINT_HEAD));

			userTracker.convertJointCoordinatesToDepth(user.getCenterOfMass().x, user.getCenterOfMass().y, user.getCenterOfMass().z, &usercenter[0], &usercenter[1]);
			cv::circle(cameraFeed, cv::Point2f(usercenter[0], usercenter[1]), 20, CV_RGB(255, 0, 0), 3, 8, 0);
			putText(cameraFeed, "User detected", cv::Point2f(usercenter[0], usercenter[1] + 20), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 1, 1);

		cv:Point2f RightHand;
			userTracker.convertJointCoordinatesToDepth(user.getSkeleton().getJoint(nite::JOINT_RIGHT_HAND).getPosition().x,
				user.getSkeleton().getJoint(nite::JOINT_RIGHT_HAND).getPosition().y,
				user.getSkeleton().getJoint(nite::JOINT_RIGHT_HAND).getPosition().z, &RightHand.x, &RightHand.y);
			cv::Point2f LeftHand;
			userTracker.convertJointCoordinatesToDepth(user.getSkeleton().getJoint(nite::JOINT_LEFT_HAND).getPosition().x,
				user.getSkeleton().getJoint(nite::JOINT_LEFT_HAND).getPosition().y,
				user.getSkeleton().getJoint(nite::JOINT_LEFT_HAND).getPosition().z, &LeftHand.x, &LeftHand.y);

			if (RightHand.x >= 540 && RightHand.x <= 640)
			{
				if (RightHand.y >= 0 && RightHand.y <= 100)
				{
					if (stateoftracking)
					{
						stateoftracking = false;
						enableObjectTracking = !enableObjectTracking;
					}
				}
			}
			else
			{
				if (!stateoftracking)
				{

					stateoftracking = true;
				}
			}

			if (LeftHand.x >= 0 && LeftHand.x <= 100)
			{
				if (LeftHand.y >= 0 && LeftHand.y <= 100)
				{

					drink = false;
					//enableObjectTracking = false;

				}
			}
			}

	}
	cv::imshow(windowName, cameraFeed); 
	cv::imshow(windowName2, thresholdMat);
	cvShowImage("Contours", drawingIpl);

	glutSwapBuffers();
	
		
}


	void DrawLimb( const nite::SkeletonJoint &joint1, const nite::SkeletonJoint &joint2)
	{	
		if (joint1.getPositionConfidence() > .1f && joint2.getPositionConfidence() > .1f)
		{
			cv::Point3f limb1;
			limb1.x = -joint1.getPosition().x / 100+12;
			limb1.y = joint1.getPosition().y / 100;
			limb1.z = ((joint1.getPosition().z / 100)* (-1));
			cv::Point3f limb2;
			limb2.x = -joint2.getPosition().x / 100+12;
			limb2.y = joint2.getPosition().y / 100;
			limb2.z = -joint2.getPosition().z / 100;
			
			
			
			glColor3f(0, 0, 1);
			if ((joint1.getType() == nite::JointType::JOINT_LEFT_SHOULDER && joint2.getType() == nite::JointType::JOINT_LEFT_ELBOW) 
				| (joint1.getType() == nite::JointType::JOINT_LEFT_ELBOW && joint2.getType() == nite::JointType::JOINT_LEFT_HAND) && (winkel2_degree >= 23.0 && winkel2_degree <= 50.0))
			{
				glColor3f(0.0, 1.0, 0.0);
			}
			if ((joint1.getType() == nite::JointType::JOINT_RIGHT_SHOULDER && joint2.getType() == nite::JointType::JOINT_RIGHT_ELBOW)
				| (joint1.getType() == nite::JointType::JOINT_RIGHT_ELBOW && joint2.getType() == nite::JointType::JOINT_RIGHT_HAND) && (winkel_degree >= 23.0 && winkel_degree <= 50.0))
			{
				glColor3f(0.0, 1.0, 0.0);
			}
			glLineWidth(20);
			glPushMatrix();
			glBegin(GL_LINES);
			glTranslatef(0.0f, 0.0f, 0.0f);
			glVertex3f(limb1.x, limb1.y, limb1.z);
			
			//glPushMatrix();
			//glTranslatef(0.0f, 0.0f, 0.0f);
			glVertex3f(limb2.x, limb2.y, limb2.z);
			
			
			glEnd();
			glPopMatrix();
			
		}
	}


	void DrawJoint(const nite::SkeletonJoint &joint)
	{
		if (joint.getPositionConfidence() > .1f)
		{
			coordinates_joint[0] = -joint.getPosition().x / 100 + 12;
			coordinates_joint[1] = joint.getPosition().y / 100 ;
			coordinates_joint[2] = -joint.getPosition().z / 100;
			//string postion = to_string(coordinates_joint[2] );
			//cout << postion + "\n \n" << endl;
		}
			glColor3f(0.7, 0.3, 0.9);
		
			if ( joint.getType() == nite::JointType::JOINT_LEFT_HAND && contact_left)
			{
				glColor3f(0.0, 1.0, 0.0);
			}
			if (joint.getType() == nite::JointType::JOINT_RIGHT_HAND && contact_right)
			{
				glColor3f(0.0, 1.0, 0.0);
			}
			glPushMatrix();
			glTranslatef(coordinates_joint[0], coordinates_joint[1], coordinates_joint[2]);
			glutSolidSphere(0.7f, 20, 20);
			glPopMatrix();
	}


	void DrawHead(const nite::SkeletonJoint &head)
	{
		if (head.getPositionConfidence() > .1f)
		{
			coordinates_head[0] = -head.getPosition().x / 100 +12;
			coordinates_head[1] = head.getPosition().y / 100;
			coordinates_head[2] = -head.getPosition().z / 100;



			glColor3f(1.0, 0.5, 0.0);
			glLineWidth(2);
			glPushMatrix();
			glTranslatef(coordinates_head[0], coordinates_head[1], coordinates_head[2]);
			glutWireSphere(1.5f, 20, 20);
			glPopMatrix();
		}
	}

	void DrawButton()
	{
		//Draw Botton
		glColor3f(0.0f, 0.0f, 0.0f);
		float w1 = 0;
		float w2 = 0;
		glLineWidth(1);
		for (w1 = 2.0; w1 < 40.0; w1 += 1.)
		{
			glColor3f(0, 0, 0);
			glBegin(GL_LINES);
			glVertex3f(w1, -15, 0.0);
			glVertex3f(w1, -15, -50);

			glEnd();
		}
		for (w2 = 0.0; w2 > -50; w2 -= 1)
		{
			glColor3f(0, 0, 0);
			glBegin(GL_LINES);
			glVertex3f(2, -15, w2);
			glVertex3f(40, -15, w2);

			glEnd();		
		}
	}

	void DrawButton( const float y)
	{
		//Draw Botton
		glColor3f(0.0f, 0.0f, 0.0f);
		float yfloor =  y / 100;
		float w1 = 0;
		float w2 = 0;
		glLineWidth(1);
		for (w1 = 2.0; w1 < 40.0; w1 += 1.)
		{
			glColor3f(0, 0, 0);
			glBegin(GL_LINES);
			glVertex3f(w1, yfloor, 0.0);
			glVertex3f(w1, yfloor, -50);

			glEnd();
		}
		for (w2 = 0.0; w2 > -50; w2 -= 1)
		{
			glColor3f(0, 0, 0);
			glBegin(GL_LINES);
			glVertex3f(2, yfloor, w2);
			glVertex3f(40, yfloor, w2);

			glEnd();
		}
	}

	void Angle_and_contact(const nite::SkeletonJoint &RShoulder, const nite::SkeletonJoint &RElbow, const nite::SkeletonJoint &RHand, 
		const nite::SkeletonJoint &LShoulder,
		const nite::SkeletonJoint &LElbow, const nite::SkeletonJoint &LHand, const nite::SkeletonJoint &Head)
	{
		cv::Point3d head(Head.getPosition().x, Head.getPosition().y, Head.getPosition().z);

		//right
		cv::Point3d r_shoulder(RShoulder.getPosition().x, RShoulder.getPosition().y, RShoulder.getPosition().z);
		cv::Point3d r_elbow(RElbow.getPosition().x, RElbow.getPosition().y, RElbow.getPosition().z);
		cv::Point3d r_hand(RHand.getPosition().x, RHand.getPosition().y, RHand.getPosition().z);
		

		cv::Vec3d r_Ober(r_elbow.x - r_shoulder.x, r_elbow.y - r_shoulder.y, r_elbow.z - r_shoulder.z);
		cv::Vec3d r_Unter(r_hand.x - r_elbow.x, r_hand.y - r_elbow.y, r_hand.z - r_elbow.z);
		cv::Vec3d R_Oberarm = normalize(r_Ober);
		cv::Vec3d R_Unterarm = normalize(r_Unter);

		cv::Vec3d head_to_righthand(r_hand.x - head.x, r_hand.y - head.y, r_hand.z - head.z);
	
		double distance_head_right_Hand = sqrt(pow(head_to_righthand[0], 2) + pow(head_to_righthand[1], 2) + pow(head_to_righthand[2], 2));
		double winkel_radiant = acos(R_Oberarm[0] * R_Unterarm[0] + R_Oberarm[1] * R_Unterarm[1] + R_Oberarm[2] * R_Unterarm[2]);
		 winkel_degree = 180 - (180 / M_PI * winkel_radiant);
		

		//left
		cv::Point3d L_shoulder(LShoulder.getPosition().x, LShoulder.getPosition().y, LShoulder.getPosition().z);
		cv::Point3d L_elbow(LElbow.getPosition().x, LElbow.getPosition().y, LElbow.getPosition().z);
		cv::Point3d L_hand(LHand.getPosition().x, LHand.getPosition().y, LHand.getPosition().z);
	

		cv::Vec3d L_Ober(L_elbow.x - L_shoulder.x, L_elbow.y - L_shoulder.y, L_elbow.z - L_shoulder.z);
		cv::Vec3d L_Unter(L_hand.x - L_elbow.x, L_hand.y - L_elbow.y, L_hand.z - L_elbow.z);
		cv::Vec3d L_Oberarm = normalize(L_Ober);
		cv::Vec3d L_Unterarm = normalize(L_Unter);

		cv::Vec3d head_to_lefthand(L_hand.x - head.x, L_hand.y - head.y, L_hand.z - head.z);

		double distance_head_left_Hand = sqrt(pow(head_to_lefthand[0], 2) + pow(head_to_lefthand[1], 2) + pow(head_to_lefthand[2], 2));
		double winkel2_radiant = acos(L_Oberarm[0] * L_Unterarm[0] + L_Oberarm[1] * L_Unterarm[1] + L_Oberarm[2] * L_Unterarm[2]);
		 winkel2_degree = 180 - (180 / M_PI * winkel2_radiant);

		//Right
		if (RShoulder.getPositionConfidence() > .5f &&RElbow.getPositionConfidence() > .5f && RHand.getPositionConfidence() > .5f)
		{
			
			
				sprintf_s(buffer1, "Angle Right Arm= %f Degree", winkel_degree);
				rAngle = true;

			
				if (distance_head_right_Hand < 400 && Head.getPositionConfidence()> .5f && (RHand.getPositionConfidence() > .5f))

			{
			
				if (winkel_degree >= 23.0 && winkel_degree <= 50.0)
				{
					
					rContact = true;
					if (contact_right)
					{
						drink = true;
						rDrinking = true;
						
					}
					else
					{
						rDrinking = false;
					}
				}
				else
				{
					rDrinking = false;
					rContact = false;
				}
				
			}
				else
				{
					rDrinking = false;
					rContact = false;
				}
			
		}
		else
		{
			rDrinking = false;
			rAngle = false;
			rContact = false;

		}
		//left
		if (LShoulder.getPositionConfidence() > .5f && LElbow.getPositionConfidence() > .5f && LHand.getPositionConfidence() > .5f)
		{
			

			sprintf_s(buffer2, "Angle Left Arm= %f Degree", winkel2_degree);
			lAngle = true;


			if (distance_head_left_Hand < 400 && Head.getPositionConfidence() > .5f && (LHand.getPositionConfidence() > .5f))

			{

				if (winkel2_degree >= 23.0 && winkel2_degree <= 50.0)
				{
					lContact = true;
					
					if (contact_left)
					{
						drink = true;
						lDrinking = true;
						
					}
					else
					{
						lDrinking = false;
					}
				}
				else
				{
					lDrinking = false;
					lContact = false;
				}
				
			}
			else
			{
				lDrinking = false;
				lContact = false;
			}
			
		}
		else
		{
			lDrinking = false;
			lAngle = false;
			lContact = true;
		}
	}

void Objecttracking()
	{
	
		
		Mat temp;
		cvSet(drawingIpl, cvScalar(0));
		CvMoments *colorMoment = (CvMoments*)malloc(sizeof(CvMoments));
		static const int thickness = 3;
		static const int lineType = 8;
		Scalar           color = CV_RGB(255, 255, 255); 

		temp = thresholdMat.clone();
		Contours->imageData = (char*)temp.data; 


		CvMemStorage*   storage = cvCreateMemStorage(0);
		CvSeq*          contours = 0;
		CvSeq*			biggestContour = 0;
		int             numCont = 0;
		int             contAthresh = 45;

		cvFindContours(Contours, storage, &contours, sizeof(CvContour),
			CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
		
		int largest_area = 0;
		

		for (; contours != 0; contours = contours->h_next) 
		{
			double a = cvContourArea(contours, CV_WHOLE_SEQ); 

			if (a > largest_area){

				largest_area = a;
				
				biggestContour = contours;
			}
			

		}
		
		if (biggestContour != 0)
		{
			cvDrawContours(drawingIpl, biggestContour, color, color, -1, thickness, lineType, cvPoint(0, 0));
		}
		
			cvMoments(drawingIpl, colorMoment, 1);
			if (enableObjectTracking)
			{

		double moment10 = cvGetSpatialMoment(colorMoment, 1, 0);
		double moment01 = cvGetSpatialMoment(colorMoment, 0, 1);
		double area = cvGetCentralMoment(colorMoment, 0, 0);
		if (area > 30)
		{

			posX = (moment10 / area);
			posY = moment01 / area;
			Object.x = posX;
			Object.y = posY;

			objectTracked = true;
			//int depthindex = posX + (posY * 640);
			string string = "Tracking Object " ;
			putText(cameraFeed, string, Point(30, 450), 1, 1, Scalar(0, 255, 0), 2);
			drawObject(posX, posY, cameraFeed);
		}
			
			else
			{
				objectTracked = false;
				posX = 0;
				posY = 0;
				Object = { 0, 0 };
				putText(cameraFeed, "No Object", Point(30, 450), 1, 1, Scalar(0, 255, 0), 2);
				
			}
		
	}
		else
		{
		
			posX = 0;
			posY = 0;
			Object = { 0, 0 };
			objectTracked = false;
		}
	
}
	
void createTrackbars()
{
		namedWindow(trackbarWindowName,0);
		
		char TrackbarName[100];
		sprintf_s(TrackbarName, "Hue_MIN", H_MIN);
		sprintf_s(TrackbarName, "Hue_MAX", H_MAX);
		sprintf_s(TrackbarName, "Saturation_MIN", S_MIN);
		sprintf_s(TrackbarName, "Saturation_MAX", S_MAX);
		sprintf_s(TrackbarName, "Value_MIN", V_MIN);
		sprintf_s(TrackbarName, "Value_MAX", V_MAX);
		
		
		createTrackbar("Hue_MIN", trackbarWindowName, &H_MIN, 256, on_trackbar);
		createTrackbar("Hue_MAX", trackbarWindowName, &H_MAX, 256, on_trackbar);
		createTrackbar("Saturation_MIN", trackbarWindowName, &S_MIN, 256, on_trackbar);
		createTrackbar("Saturation_MAX", trackbarWindowName, &S_MAX, 256, on_trackbar);
		createTrackbar("Value_MIN", trackbarWindowName, &V_MIN, 256, on_trackbar);
		createTrackbar("Value_MAX", trackbarWindowName, &V_MAX, 256, on_trackbar);
	}

void on_trackbar(int, void*)
{//This function gets called whenever a
	// trackbar position is changed



}
void drawObject(int x, int y, Mat &frame)
{

	circle(frame, Point(x, y), 20, Scalar(0, 255, 0), 2);
	if (y - 25>0)
		line(frame, Point(x, y), Point(x, y - 25), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(x, 0), Scalar(0, 255, 0), 2);
	if (y + 25<FRAME_HEIGHT)
		line(frame, Point(x, y), Point(x, y + 25), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(x, FRAME_HEIGHT), Scalar(0, 255, 0), 2);
	if (x - 25>0)
		line(frame, Point(x, y), Point(x - 25, y), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(0, y), Scalar(0, 255, 0), 2);
	if (x + 25<FRAME_WIDTH)
		line(frame, Point(x, y), Point(x + 25, y), Scalar(0, 255, 0), 2);
	else line(frame, Point(x, y), Point(FRAME_WIDTH, y), Scalar(0, 255, 0), 2);

	putText(frame, to_string(x) + "," + to_string(y), Point(x, y + 30), 1, 1, Scalar(0, 255, 0), 2);

}


void renderBitmapString(float x, float y, void *font,  char *string, int color, int size)
{
	glMatrixMode(GL_PROJECTION);					
	glPushMatrix();							
	glLoadIdentity();						
	glOrtho(0, 1280, 0, 640, -1, 1);
	glMatrixMode(GL_MODELVIEW);					
	glPushMatrix();							
	glLoadIdentity();

	
	switch (color)
	{
	case 1:
		glColor3d(1.0, 0.0, 0.0);
		break;

	case 2:
		glColor3d(0.0, 0.0, 1.0);
		break;

	case 3:
		glColor3d(0.0, 1.0, 0.0);
		break;


	case 4:
		glColor3d(0.0, 0.0, 0.0);
		break;

	case 5:
		glColor3d(0.5, 0.8, 0.6);
			break;


	default:
		break;
	}
	 char *c;
	/*glRasterPos2f(x, y);
	for (c = string; *c != '\0'; c++) {
		glutBitmapCharacter(font, *c);
	}*/


	 glPushMatrix();
	 glTranslatef(x, 640-y, 0);
	 switch (size)
	 {
	 case 1:
		 glScalef(0.2, 0.2, 0.1);
		 glLineWidth(3);
		 break;
	 case 2:
		 glScalef(0.1, 0.15, 0.1);
		 glLineWidth(2);
		 break;
	 default:
		 break;
	 }
	
	 
	 for (c = string; *c; c++)
		 glutStrokeCharacter(GLUT_STROKE_ROMAN, *c);
	 glPopMatrix();
	glMatrixMode(GL_PROJECTION);					// Select Projection
	glPopMatrix();							// Pop The Matrix
	glMatrixMode(GL_MODELVIEW);					// Select Modelview
	glPopMatrix();
}