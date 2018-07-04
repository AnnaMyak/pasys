#include<iostream>
#include<cmath>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include "opencv2/imgcodecs.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>
#include <math.h>
#include <string> 


using namespace std;
using namespace cv;

#define PI 3.14
#define MIN_RADIUS 15
#define ACCU_MATRIX_ROWS 400
#define ACCU_MATRIX_COLS 281
#define NUMBER_OF_RADII 15	



int getMaxRadius(int iteration)
	{
		int r=MIN_RADIUS;
		for (int i=1;i<NUMBER_OF_RADII;i++)
			{
				r=r+iteration;
			}
		
		return r;
	}

void calculateCircles(Mat img, int iteration, int A[ACCU_MATRIX_ROWS][ACCU_MATRIX_COLS][NUMBER_OF_RADII])
	{
		//int A[ACCU_MATRIX_ROWS][ACCU_MATRIX_COLS][NUMBER_OF_RADII] = {};
		int maxRadius=getMaxRadius(iteration);
		
		for (int x=1; x<img.rows; x++)
		{
			for (int y=1; y<img.rows; y++)
				{
					if (img.at<uchar>(y,x)==255)
						{
							for(int r=MIN_RADIUS;r<maxRadius; r=r+iteration)
								{
									for(int i=0; i<360; i++)
										{
											int a= x - r*cos(r*PI/180);
											int b= y - r*sin(r*PI/180);
						
											if(a>0 && b>0)
												{
													A[a][b][r]=A[a][b][r]+1;
												}
										}
								}
						}
				}
		}
		
		printf ("Lokale Maximas. Range %d - %d", MIN_RADIUS, maxRadius); 
		int k=0;
    
			for (int i=1; i<ACCU_MATRIX_ROWS; i++)
				{
					for (int j=1; j<ACCU_MATRIX_COLS; j++)
						{
							for(int r=0;r<NUMBER_OF_RADII; r++)
								{
									if (A[i][j][r]!=0 && A[i][j][r]%360!=0 && A[i][j][r]>0)
									{
									k++;
									}
									
								}
						}
				}
    
		printf (" Insgesamt %d Kreise\n",k);
	}
	











int main()
	{
	
	float Pi=3.14;
	int A[ACCU_MATRIX_ROWS][ACCU_MATRIX_COLS][NUMBER_OF_RADII] = {};
	
	
	Mat img=imread("result.png",CV_LOAD_IMAGE_GRAYSCALE);
	calculateCircles(img,2,A);
	calculateCircles(img,3,A);
	calculateCircles(img,4,A);
	calculateCircles(img,5,A);
	
	
	/*
	for (int x=1; x<img.rows; x++)
	{
		for (int y=1; y<img.rows; y++)
		{
			if (img.at<uchar>(y,x)==255)
			{
				for(int r=15;r<85; r=r+5)
				{
					for(int i=0; i<360; i++)
					{
						int a= x - r*cos(r*Pi/180);
						int b= y - r*sin(r*Pi/180);
						
							if(a>0 && b>0)
								{
									A[a][b][r]=A[a][b][r]+1;
								}
					}
					}
				}
			}
		
	}
    
    printf ("Lokale Maximas. Range 15 - 85");
    int k=0;
    
    for (int i=1; i<400; i++)
    {
		for (int j=1; j<281; j++)
		{
			for(int r=15;r<30; r++)
				{
					if (A[i][j][r]!=0 && A[i][j][r]%360!=0 && A[i][j][r]>0)
					{
						k++;
					}
									
				}
		}
	}
    
    printf (" Insgesamt %d Kreise\n",k);
    
    
    
    */
    
    
    
 
    
    
    
    
    return 0;
	}


