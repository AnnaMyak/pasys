/*
 * Angewandte Informatk (M)
 * Programmierkonzepte und Algorithmen
 * Bildverarbeitung (inklusive Analyse) mit OpenMPI & OpenMP
 * 
 * Autorin:
 * Anna Myakinen

 * 
 * Die Aufgabenstellung besteht aus 2 Sub-Aufgaben
 * 1. Bilverarbeitung: Mittelwert-Filter, Thresholding, Erosion&Dilatation, Sobel-Operator
 * 	  Parallelisierung des Bearbeitungsrozesses mit Hilfe von Open MPI(Erosion&Dilatation, Sobel-Operator)
 * 2. Hough Transform Analyse -> Kreise
 * 	  Parallelisierung des Bearbeitungsrozesses mit Hilfe von Open MP
 *  
 * 
 * Kompilieren:
 * mpicxx -openmp -o exe_mpi main.cpp `pkg-config --libs opencv`
 * 
 * Ausfuerung:
 * mpirun -n 10 ./exe_mpi test.jpeg result.jpeg
 * 
 * */


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include "mpi.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>
#include <math.h>
#include <omp.h>

#define MASTER 0
#define PI 3.14
#define MIN_RADIUS 15
#define ACCU_MATRIX_ROWS 400
#define ACCU_MATRIX_COLS 281
#define NUMBER_OF_RADII 15

using namespace std;
using namespace cv;



Mat middleValue(Mat img) {
  
  int newValue;  
  
  for(int y = 0; y < img.rows ; y++)
  {
            for(int x = 0; x < img.cols ; x++)
            {
				
				newValue =(int)img.at<uchar>(y,x);
				newValue=newValue + (int)img.at<uchar>(y-1,x);
				newValue=newValue + (int)img.at<uchar>(y+1,x);
				newValue=newValue + (int)img.at<uchar>(y-2,x);
				newValue=newValue + (int)img.at<uchar>(y+2,x);
				
				newValue=newValue + (int)img.at<uchar>(y,x-1);
				newValue=newValue + (int)img.at<uchar>(y-1,x-1);
				newValue=newValue + (int)img.at<uchar>(y+1,x-1);
				newValue=newValue + (int)img.at<uchar>(y-2,x-1);
				newValue=newValue + (int)img.at<uchar>(y+2,x-1);
				
				newValue=newValue + (int)img.at<uchar>(y,x+1);
				newValue=newValue + (int)img.at<uchar>(y-1,x+1);
				newValue=newValue + (int)img.at<uchar>(y+1,x+1);
				newValue=newValue + (int)img.at<uchar>(y-2,x+1);
				newValue=newValue + (int)img.at<uchar>(y+2,x+1);
				
				newValue= newValue/25;
				
				img.at<uchar>(y,x)=(double)newValue;
				
			}
		}
		return img;
	}



Mat* erode (Mat *slice)
	{
		for(int y = 1; y < slice->rows-1; y++)
            for(int x = 1; x < slice->cols-1; x++)
        {
			{
					
					if (slice->at<uchar>(y,x)!=slice->at<uchar>(y,x-1)&& slice->at<uchar>(y,x)!=slice->at<uchar>(y-1,x)&&
					 slice->at<uchar>(y,x)!=slice->at<uchar>(y-1,x)&& slice->at<uchar>(y,x)!=slice->at<uchar>(y,x+1)
					 && slice->at<uchar>(y,x)!=slice->at<uchar>(y+1,x))
					 {
							slice->at<uchar>(y,x)=0;
							slice->at<uchar>(y,x-1)=0;
							slice->at<uchar>(y-1,x)=0;
							slice->at<uchar>(y,x+1)=0;
							slice->at<uchar>(y+1,x)=0;
					 }
			}
		}
		
		for(int y = 1; y < slice->rows-1; y++)
            for(int x = 1; x < slice->cols-1; x++)
        {
			{
					
					if (slice->at<uchar>(y,x)!=slice->at<uchar>(y,x-1)&& slice->at<uchar>(y,x)!=slice->at<uchar>(y-1,x)&&
					 slice->at<uchar>(y,x)!=slice->at<uchar>(y-1,x)&& slice->at<uchar>(y,x)!=slice->at<uchar>(y,x+1)
					 && slice->at<uchar>(y,x)!=slice->at<uchar>(y+1,x))
					 {
							slice->at<uchar>(y,x)=0;
							slice->at<uchar>(y,x-1)=0;
							slice->at<uchar>(y-1,x)=0;
							slice->at<uchar>(y,x+1)=0;
							slice->at<uchar>(y+1,x)=0;
					 }
			}
		}
		return slice;
	}



Mat* dilate (Mat * slice)
{
			for(int y = 2; y < slice->rows-2; y++)
            for(int x = 2; x < slice->cols-2; x++)
        {
			{
					
					if (slice->at<uchar>(y,x)!=slice->at<uchar>(y,x-1)&& slice->at<uchar>(y,x)!=slice->at<uchar>(y-1,x)&&
					 slice->at<uchar>(y,x)!=slice->at<uchar>(y-1,x)&& slice->at<uchar>(y,x)!=slice->at<uchar>(y,x+1)
					 && slice->at<uchar>(y,x)!=slice->at<uchar>(y+1,x) && slice->at<uchar>(y,x)!=slice->at<uchar>(y-2,x)
					 && slice->at<uchar>(y,x)!=slice->at<uchar>(y-1,x+1)&& slice->at<uchar>(y,x)!=slice->at<uchar>(y+1,x+1)
					 && slice->at<uchar>(y,x)!=slice->at<uchar>(y,x+2) && slice->at<uchar>(y,x)!=slice->at<uchar>(y+2,x+1)
					 && slice->at<uchar>(y,x)!=slice->at<uchar>(y+1,x-1) && slice->at<uchar>(y,x)!=slice->at<uchar>(y,x-2)
					 && slice->at<uchar>(y,x)!=slice->at<uchar>(y-1,x-1)  )
					 {
							slice->at<uchar>(y,x)=255;
							slice->at<uchar>(y,x-1)=255;
							slice->at<uchar>(y-1,x)=255;
							slice->at<uchar>(y,x+1)=255;
							slice->at<uchar>(y+1,x)=255;
							slice->at<uchar>(y-2,x)=255;
							slice->at<uchar>(y-1,x+1)=255;
							slice->at<uchar>(y,x+2)=255;
							slice->at<uchar>(y+2,x)=255;
							slice->at<uchar>(y+1,x-1)=255;
							slice->at<uchar>(y,x-2)=255;
							slice->at<uchar>(y-1,x-1)=255;
							
					 
					 }
			}
		}
		
		
		
		 for(int y = 2; y < slice->rows-2; y++)
            for(int x = 2; x < slice->cols-2; x++)
        {
			{
					
					if (slice->at<uchar>(y,x)!=slice->at<uchar>(y,x-1)&& slice->at<uchar>(y,x)!=slice->at<uchar>(y-1,x)&&
					 slice->at<uchar>(y,x)!=slice->at<uchar>(y-1,x)&& slice->at<uchar>(y,x)!=slice->at<uchar>(y,x+1)
					 && slice->at<uchar>(y,x)!=slice->at<uchar>(y+1,x) && slice->at<uchar>(y,x)!=slice->at<uchar>(y-2,x)
					 && slice->at<uchar>(y,x)!=slice->at<uchar>(y-1,x+1)&& slice->at<uchar>(y,x)!=slice->at<uchar>(y+1,x+1)
					 && slice->at<uchar>(y,x)!=slice->at<uchar>(y,x+2) && slice->at<uchar>(y,x)!=slice->at<uchar>(y+2,x+1)
					 && slice->at<uchar>(y,x)!=slice->at<uchar>(y+1,x-1) && slice->at<uchar>(y,x)!=slice->at<uchar>(y,x-2)
					 && slice->at<uchar>(y,x)!=slice->at<uchar>(y-1,x-1)  )
					 {
							slice->at<uchar>(y,x)=255;
							slice->at<uchar>(y,x-1)=255;
							slice->at<uchar>(y-1,x)=255;
							slice->at<uchar>(y,x+1)=255;
							slice->at<uchar>(y+1,x)=255;
							slice->at<uchar>(y-2,x)=255;
							slice->at<uchar>(y-1,x+1)=255;
							slice->at<uchar>(y,x+2)=255;
							slice->at<uchar>(y+2,x)=255;
							slice->at<uchar>(y+1,x-1)=255;
							slice->at<uchar>(y,x-2)=255;
							slice->at<uchar>(y-1,x-1)=255;
							
					 
					 }
			}
		}
		
		
		

	
	
	
	return slice;
	}



int xGradient(Mat *slice, int x, int y)
{
    return slice->at<uchar>(y-1, x-1) +
                2*slice->at<uchar>(y, x-1) +
                 slice->at<uchar>(y+1, x-1) -
                  slice->at<uchar>(y-1, x+1) -
                   2*slice->at<uchar>(y, x+1) -
                    slice->at<uchar>(y+1, x+1);
}



int yGradient(Mat *slice, int x, int y)
{
    return slice->at<uchar>(y-1, x-1) +
                2*slice->at<uchar>(y-1, x) +
                 slice->at<uchar>(y-1, x+1) -
                  slice->at<uchar>(y+1, x-1) -
                   2*slice->at<uchar>(y+1, x) -
                    slice->at<uchar>(y+1, x+1);
}



Mat* sobel(Mat* slice, int imgCols, int imgRows){
	
	Mat* dst = new Mat(imgRows, imgCols, CV_8UC1);
    int gx, gy, sum;
    
    for(int y = 0; y < slice->rows; y++)
            for(int x = 0; x < slice->cols; x++)
                dst->at<uchar>(y,x) = 0.0;

        for(int y = 1; y < slice->rows - 1; y++){
            for(int x = 1; x < slice->cols - 1; x++){
                gx = xGradient(slice, x, y);
                gy = yGradient(slice, x, y);
                sum = abs(gx) + abs(gy);
                sum = sum > 255 ? 255:sum;
                sum = sum < 0 ? 0 : sum;
                dst->at<uchar>(y,x) = sum;
            }
        }
    
    return dst;
	
	} 



vector<Point> getAllaActivePoints (Mat *result)
{
	vector<Point> v;
	for(int x = 1; x < result->rows-1; x++)
            for(int y = 1; y < result->cols-1; y++)
        {
			{
				if (result->at<uchar>(y,x)==255)
				{
					v.push_back(Point(x,y));
				}
			}
		}
	
	return v;
}



int getMaxRadius(int iteration)
	{
		int r=MIN_RADIUS;
		for (int i=1;i<NUMBER_OF_RADII;i++)
			{
				r=r+iteration;
			}
		
		return r;
	}

void votingAndCalculating(vector<Point> v, int iteration, int A[ACCU_MATRIX_ROWS][ACCU_MATRIX_COLS][NUMBER_OF_RADII])
	{
		
		int maxRadius=getMaxRadius(iteration);
		
					for (int i=0; i<v.size();i++)
						{
							for(int radius=MIN_RADIUS;radius<maxRadius; radius=radius+iteration)
								{
									for(int theta=0; theta<360; theta++)
										{
											int a= v[i].x - radius*cos(theta*PI/180);
											int b= v[i].y - radius*sin(theta*PI/180);
						
											if(a>0 && b>0)
												{
													A[a][b][radius]=A[a][b][radius]+1;
												}
										}
								}
						}
		
		printf ("Lokale Maximas. Range %d - %d", MIN_RADIUS, maxRadius); 
		int count=0;
    
			for (int i=1; i<ACCU_MATRIX_ROWS; i++)
				{
					for (int j=1; j<ACCU_MATRIX_COLS; j++)
						{
							for(int r=0;r<NUMBER_OF_RADII; r++)
								{
									if (A[i][j][r]!=0 && A[i][j][r]%360!=0 && A[i][j][r]>0)
									{
									count++;
									}
									
								}
						}
				}
    
		printf (" Insgesamt %d Kreise\n",count);
	}




int main(int argc, char *argv[])
{
	int A[ACCU_MATRIX_ROWS][ACCU_MATRIX_COLS][NUMBER_OF_RADII] = {};
    //MPI
    
    //Rank und Anzahl von erzeugten Prozessen  
    int rank, comm_sz;  
    //Variablen, die noetig fuer die Verteilung zwischen Prozessen der Sourcedatei
    int *counts, *rowsForProcess, *displacements; 
    
    //Bildverarbeitung
    
    //An die Variable wird die Sorcedatei(Eingabebild) uebergeben 
    Mat source;
    //In die Variable werden  die Ergebnisse geschrieben
    Mat* result = new Mat();
    //in den Variablen  werden Zeilen- und Spaltenanzahl geschrieben 
    int imgRows, imgCols;
    
    //Variablen fuer den Timer
    double start, total;
    
	//Initialisierung von MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&comm_sz);
    
    //Berechnen, wie viel Bildaufschnitte werden im Programm bearbeitet
    int numOfSlices = comm_sz-1;
    int colsPerProcess;

    if (rank == MASTER) {
		
		
        
		
		 
        if(argc!=3) {
            return -1;
        }
		//Die Programmausfuerung ist moeglich nur mit mindestens 2 Prozessen
        if(comm_sz == 1) {
            printf("Achtung!!! Bitte min. 2 Prozesse starten lassen!!!\n");
            return -1;
        }

		//Eingabe- und  Ergebnisbild initialisieren
        source = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
        imgCols = source.cols;
        imgRows = source.rows;
        result = new Mat(imgRows, imgCols, CV_8UC1);
        
        //Alle unnoetge Informationen wie Rausch oder unnoetige Kanten aus dem Bild loeschen
        //Glaetung
        source=middleValue(source);
        source=middleValue(source);
        source=middleValue(source);
        
        //Schwellenwert festlegen
        //Objekte von Hintergrund separieren
        //Objekte sind schwarz
        //Hintergrund weiss 
        threshold(source, source, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
        
        
        //in den naesten Zeilen wird berechnet, 
        //wie viel jeder Slave-Prozess Informationen (rows, cols und bytes) bekommt
        //natuerlich mit der Ausnahme Master-Prozess
        //Das Programm versucht das Eingabebild (die Bildmatrix) zwischen Prozessen maximal gleichmaessig zu verteilen       
        counts = new int[numOfSlices];
        displacements = new int[numOfSlices];
        rowsForProcess = new int[comm_sz];

        for (int i = 0; i < comm_sz; i++) {
            counts[i] = displacements[i] = 0;
        }

        rowsForProcess[0] = 0; 
        
        int colSize = imgRows / numOfSlices;

        for (int i = 1; i < comm_sz; i++) {
            rowsForProcess[i] = colSize;
            displacements[i] = displacements[i-1] + counts[i-1];
            counts[i] = colSize * imgCols * 1;
        }


        if(numOfSlices > 1) {
            int lastSliceSize = imgRows % (colSize * (numOfSlices - 1));

            if (lastSliceSize == 0) {
                lastSliceSize = colSize;
            }
            rowsForProcess[comm_sz - 1] = lastSliceSize;
            displacements[comm_sz - 1] = displacements[comm_sz - 2] + counts[comm_sz - 2];
            counts[comm_sz - 1] = lastSliceSize * imgCols * 1;
        }
    }

	//Anzahl von Zeilen an die Prozesse verteilen
    MPI_Scatter(rowsForProcess, 1, MPI_INT, &imgRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    //Anzahl von Spalten an die Prozeesse schicken
    MPI_Bcast(&imgCols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    //Jeder Slave-Prozess bearbeitet seine Menge der Arbeit mit eigener Geschwindigkeit
    //Die Prozesse muessen synchronisiert werden
    MPI_Barrier(MPI_COMM_WORLD); 
    
	//Bilden Bilschnitt
	//An den wird eine vorher berechnete Menge der Bildinformationen gesendet 
    Mat* slice = new Mat(imgRows, imgCols, CV_8UC1);
    
    //Bildinformationen an die Prozesse, bzw. Bildschniite verteilen(senden)
    MPI_Scatterv(source.data, counts, displacements, MPI_BYTE,
                 slice->data, imgRows * imgCols * 1, MPI_BYTE, 0, MPI_COMM_WORLD);

    //Start der Bearbeitung speichern 
    start = MPI_Wtime();
    
    //Slave-Prozesse bearbeiten die Bildschnite pixelweise 
    if (1 <= rank && rank < comm_sz) {
    
    //Kleine unnoetige elemente loeschen
	slice=erode(slice);
	slice=erode(slice);
	slice=erode(slice);
	
	//Die Loecher schliessen
    slice=dilate(slice);
    slice=dilate(slice);
    slice=dilate(slice);
    
    //Die Kanten aus dem Bild extraheren
    slice=sobel(slice,imgCols,imgRows);
	
    
    }
	//Die bearbeitete Bildschnitte zusammenfuehren
    MPI_Gatherv(slice->data, imgRows * imgCols * 1, MPI_BYTE,
                result->data, counts, displacements, MPI_BYTE, 0, MPI_COMM_WORLD);
    
    //Bearbeitungszeit berechen
    total = MPI_Wtime() - start;
    
    //MPI beenden, Anzahl der Prozessen ist jetzt undefiniert
    MPI_Finalize();

    //Nach der Ausfuehrung
    //Zurueck in den Master-Prozess, rank==0
    if (rank == MASTER) {

        printf("Bearbeitungszeit betraegt: %f ms\n", (total*1000) );
        
        //Ergebnisse in die Datei schreiben
        imwrite(argv[2], *result);
		
		//Speicherplatz freigeben
        free(counts);
        free(displacements);
    }
    free(slice);
    
    //Alle aktive Punkte (Value==255) aus dem Bild in einem Vektor zu speichern
    //Der Vektor wird staat des Bildes im weiteren an die verschidenen Threads fuer das Voting und Analyse uebergeben  
    vector<Point> v= getAllaActivePoints(result);
	//Das Bild wird im weiteren nicht mehr benutzt. Der Speicher kann jetzt freigegeben werden
	free(result);
	
    
    //Parallelesierung mit Open MP 
    
    
    // Hough Transform
    
    //Jede MP-Sektion bekommt einen Radiusraum und sucht im Bild die Anzahl von Kreisen
    //Die gefundenen Informationen werden in die Konsole ausgegeben
    
	#pragma omp parallel sections
    {    
        #pragma omp section
        votingAndCalculating(v, 2, A);
            
        #pragma omp section
        votingAndCalculating(v, 4, A);
        
        #pragma omp section
        votingAndCalculating(v, 6, A);
        
        #pragma omp section
        votingAndCalculating(v, 8, A);
        
    }
	
    return 0;
}



