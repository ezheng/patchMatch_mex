#ifndef UTILITY_H
#define UTILITY_H

#include <iostream>
#include <cstdio>
#include <ctime>
#include <string>

class timer
{
private:
	std::clock_t t;
    
	void endTimer()
	{
		duration = ( std::clock() - t ) / (double) CLOCKS_PER_SEC;
	}

	void startTimer()
	{
		t = std::clock();
	}

	

public:
	double duration;
	double totalTime;
	timer()
	{
		totalTime = 0;
	}		

	//void printTime()
	//{
	//	endTimer();
	//	//printf( " It takes: %f \n", duration );
	//}

	void calculateTotalTime()
	{		
		endTimer();
		totalTime += duration;
	}

};

#endif