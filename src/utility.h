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
		duration =  static_cast<double>( std::clock() - t );
	}	

public:
	double duration;
	double totalTime;
	timer()
	{
		totalTime = 0;
	}		

	void printTime()
	{		
		printf( "Total time is: %f \n", totalTime / (double) CLOCKS_PER_SEC );
	}

	void startTimer()
	{
		t = std::clock();
	}

	void calculateTotalTime()
	{		
		endTimer();
		totalTime += duration;
	}

};

#endif