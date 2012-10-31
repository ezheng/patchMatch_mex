#include "parseParms.h"
#include <mex.h>
#include <string>
#include <vector>


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if( nrhs != 5)
		mexErrMsgTxt("the number of input is not correct. \nThe input: ref_img_struct, otherImage_struct, depthMap, randMap, mapDistribution\n");
	if( nlhs != 2)
		mexErrMsgTxt("the number of output is not correct. \nThe output: depthMap, mapDistributio\n ");


	patchMatch pm(prhs);

	pm.leftToRight();	
	pm._tt.printTime();
	
	//printf("Total time is: %f", pm._totalTime);
	// copy the output:

	plhs[0] = pm._depthMaps.p;
	plhs[1] = pm._distributionMap.p;

}
