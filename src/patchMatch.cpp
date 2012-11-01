#include "parseParms.h"
#include <mex.h>
#include <string>
#include <vector>


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if( nrhs != 6)
		mexErrMsgTxt("the number of input is not correct. \nThe input: ref_img_struct, otherImage_struct, depthMap, randMap, mapDistribution, \
			choice (0 -- leftToRight, 1-- topToDown, 2 -- rightToLeft, 3 -- downToTop )\n");
	if( nlhs != 2)
		mexErrMsgTxt("the number of output is not correct. \nThe output: depthMap, mapDistributio\n ");


	patchMatch pm(prhs);


	int choice = static_cast<int>(mxGetScalar( prhs[5]));
	switch(choice)
	{
	case 0:
		pm.leftToRight();
		break;
	case 1:
		pm.TopToDown();
		break;
	case 2:
		pm.RightToLeft();
		break;
	case 3:
		pm.DownToTop();
		break;
	default:
		mexErrMsgTxt("Error: the choice is not correctly found. Choice (0 -- leftToRight, 1-- topToDown, 2 -- rightToLeft, 3 -- downToTop");
	}
	
	


	//pm._tt.printTime();	
	//printf("Total time is: %f", pm._totalTime);
	// copy the output:

	plhs[0] = pm._depthMaps.p;
	plhs[1] = pm._distributionMap.p;

}
