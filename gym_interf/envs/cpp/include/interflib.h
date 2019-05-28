#pragma onces

#include "defines.h"
#include <cstdint>

extern "C" {
	EXPORT void calc_image(
		double start, double end, int nPoints,
		const double*  waveVector1, const double*  center1, double radius1,
        const double*  waveVector2, const double*  center2, double radius2,
        int nFrames, double lambda, double omega, bool hasInterference,
        int nThreads, uint8_t* image);
}

