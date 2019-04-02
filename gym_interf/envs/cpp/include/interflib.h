extern "C" {
	__declspec( dllexport )  void calc_image(
		double start, double end, int nPoints,
		const void*  wave_vector1, const void*  center1, double radius1,
        const void*  wave_vector2, const void*  center2, double radius2,
        double time, double lambda, double omega,
        int nThreads, void* image);
}

