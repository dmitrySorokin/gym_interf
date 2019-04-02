#include "interflib.h"

#include "utils.h"

#include <vector>
#include <cmath>
#include <future>

namespace {

struct Wave {
	double ampl;
	double phase;
};

double calcIntens(const Wave& w1, const Wave& w2)
{
	const auto i1 = w1.ampl * w1.ampl;
	const auto i2 = w2.ampl * w2.ampl;

	return i1 + i2 + 2 * sqrt(i1 * i2) * cos(w1.phase - w2.phase);
}

void calcImage(
		double start, double end, int nPoints,
		const Vector& wave_vector1, const Vector& center1, double radius1,
        const Vector& wave_vector2, const Vector& center2, double radius2,
        double time, double lambda, double omega,
        int nThreads, double* image)
{
	const double k = 2 * M_PI / lambda;

    auto ampl1 = [&](double z, double x, double y) {
    	const double r2 = (x - center1[0]) * (x - center1[0]) + (y - center1[1]) * (y - center1[1]);
    	return Wave{std::exp(-r2 / (radius1 * radius1)), z * k};
    };

    auto ampl2 = [&](double z, double x, double y) {
    	const double r2 = (x - center2[0]) * (x - center2[0]) + (y - center2[1]) * (y - center2[1]);
    	return Wave{std::exp(-r2 / (radius2 * radius2)), z * k - omega * time};
    };

	const double step = (end - start) / nPoints;

	auto worker = [&](int kStart, int kEnd) {
		for (int k = kStart; k < kEnd; ++k) {
			int i = k / nPoints;
			int j = k - i * nPoints;
			const Vector point = {start + i * step, start + j * step, 0};

			const Vector source2 = utils::backTrack(point, wave_vector2, center2);
	        const double dist2 = utils::dist(point, source2);
	        auto u2 = ampl2(dist2, source2[0], source2[1]);

	        const Vector source1 = utils::backTrack(point, wave_vector1, center1);
	        const double dist1 = utils::dist(point, source1);
	        auto u1 = ampl1(dist1, source1[0], source1[1]);

	        double intens = calcIntens(u1, u2);

	        image[k] = intens;
		}
	};


	const int totalPoints = nPoints * nPoints;
	const int pointsPerThread = totalPoints / nThreads;
	std::vector<std::future<void>> futures;

	for (int iThread = 0; iThread < nThreads; ++iThread) {
		int kStart = pointsPerThread * iThread;
		int kEnd = kStart + pointsPerThread;
		futures.push_back(std::async(std::launch::async, worker, kStart, kEnd));
	}

	for (const auto& f : futures) {
		f.wait();
	}
}

} // namespace


void calc_image(
		double start, double end, int nPoints,
		const void* v1, const void*  c1, double radius1,
        const void* v2, const void*  c2, double radius2,
        double time, double lambda, double omega,
        int nThreads, void* img)
{
	const double* vector1 = static_cast<const double*>(v1);
	auto wave_vector1 = Vector{vector1[0], vector1[1], vector1[2]};

	const double* vector2 = static_cast<const double*>(v2);
	auto wave_vector2 = Vector{vector2[0], vector2[1], vector2[2]};

	const double* cnt1 = static_cast<const double*>(c1);
	auto center1 = Vector{cnt1[0], cnt1[1], cnt1[2]};

	const double* cnt2 = static_cast<const double*>(c2);
	auto center2 = Vector{cnt2[0], cnt2[1], cnt2[2]};

	double* image = static_cast<double*>(img);

	calcImage(start, end, nPoints,
		wave_vector1, center1, radius1,
		wave_vector2, center2, radius2,
		time, lambda, omega,
		nThreads, image);
}