#include "interflib.h"

#include "utils.h"

#include <vector>
#include <math.h>
#include <future>
#include <algorithm>
#include <random>


namespace {

struct Wave {
	double ampl;
	double phase;
};

void calcImage(
		double start, double end, int nPoints,
		const Vector& wave_vector1, const Vector& center1, double radius1,
        const Vector& wave_vector2, const Vector& center2, double radius2,
        int nFrames, double lambda, double omega, bool hasInterference,
        double noiseCoeff, int nThreads, uint8_t* image, double* totIntens)
{
    std::random_device rnd;
    std::mt19937 generator(rnd());
    std::uniform_int_distribution<> distrib(0, noiseCoeff);

    auto noise = [&]() {
        return distrib(generator);
    };

	const double k = 2 * M_PI / lambda;

    auto calcWave1 = [&](double z, double x, double y) {
    	const double r2 = (x - center1[0]) * (x - center1[0]) + (y - center1[1]) * (y - center1[1]);
    	return Wave{std::exp(-r2 / (radius1 * radius1)), z * k};
    	//return Wave{1.0 * (r2 <= radius1 * radius1), z * k};
    };

    auto calcWave2 = [&](double z, double x, double y) {
    	const double r2 = (x - center2[0]) * (x - center2[0]) + (y - center2[1]) * (y - center2[1]);
    	return Wave{std::exp(-r2 / (radius2 * radius2)), z * k};
    	//return Wave{1.0 * (r2 <= radius2 * radius2), z * k};
    };


    auto calcIntens = [&](double a1, double a2, double deltaPhi) {
        const auto i1 = a1 * a1;
        const auto i2 = a2 * a2;
        double result = 0;

        if (hasInterference) {
            result = i1 + i2 + 2 * sqrt(i1 * i2) * cos(deltaPhi);
        } else {
            result = i1 + i2;
        }

        return result;
    };

    const int totalPoints = nPoints * nPoints;
    std::vector<double> ampl1(totalPoints);
    std::vector<double> ampl2(totalPoints);
    std::vector<double> deltaPhase(totalPoints);

	const double step = (end - start) / nPoints;

	auto worker = [&](int kStart, int kEnd) {
		for (int k = kStart; k < kEnd; ++k) {
			int i = k / nPoints;
			int j = k - i * nPoints;
			const Vector point = {start + i * step, start + j * step, 0};

			const Vector source2 = utils::backTrack(point, wave_vector2, center2);
	        const double dist2 = utils::dist(point, source2);
	        auto w2 = calcWave2(dist2, source2[0], source2[1]);

	        const Vector source1 = utils::backTrack(point, wave_vector1, center1);
	        const double dist1 = utils::dist(point, source1);
	        auto w1 = calcWave1(dist1, source1[0], source1[1]);

            ampl1[k] = w1.ampl;
            ampl2[k] = w2.ampl;
            deltaPhase[k] = w1.phase - w2.phase;
		}
	};

	const int pointsPerThread = totalPoints / nThreads;
	std::vector<std::future<void>> futures;

	for (int iThread = 0; iThread < nThreads; ++iThread) {
		int kStart = pointsPerThread * iThread;
		int kEnd = kStart + pointsPerThread;
		futures.push_back(std::async(std::launch::async, worker, kStart, kEnd));
	}

    // wait for intens1, intens2, deltaPhase
	for (const auto& f : futures) {
		f.wait();
	}

    auto imageWorker = [&](uint8_t* img, double time, double& integIntens) {
        for (int k = 0; k < totalPoints; ++k) {
            const double intens = calcIntens(
                ampl1[k], 
                ampl2[k], 
                deltaPhase[k] + omega * time
            );


            const double intensWithNoise =
                (255.0 * intens / 4.0 + noise()) / (255.0 + noiseCoeff) * 255.0;
            img[k] = static_cast<uint8_t>(intensWithNoise);
            integIntens += intens;
        }
    };

    std::vector<std::future<void>> imageFutures;
    for (int iFrame = 0; iFrame < nFrames; ++iFrame) {
        double time = 2 * M_PI * iFrame / nFrames;
        int ind = iFrame * totalPoints;
        uint8_t* img = image + ind;
        imageFutures.push_back(std::async(
            std::launch::async, imageWorker, img, time,
            std::ref(totIntens[iFrame])));
    }

    // wait for frames
    for (const auto& f : imageFutures) {
        f.wait();
    }
}

} // namespace


void calc_image(
		double start, double end, int nPoints,
		const double* vector1, const double*  cnt1, double radius1,
        const double* vector2, const double*  cnt2, double radius2,
        int nFrames, double lambda, double omega, bool hasInterference,
        double noiseCoeff, int nThreads, uint8_t* image, double* totIntens)
{
	auto wave_vector1 = Vector{vector1[0], vector1[1], vector1[2]};
	auto wave_vector2 = Vector{vector2[0], vector2[1], vector2[2]};
	auto center1 = Vector{cnt1[0], cnt1[1], cnt1[2]};
	auto center2 = Vector{cnt2[0], cnt2[1], cnt2[2]};

	calcImage(start, end, nPoints,
		wave_vector1, center1, radius1,
		wave_vector2, center2, radius2,
		nFrames, lambda, omega, hasInterference,
		noiseCoeff, nThreads, image, totIntens);
}
