
#include <iostream>
#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

extern "C" {
#include "sampling.hpp"
}


const float pi = std::acos(-1);
const float pi_2 = pi/2;


class prng {
    public:
        prng(int seed) : gen(seed), dis(0, 1) {}
        float operator()() {
            return dis(gen);
        }

    private:
        std::mt19937 gen;
        std::uniform_real_distribution<float> dis;
};


struct recursion_params {
    float A[2];
    float B[2];
    float theta_a;
    float theta_b;
    int N;
    int offset;

    recursion_params(
        float a[2],
        float b[2],
        float t_a,
        float t_b,
        int n,
        int o
    ) {
        A[0] = a[0];
        A[1] = a[1];
        B[0] = b[0];
        B[1] = b[1];
        theta_a = t_a;
        theta_b = t_b;
        N = n;
        offset = o;
    }
};


inline float fexp(float x, float p) {
    return std::copysign(std::pow(std::abs(x), p), x);
}


inline void xy(float theta, float a1, float a2, float e, float C[2]) {
    C[0] = a1 * fexp(std::cos(theta), e);
    C[1] = a2 * fexp(std::sin(theta), e);
}

inline float distance(float A[2], float B[2]) {
    float d1 = A[0]-B[0];
    float d2 = A[1]-B[1];
    return std::sqrt(d1*d1 + d2*d2);
}


void sample_superellipse_divide_conquer(
    float a1,
    float a2,
    float e,
    float theta_a,
    float theta_b,
    std::vector<float> &buffer,
    std::vector<recursion_params> &stack
) {
    float A[2], B[2], C[2], theta, dA, dB;
    int nA, nB;

    xy(theta_a, a1, a2, e, A);
    xy(theta_b, a1, a2, e, B);
    buffer[0] = theta_a;
    stack.emplace_back(A, B, theta_a, theta_b, buffer.size()-2, 1);

    while (stack.size() > 0) {
        recursion_params params = stack.back();
        stack.pop_back();

        if (params.N <= 0) {
            continue;
        }

        theta = (params.theta_a + params.theta_b)/2;
        xy(theta, a1, a2, e, C);
        dA = distance(params.A, C);
        dB = distance(C, params.B);
        nA = static_cast<int>(std::round((dA/(dA+dB))*(params.N-1)));
        nB = params.N - nA - 1;

        buffer[nA+params.offset] = theta;

        stack.emplace_back(
            params.A, C,
            params.theta_a, theta,
            nA,
            params.offset
        );
        stack.emplace_back(
            C, params.B,
            theta, params.theta_b,
            nB,
            params.offset + nA + 1
        );
    }

    buffer[buffer.size()-1] = theta_b;
}


void sample_etas(
    std::function<float()> rand,
    float a1a2,
    float e1,
    std::vector<float> &buffer,
    std::vector<float> &cdf,
    float *etas,
    int N
) {
    const float smoothing = 0.001;
    float s;

    // Make the sampling distribution's CDF
    cdf[0] = smoothing;
    for (unsigned int i=1; i<cdf.size(); i++) {
        cdf[i] = cdf[i-1] + smoothing + a1a2*fexp(std::cos(buffer[i]), e1);
    }
    s = cdf.back();
    for (unsigned int i=0; i<cdf.size(); i++) {
        cdf[i] /= s;
    }

    // Sample N points
    for (int i=0; i<N; i++) {
        auto pos = std::lower_bound(cdf.begin(), cdf.end(), rand());
        etas[i] = buffer[std::distance(cdf.begin(), pos)];
    }
}


void sample_on_batch(
    float *shapes,
    float *epsilons,
    float *etas,
    float *omegas,
    int B,
    int M,
    int N,
    int buffer_size,
    int seed
) {
    prng rand(seed);

    std::vector<float> buffer(buffer_size);
    std::vector<float> eta_cdf(buffer_size);
    std::vector<recursion_params> stack;

    for (int b=0; b<B; b++) {
        for (int m=0; m<M; m++) {
            float *a1a2a3 = shapes + b*M*3 + m*3;
            float *e1e2 = epsilons + b*M*2 + m*2;
            float *eta = etas + b*M*N + m*N;
            float *omega = omegas + b*M*N + m*N;

            // Sample the etas
            sample_superellipse_divide_conquer(
                a1a2a3[0],
                a1a2a3[2],
                e1e2[0],
                pi_2, -pi_2,
                buffer,
                stack
            );
            sample_etas(
                std::ref(rand),
                a1a2a3[0]+a1a2a3[1],
                e1e2[0],
                buffer,
                eta_cdf,
                eta,
                N
            );

            // Sample the omegas
            sample_superellipse_divide_conquer(
                a1a2a3[0],
                a1a2a3[1],
                e1e2[1],
                pi, -pi,
                buffer,
                stack
            );
            for (int i=0; i<N; i++) {
                omega[i] = buffer[static_cast<int>(rand()*buffer_size)];
            }
        }
    }
}
