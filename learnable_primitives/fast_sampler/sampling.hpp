#ifndef _SAMPLING_HPP_
#define _SAMPLING_HPP_


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
);


#endif // _SAMPLING_HPP_
