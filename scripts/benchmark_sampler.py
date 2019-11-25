
import time

import numpy as np

from learnable_primitives.fast_sampler._sampler import \
    fast_sample_on_batch


if __name__ == "__main__":
    n_runs = 10
    shapes = (np.ones((32, 15, 3))*[[[0.2, 0.1, 0.2]]]).astype(np.float32)
    epsilons = (np.ones((32, 15, 2))*[[[0.25, 0.25]]]).astype(np.float32)

    start = time.time()
    for i in range(n_runs):
        etas, omegas = fast_sample_on_batch(shapes, epsilons, 200)
    end = time.time()

    print("Time per sample", (end-start)/n_runs)
