import os
import sys
import threading
import time

import numpy as np
import torch

from ..utils.progbar import Progbar


class BatchProvider(object):
    """BatchProvider class is a wrapper class to generate batches to train the
    network.
    """
    def __init__(
        self,
        dataset,
        batch_size,
        cache_size=500,
        verbose=1
    ):
        self._dataset = dataset
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.verbose = verbose

        # Shape of the input tensor to the network
        self.input_shape = self._dataset.input_dim
        # Shape of the output tensor from the network
        self.output_shape = self._dataset.output_dim

        # Member variable to stop the thread (to be set via a call to stop)
        self._stop = False
        self._ready = False

        # Start a thread to fill the cache
        self._start_producer_thread()

    def ready(self, blocking=True):
        while blocking and not self._ready:
            time.sleep(0.1)
        return self._ready

    def stop(self):
        self._stop = True
        self._producer_thread.join()

    def __iter__(self):
        return self

    def __next__(self):
        idxs = np.random.randint(0, self.cache_size, size=self.batch_size)
        with self.cache_lock:
            return self.X[idxs], self.y[idxs]

    def _start_producer_thread(self):
        # This is going to be the amount of cached elements
        N = self.cache_size
        self.X = torch.empty((N,) + self.input_shape, dtype=torch.float32)
        self.y = torch.empty((N,) + self.output_shape, dtype=torch.float32)

        self.cache_lock = threading.RLock()
        self._producer_thread = threading.Thread(target=self._producer)
        self._producer_thread.daemon = True
        self._producer_thread.start()

    def _producer(self):
        N = self.cache_size
        passes = 0
        if self.verbose > 0:
            prog = Progbar(N)

        while True:
            # Acquire the lock for the whole first pass
            if passes == 0:
                self.cache_lock.acquire()

            for idx in range(N):
                # We 're done stop now
                if self._stop and passes > 0:
                    return

                while True:
                    sample = self._dataset.get_random_datapoint()
                    X = sample[0]
                    y = sample[1]
                    if X is None or y is None:
                        continue
                    break

                # Do the copy to the cache but make sure you lock first and
                # unlock afterwards
                with self.cache_lock:
                    try:
                        self.X[idx] = X
                        self.y[idx] = y
                    except Exception as e:
                        sys.stderr.write(
                            "Exception caught in producer thread " + str(e)
                        )

                # Show progress if it is the first pass
                if passes == 0 and self.verbose > 0:
                    prog.update(idx + 1)

            # Release the lock if it was the first pass
            if passes == 0:
                self._ready = True
                self.cache_lock.release()

            # Count the passes
            passes += 1
