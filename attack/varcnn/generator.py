import threading
import numpy as np

class ThreadSafeIter:
    """Takes an iterator/generator and makes it thread-safe.

    Does this by serializing call to the `next` method of given iterator/
    generator. See https://anandology.com/blog/using-iterators-and-generators/
    for more information.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):  # Py3
        with self.lock:
            return next(self.it)

    def next(self):  # Py2
        with self.lock:
            return self.it.next()


def thread_safe_generator(f):
    """Decorator that takes a generator function and makes it thread-safe."""

    def g(*a, **kw):
        return ThreadSafeIter(f(*a, **kw))

    return g


@thread_safe_generator
def generate(batch_size, data_type, dir_seq, labels):
    """Yields batch of data with the correct content and formatting.

    Args:
        data_type (str): Either 'training_data', 'validation_data', or
            'test_data'
        config (dict): Deserialized JSON config file (see config.json)
    """
    batch_start = 0
    dir_seq = np.reshape(dir_seq, (dir_seq.shape[0], dir_seq.shape[1], 1))
    while True:
        if batch_start >= len(labels):
            batch_start = 0
        batch_end = batch_start + batch_size

        batch_data = (
            {},
            {'model_output': labels[batch_start:batch_end]}
        )

        # Accesses and stores relevant data slices
        batch_structs a `keras.models.Model` object using the given block count.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)
    :param blocks: the network’s residual architecture
    :param block: a residual block (e.g. an