import numpy as np

class DataGenerator(object):
    """Data generator for DLWF on Keras"""

    def __init__(self, batch_size=32, shuffle=True):
        # Initialization
        self.batch_size = batch_size
        self.shuffle = shuffle

    def generate(self, data, labels, indices):
        """Generates batches of samples"""

        nb_instances = data.shape[0]
        nb_classes = labels.shape[1]
        sample_shape = data[0].shape
        batch_data_shape = tuple([self.batch_size] + list(sample_shape))
        batch_label_shape = (self.batch_size, nb_classes)
        # Infinite loop
        while True:
            # Generate an exploration order
            # indices = np.arange(nb_instances)
            if self.shuffle is True:
                np.random.shuffle(indices)

            # Generate batches
            imax = int(len(indices) / self.batch_size)
            for i in range(imax):
                # Form a batch
                x = np.empty(batch_data_shape)
                y = np.empty(batch_label_shape)
                for j, k in enumerate(indices[i * self.batch_size: (i + 1) * self.batch_size]):
                    x[j] = data[k]
                    y[j] = labels[k]
                if x.shape != batch_data_shape:
                    print(x.shape)
                yield x, y
