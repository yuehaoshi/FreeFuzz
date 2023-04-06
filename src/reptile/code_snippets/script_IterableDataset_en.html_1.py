import numpy as np
from paddle.io import IterableDataset

# define a random dataset
class RandomDataset(IterableDataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        for i in range(self.num_samples):
            image = np.random.random([784]).astype('float32')
            label = np.random.randint(0, 9, (1, )).astype('int64')
            yield image, label

dataset = RandomDataset(10)
for img, lbl in dataset:
    print(img, lbl)