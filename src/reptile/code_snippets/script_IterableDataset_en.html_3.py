import math
import paddle
import numpy as np
from paddle.io import IterableDataset, DataLoader, get_worker_info

class RangeIterableDataset(IterableDataset):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __iter__(self):
        for i in range(self.start, self.end):
            yield np.array([i])

dataset = RangeIterableDataset(start=2, end=9)

def worker_init_fn(worker_id):
    worker_info = get_worker_info()

    dataset = worker_info.dataset
    start = dataset.start
    end = dataset.end
    num_per_worker = int(
        math.ceil((end - start) / float(worker_info.num_workers)))

    worker_id = worker_info.id
    dataset.start = start + worker_id * num_per_worker
    dataset.end = min(dataset.start + num_per_worker, end)

dataloader = DataLoader(
    dataset,
    num_workers=2,
    batch_size=1,
    drop_last=True,
    worker_init_fn=worker_init_fn)

for data in dataloader:
    print(data)
# outputs: [2, 5, 3, 6, 4, 7]