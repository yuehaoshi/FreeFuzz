import math
import paddle
import numpy as np
from paddle.io import IterableDataset, DataLoader, get_worker_info

class SplitedIterableDataset(IterableDataset):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            iter_start = self.start
            iter_end = self.end
        else:
            per_worker = int(
                math.ceil((self.end - self.start) / float(
                    worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)

        for i in range(iter_start, iter_end):
            yield np.array([i])

dataset = SplitedIterableDataset(start=2, end=9)
dataloader = DataLoader(
    dataset,
    num_workers=2,
    batch_size=1,
    drop_last=True)

for data in dataloader:
    print(data)
    # outputs: [2, 5, 3, 6, 4, 7]