import numpy as np
#input:
x = np.array([[1, 1], [2, 2], [3, 3]])
index = np.array([2, 1, 0, 1])
# shape of updates should be the same as x
# shape of updates with dim > 1 should be the same as input
updates = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
overwrite = False
# calculation:
if not overwrite:
    for i in range(len(index)):
        x[index[i]] = np.zeros((2))
for i in range(len(index)):
    if (overwrite):
        x[index[i]] = updates[i]
    else:
        x[index[i]] += updates[i]
# output:
out = np.array([[3, 3], [6, 6], [1, 1]])
out.shape # [3, 2]