import paddle
import paddle.nn as nn
import numpy as np

np.random.seed(0)
x1 = np.random.rand(2,3)
x2 = np.random.rand(2,3)
x1 = paddle.to_tensor(x1)
x2 = paddle.to_tensor(x2)

cos_sim_func = nn.CosineSimilarity(axis=0)
result = cos_sim_func(x1, x2)
print(result)
# [0.99806249 0.9817672  0.94987036]