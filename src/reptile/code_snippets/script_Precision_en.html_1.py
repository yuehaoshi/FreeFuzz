import numpy as np
import paddle

x = np.array([0.1, 0.5, 0.6, 0.7])
y = np.array([0, 1, 1, 1])

m = paddle.metric.Precision()
m.update(x, y)
res = m.accumulate()
print(res) # 1.0