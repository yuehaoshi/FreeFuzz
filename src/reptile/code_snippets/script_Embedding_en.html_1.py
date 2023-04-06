import paddle
import numpy as np

x_data = np.arange(3, 6).reshape((3, 1)).astype(np.int64)
y_data = np.arange(6, 12).reshape((3, 2)).astype(np.float32)

x = paddle.to_tensor(x_data, stop_gradient=False)
y = paddle.to_tensor(y_data, stop_gradient=False)

embedding = paddle.nn.Embedding(10, 3, sparse=True)

w0=np.full(shape=(10, 3), fill_value=2).astype(np.float32)
embedding.weight.set_value(w0)

adam = paddle.optimizer.Adam(parameters=[embedding.weight], learning_rate=0.01)
adam.clear_grad()

# weight.shape = [10, 3]

# x.data = [[3],[4],[5]]
# x.shape = [3, 1]

# out.data = [[2,2,2], [2,2,2], [2,2,2]]
# out.shape = [3, 1, 3]
out=embedding(x)
out.backward()
adam.step()