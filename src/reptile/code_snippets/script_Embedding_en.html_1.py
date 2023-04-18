import paddle

x = paddle.to_tensor([[0], [1], [3]], dtype="int64", stop_gradient=False)
embedding = paddle.nn.Embedding(4, 3, sparse=True)

w0 = paddle.to_tensor([[0., 0., 0.],
                    [1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]], dtype="float32")
embedding.weight.set_value(w0)
print(embedding.weight)
# Tensor(shape=[4, 3], dtype=float32, place=Place(gpu:0), stop_gradient=False,
#        [[0., 0., 0.],
#         [1., 1., 1.],
#         [2., 2., 2.],
#         [3., 3., 3.]])

adam = paddle.optimizer.Adam(parameters=[embedding.weight], learning_rate=0.01)
adam.clear_grad()


out = embedding(x)
print(out)
# Tensor(shape=[3, 1, 3], dtype=float32, place=Place(gpu:0), stop_gradient=False,
#        [[[0., 0., 0.]],
#         [[1., 1., 1.]],
#         [[3., 3., 3.]]])

out.backward()
adam.step()