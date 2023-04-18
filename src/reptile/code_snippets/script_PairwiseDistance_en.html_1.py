import paddle
x = paddle.to_tensor([[1., 3.], [3., 5.]], dtype=paddle.float64)
y = paddle.to_tensor([[5., 6.], [7., 8.]], dtype=paddle.float64)
dist = paddle.nn.PairwiseDistance()
distance = dist(x, y)
print(distance.numpy()) # [5. 5.]