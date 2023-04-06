import paddle

x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
m = paddle.nn.Silu()
out = m(x) # [ 0.731059, 1.761594, 2.857722, 3.928055 ]