import paddle

m = paddle.nn.Sigmoid()
x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
out = m(x) # [0.7310586, 0.880797, 0.95257413, 0.98201376]