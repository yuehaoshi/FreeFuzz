import paddle

x = paddle.to_tensor([0.9383, 0.1983, 3.2, 1.2])
y = paddle.to_tensor([1.0, 1.0, 1.0, 1.0])
out = paddle.where(x>1, x, y)

print(out)
#out: [1.0, 1.0, 3.2, 1.2]