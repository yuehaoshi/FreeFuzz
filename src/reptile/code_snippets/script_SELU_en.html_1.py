import paddle

x = paddle.to_tensor([[0.0, 1.0],[2.0, 3.0]])
m = paddle.nn.SELU()
out = m(x)
print(out)
# [[0, 1.050701],[2.101402, 3.152103]]