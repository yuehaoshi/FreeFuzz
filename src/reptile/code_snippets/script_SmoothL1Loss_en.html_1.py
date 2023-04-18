import paddle
input = paddle.rand([3, 3]).astype("float32")
label = paddle.rand([3, 3]).astype("float32")
loss = paddle.nn.SmoothL1Loss()
output = loss(input, label)
print(output)
# [0.049606]