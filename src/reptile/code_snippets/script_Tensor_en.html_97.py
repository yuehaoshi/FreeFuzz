import paddle
var = paddle.ones(shape=[4, 2, 3], dtype="float32")
print(var.inplace_version)  # 0

var[1] = 2.2
print(var.inplace_version)  # 1