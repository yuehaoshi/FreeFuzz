import paddle

input = paddle.rand([2,3,6,10], dtype="float32")
upsample_out = paddle.nn.Upsample(size=[12,12])

output = upsample_out(x=input)
print(output.shape)
# [2, 3, 12, 12]