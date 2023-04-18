import paddle

x = paddle.rand((2, 2, 3))
instance_norm = paddle.nn.InstanceNorm1D(2)
instance_norm_out = instance_norm(x)

print(instance_norm_out)