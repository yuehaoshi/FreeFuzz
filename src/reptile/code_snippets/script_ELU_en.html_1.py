import paddle

x = paddle.to_tensor([[-1. ,6.], [1., 15.6]])
m = paddle.nn.ELU(0.2)
out = m(x)
# [[-0.12642411  6.        ]
#  [ 1.          15.6      ]]