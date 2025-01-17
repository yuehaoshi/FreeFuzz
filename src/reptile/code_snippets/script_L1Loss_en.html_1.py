import paddle

input = paddle.to_tensor([[1.5, 0.8], [0.2, 1.3]])
label = paddle.to_tensor([[1.7, 1], [0.4, 0.5]])

l1_loss = paddle.nn.L1Loss()
output = l1_loss(input, label)
print(output.numpy())
# [0.35]

l1_loss = paddle.nn.L1Loss(reduction='sum')
output = l1_loss(input, label)
print(output.numpy())
# [1.4]

l1_loss = paddle.nn.L1Loss(reduction='none')
output = l1_loss(input, label)
print(output)
# [[0.20000005 0.19999999]
# [0.2        0.79999995]]