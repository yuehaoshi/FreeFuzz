import paddle
paddle.set_device('cpu')

input = paddle.uniform([2, 3])
# [[-0.2820413   0.9528898  -0.81638825] # random
#  [-0.6733154  -0.33866507  0.25770962]] # random
label = paddle.to_tensor([0, 1, 4, 5])
m = paddle.nn.HSigmoidLoss(3, 5)
out = m(input, label)
# [[2.4543471]
#  [1.9359267]]