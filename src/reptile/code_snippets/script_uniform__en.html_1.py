import paddle
# example:
x = paddle.ones(shape=[3, 4])
x.uniform_()
print(x)
# [[ 0.84524226,  0.6921872,   0.56528175,  0.71690357], # random
#  [-0.34646994, -0.45116323, -0.09902662, -0.11397249], # random
#  [ 0.433519,    0.39483607, -0.8660099,   0.83664286]] # random