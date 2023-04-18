import paddle
paddle.set_device('cpu')
paddle.seed(100)

x = paddle.empty([2,3])
x.exponential_()
# [[0.80643415, 0.23211166, 0.01169797],
#  [0.72520673, 0.45208144, 0.30234432]]