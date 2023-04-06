import paddle
import numpy

layer1 = numpy.random.random((5, 5)).astype('float32')
layer2 = numpy.random.random((5, 4)).astype('float32')
bilinear = paddle.nn.Bilinear(
    in1_features=5, in2_features=4, out_features=1000)
result = bilinear(paddle.to_tensor(layer1),
                paddle.to_tensor(layer2))     # result shape [5, 1000]