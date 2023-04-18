import paddle
import numpy

paddle.static.global_scope().var("data").get_tensor().set(numpy.ones((2, 2)), paddle.CPUPlace())
numpy.array(paddle.static.global_scope().find_var("data").get_tensor())