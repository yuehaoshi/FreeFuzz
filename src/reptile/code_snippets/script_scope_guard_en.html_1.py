import paddle
import numpy
paddle.enable_static()

new_scope = paddle.static.Scope()
with paddle.static.scope_guard(new_scope):
     paddle.static.global_scope().var("data").get_tensor().set(numpy.ones((2, 2)), paddle.CPUPlace())
numpy.array(new_scope.find_var("data").get_tensor())