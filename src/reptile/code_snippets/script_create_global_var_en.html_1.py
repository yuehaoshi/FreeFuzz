import paddle
paddle.enable_static()
var = paddle.static.create_global_var(shape=[2,3], value=1.0, dtype='float32',
                               persistable=True, force_cpu=True, name='new_var')