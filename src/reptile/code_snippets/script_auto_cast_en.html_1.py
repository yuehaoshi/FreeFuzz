import paddle

conv2d = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
data = paddle.rand([10, 3, 32, 32])

with paddle.amp.auto_cast():
    conv = conv2d(data)
    print(conv.dtype) # FP16

with paddle.amp.auto_cast(enable=False):
    conv = conv2d(data)
    print(conv.dtype) # FP32

with paddle.amp.auto_cast(custom_black_list={'conv2d'}):
    conv = conv2d(data)
    print(conv.dtype) # FP32

a = paddle.rand([2,3])
b = paddle.rand([2,3])
with paddle.amp.auto_cast(custom_white_list={'elementwise_add'}):
    c = a + b
    print(c.dtype) # FP16

with paddle.amp.auto_cast(custom_white_list={'elementwise_add'}, level='O2'):
    d = a + b
    print(d.dtype) # FP16