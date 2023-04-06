import paddle
paddle.enable_static()
with paddle.static.name_scope("s1"):
   a = paddle.static.data(name='data', shape=[None, 1], dtype='int32')
   b = a + 1
   with paddle.static.name_scope("s2"):
      c = b * 1
   with paddle.static.name_scope("s3"):
      d = c / 1
with paddle.static.name_scope("s1"):
      f = paddle.tensor.pow(d, 2.0)
with paddle.static.name_scope("s4"):
      g = f - 1

# Op are created in the default main program.
for op in paddle.static.default_main_program().block(0).ops:
    # elementwise_add is created in /s1/
    if op.type == 'elementwise_add':
        assert op.desc.attr("op_namescope") == '/s1/'
    # elementwise_mul is created in '/s1/s2'
    elif op.type == 'elementwise_mul':
        assert op.desc.attr("op_namescope") == '/s1/s2/'
    # elementwise_div is created in '/s1/s3'
    elif op.type == 'elementwise_div':
        assert op.desc.attr("op_namescope") == '/s1/s3/'
    # elementwise_sum is created in '/s4'
    elif op.type == 'elementwise_sub':
        assert op.desc.attr("op_namescope") == '/s4/'
    # pow is created in /s1_1/
    elif op.type == 'pow':
        assert op.desc.attr("op_namescope") == '/s1_1/'