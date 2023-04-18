paddle.enable_static()

  x = paddle.static.data(name='x', shape=[None, 2, 8, 8], dtype='float32')
  x.stop_gradient=False
  y = paddle.static.nn.conv2d(x, 4, 1, bias_attr=False)
  y = F.relu(y)
  z = paddle.static.gradients([y], x)
  print(z) # [var x@GRAD : LOD_TENSOR.shape(-1, 2, 8, 8).dtype(float32).stop_gradient(False)]