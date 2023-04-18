import paddle


x = paddle.to_tensor([0., 1., 2., 3.])

chain = paddle.distribution.ChainTransform((
    paddle.distribution.AffineTransform(
        paddle.to_tensor(0.), paddle.to_tensor(1.)),
    paddle.distribution.ExpTransform()
))
print(chain.forward(x))
# Tensor(shape=[4], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [1.         , 2.71828175 , 7.38905621 , 20.08553696])
print(chain.inverse(chain.forward(x)))
# Tensor(shape=[4], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [0., 1., 2., 3.])
print(chain.forward_log_det_jacobian(x))
# Tensor(shape=[4], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [0., 1., 2., 3.])
print(chain.inverse_log_det_jacobian(chain.forward(x)))
# Tensor(shape=[4], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [ 0., -1., -2., -3.])