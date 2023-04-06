# required: skip
import paddle

linear=paddle.nn.Linear(2, 2)
linear.weight
#Parameter containing:
#Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=False,
#       [[-0.32770029,  0.38653070],
#        [ 0.46030545,  0.08158520]])

linear.to(dtype='float64')
linear.weight
#Tenor(shape=[2, 2], dtype=float64, place=CUDAPlace(0), stop_gradient=False,
#       [[-0.32770029,  0.38653070],
#        [ 0.46030545,  0.08158520]])

linear.to(device='cpu')
linear.weight
#Tensor(shape=[2, 2], dtype=float64, place=CPUPlace, stop_gradient=False,
#       [[-0.32770029,  0.38653070],
#        [ 0.46030545,  0.08158520]])
linear.to(device=paddle.CUDAPinnedPlace(), blocking=False)
linear.weight
#Tensor(shape=[2, 2], dtype=float64, place=CUDAPinnedPlace, stop_gradient=False,
#       [[-0.04989364, -0.56889004],
#        [ 0.33960250,  0.96878713]])