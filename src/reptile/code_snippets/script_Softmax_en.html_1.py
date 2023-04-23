import paddle

x = paddle.to_tensor([[[2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 8.0, 9.0]],
            [[1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [6.0, 7.0, 8.0, 9.0]]], dtype='float32')
m = paddle.nn.Softmax()
out = m(x)
# [[[0.0320586 , 0.08714432, 0.23688282, 0.64391426],
#   [0.0320586 , 0.08714432, 0.23688282, 0.64391426],
#   [0.07232949, 0.19661193, 0.19661193, 0.53444665]],
# [[0.0320586 , 0.08714432, 0.23688282, 0.64391426],
#   [0.0320586 , 0.08714432, 0.23688282, 0.64391426],
#   [0.0320586 , 0.08714432, 0.23688282, 0.64391426]]]