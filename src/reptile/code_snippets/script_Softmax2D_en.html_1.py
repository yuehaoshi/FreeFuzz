import paddle

x = paddle.rand([1, 2, 3, 4])
# [[[[0.42496058 0.1172187  0.14664008 0.8151267 ]
#    [0.24430142 0.42052492 0.60372984 0.79307914]
#    [0.4539401  0.90458065 0.10235776 0.62009853]]

#   [[0.11731581 0.16053623 0.05667042 0.91876775]
#    [0.9413854  0.30770817 0.6788164  0.9543593 ]
#    [0.4145064  0.75909156 0.11598814 0.73599935]]]]
m = paddle.nn.Softmax2D()
out = m(x)
# [[[[0.5763103  0.48917228 0.5224772  0.4741129 ]
#    [0.3324591  0.5281743  0.48123717 0.45976716]
#    [0.5098571  0.5363083  0.49659243 0.4710572 ]]

#   [[0.42368975 0.51082766 0.47752273 0.5258871 ]
#    [0.66754097 0.47182566 0.5187628  0.5402329 ]
#    [0.49014282 0.46369177 0.50340754 0.5289428 ]]]]