import paddle

x = paddle.arange(12).reshape([3, 4])
# x is [[0 , 1 , 2 , 3 ],
#       [4 , 5 , 6 , 7 ],
#       [8 , 9 , 10, 11]]

y1 = paddle.median(x)
# y1 is [5.5]

y2 = paddle.median(x, axis=0)
# y2 is [4., 5., 6., 7.]

y3 = paddle.median(x, axis=1)
# y3 is [1.5, 5.5, 9.5]

y4 = paddle.median(x, axis=0, keepdim=True)
# y4 is [[4., 5., 6., 7.]]