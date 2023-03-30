import paddle


# arrange
out1 = paddle.arange(5)



# any
x = paddle.to_tensor([[1, 0], [1, 1]], dtype='int32')
x = paddle.assign(x)
print(x)
x = paddle.cast(x, 'bool')
# x is a bool Tensor with following elements:
#    [[True, False]
#     [True, True]]

# out1 should be [True]
out1 = paddle.any(x)  # [True]
print(out1)

# out2 should be [True, True]
out2 = paddle.any(x, axis=0)  # [True, True]
print(out2)

# keepdim=False, out3 should be [True, True], out.shape should be (2,)
out3 = paddle.any(x, axis=-1)  # [True, True]
print(out3)

# keepdim=True, result should be [[True], [True]], out.shape should be (2,1)
out4 = paddle.any(x, axis=1, keepdim=True)  # [[True], [True]]
print(out4)



# argmax
x = paddle.to_tensor([[5,8,9,5],
                     [0,0,1,7],
                     [6,9,2,4]])
out1 = paddle.argmax(x)
print(out1) # 2
out2 = paddle.argmax(x, axis=0)
print(out2)
# [2, 2, 0, 1]
out3 = paddle.argmax(x, axis=-1)
print(out3)
# [2, 3, 1]
out4 = paddle.argmax(x, axis=0, keepdim=True)
print(out4)
# [[2, 2, 0, 1]]



# argmin
x =  paddle.to_tensor([[5,8,9,5],
                         [0,0,1,7],
                         [6,9,2,4]])
out1 = paddle.argmin(x)
print(out1) # 4
out2 = paddle.argmin(x, axis=0)
print(out2)
# [1, 1, 1, 2]
out3 = paddle.argmin(x, axis=-1)
print(out3)
# [0, 0, 2]
out4 = paddle.argmin(x, axis=0, keepdim=True)
print(out4)
# [[1, 1, 1, 2]]



# argsort
x = paddle.to_tensor([[[5,8,9,5],
                       [0,0,1,7],
                       [6,9,2,4]],
                      [[5,2,4,2],
                       [4,7,7,9],
                       [1,7,0,6]]],
                    dtype='float32')
out1 = paddle.argsort(x, axis=-1)
out2 = paddle.argsort(x, axis=0)
out3 = paddle.argsort(x, axis=1)

print(out1)
#[[[0 3 1 2]
#  [0 1 2 3]
#  [2 3 0 1]]
# [[1 3 2 0]
#  [0 1 2 3]
#  [2 0 3 1]]]

print(out2)
#[[[0 1 1 1]
#  [0 0 0 0]
#  [1 1 1 0]]
# [[1 0 0 0]
#  [1 1 1 1]
#  [0 0 0 1]]]

print(out3)
#[[[1 1 1 2]
#  [0 0 2 0]
#  [2 2 0 1]]
# [[2 0 2 0]
#  [1 1 0 2]
#  [0 2 1 1]]]