import paddle
import numpy as np

image_shape=(3, 2, 2)
x = np.arange(image_shape[0] * image_shape[1] * image_shape[2]).reshape(image_shape)
x = x.astype('float32')
img = paddle.to_tensor(x)
tmp = paddle.flip(img, [0,1])
print(tmp) # [[[10,11],[8, 9]], [[6, 7],[4, 5]], [[2, 3],[0, 1]]]

out = paddle.flip(tmp,-1)
print(out) # [[[11,10],[9, 8]], [[7, 6],[5, 4]], [[3, 2],[1, 0]]]