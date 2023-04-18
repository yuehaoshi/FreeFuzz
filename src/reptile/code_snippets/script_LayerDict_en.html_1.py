import paddle
import numpy as np
from collections import OrderedDict

sublayers = OrderedDict([
    ('conv1d', paddle.nn.Conv1D(3, 2, 3)),
    ('conv2d', paddle.nn.Conv2D(3, 2, 3)),
    ('conv3d', paddle.nn.Conv3D(4, 6, (3, 3, 3))),
])

layers_dict = paddle.nn.LayerDict(sublayers=sublayers)

l = layers_dict['conv1d']

for k in layers_dict:
    l = layers_dict[k]

len(layers_dict)
#3

del layers_dict['conv2d']
len(layers_dict)
#2

conv1d = layers_dict.pop('conv1d')
len(layers_dict)
#1

layers_dict.clear()
len(layers_dict)
#0