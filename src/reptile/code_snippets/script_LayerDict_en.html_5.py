import paddle
from collections import OrderedDict

sublayers = OrderedDict([
    ('conv1d', paddle.nn.Conv1D(3, 2, 3)),
    ('conv2d', paddle.nn.Conv2D(3, 2, 3)),
    ('conv3d', paddle.nn.Conv3D(4, 6, (3, 3, 3))),
])

layer_dict = paddle.nn.LayerDict(sublayers=sublayers)
for k, v in layer_dict.items():
    print(k, ":", v)

#conv1d : Conv1D(3, 2, kernel_size=[3], data_format=NCL)
#conv2d : Conv2D(3, 2, kernel_size=[3, 3], data_format=NCHW)
#conv3d : Conv3D(4, 6, kernel_size=[3, 3, 3], data_format=NCDHW)