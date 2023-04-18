import paddle
import paddle.nn as nn

net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))

def init_weights(layer):
    if type(layer) == nn.Linear:
        print('before init weight:', layer.weight.numpy())
        new_weight = paddle.full(shape=layer.weight.shape, dtype=layer.weight.dtype, fill_value=0.9)
        layer.weight.set_value(new_weight)
        print('after init weight:', layer.weight.numpy())

net.apply(init_weights)

print(net.state_dict())