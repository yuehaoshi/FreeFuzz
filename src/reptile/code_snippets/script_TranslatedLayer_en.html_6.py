import paddle

class MySequential(paddle.nn.Layer):
    def __init__(self, *layers):
        super(MySequential, self).__init__()
        if len(layers) > 0 and isinstance(layers[0], tuple):
            for name, layer in layers:
                self.add_sublayer(name, layer)
        else:
            for idx, layer in enumerate(layers):
                self.add_sublayer(str(idx), layer)

    def forward(self, input):
        for layer in self._sub_layers.values():
            input = layer(input)
        return input

fc1 = paddle.nn.Linear(10, 3)
fc2 = paddle.nn.Linear(3, 10, bias_attr=False)
model = MySequential(fc1, fc2)
for prefix, layer in model.named_sublayers():
    print(prefix, layer)