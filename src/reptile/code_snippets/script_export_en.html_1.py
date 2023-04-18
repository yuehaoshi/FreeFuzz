import paddle

class LinearNet(paddle.nn.Layer):
    def __init__(self):
        super(LinearNet, self).__init__()
        self._linear = paddle.nn.Linear(128, 10)

    def forward(self, x):
        return self._linear(x)

# Export model with 'InputSpec' to support dynamic input shape.
def export_linear_net():
    model = LinearNet()
    x_spec = paddle.static.InputSpec(shape=[None, 128], dtype='float32')
    paddle.onnx.export(model, 'linear_net', input_spec=[x_spec])

export_linear_net()

class Logic(paddle.nn.Layer):
    def __init__(self):
        super(Logic, self).__init__()

    def forward(self, x, y, z):
        if z:
            return x
        else:
            return y

# Export model with 'Tensor' to support pruned model by set 'output_spec'.
def export_logic():
    model = Logic()
    x = paddle.to_tensor([1])
    y = paddle.to_tensor([2])
    # Static and run model.
    paddle.jit.to_static(model)
    out = model(x, y, z=True)
    paddle.onnx.export(model, 'pruned', input_spec=[x], output_spec=[out])

export_logic()