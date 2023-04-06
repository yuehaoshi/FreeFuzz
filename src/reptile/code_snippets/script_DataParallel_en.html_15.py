import paddle

class LinearNet(paddle.nn.Layer):
    def __init__(self):
        super(LinearNet, self).__init__(name_scope = "demo_linear_net")
        self._linear = paddle.nn.Linear(1, 1)

    def forward(self, x):
        return self._linear(x)

linear_net = LinearNet()
print(linear_net.full_name())   # demo_linear_net_0