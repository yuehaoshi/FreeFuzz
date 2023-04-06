import paddle

class MyLayer(paddle.nn.Layer):
    def __init__(self):
        super(MyLayer, self).__init__()
        self._linear = paddle.nn.Linear(1, 1)
        self._dropout = paddle.nn.Dropout(p=0.5)

    def forward(self, input):
        temp = self._linear(input)
        temp = self._dropout(temp)
        return temp

mylayer = MyLayer()
print(mylayer.sublayers())  # [<paddle.nn.layer.common.Linear object at 0x7f44b58977d0>, <paddle.nn.layer.common.Dropout object at 0x7f44b58978f0>]