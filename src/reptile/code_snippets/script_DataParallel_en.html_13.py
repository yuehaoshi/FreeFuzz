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

x = paddle.randn([10, 1], 'float32')
mylayer = MyLayer()
mylayer.eval()  # set mylayer._dropout to eval mode
out = mylayer(x)
print(out)