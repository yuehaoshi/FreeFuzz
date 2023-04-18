import paddle

class MyLayer(paddle.nn.Layer):
    def __init__(self):
        super(MyLayer, self).__init__()
        self._linear = paddle.nn.Linear(1, 1)
        w_tmp = self.create_parameter([1,1])
        self.add_parameter("w_tmp", w_tmp)

    def forward(self, input):
        return self._linear(input)

mylayer = MyLayer()
for name, param in mylayer.named_parameters():
    print(name, param)      # will print w_tmp,_linear.weight,_linear.bias