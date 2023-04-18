import paddle

class MyLinear(paddle.nn.Layer):
    def __init__(self,
                in_features,
                out_features):
        super(MyLinear, self).__init__()
        self.linear = paddle.nn.Linear( 10, 10)

        self.back_var = self.create_variable(name = "linear_tmp_0", dtype=self._dtype)

    def forward(self, input):
        out = self.linear(input)
        paddle.assign( out, self.back_var)

        return out