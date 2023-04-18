import paddle
new_parameter = paddle.static.create_parameter(name="X",
                                    shape=[10, 23, 48],
                                    dtype='float32')
if new_parameter.is_parameter:
    print("Current var is a Parameter")
else:
    print("Current var is not a Parameter")

# Current var is a Parameter