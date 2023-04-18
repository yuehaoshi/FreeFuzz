import paddle

paddle.enable_static()

path_prefix = "./infer_model"

# User defined network, here a softmax regession example
image = paddle.static.data(name='img', shape=[None, 28, 28], dtype='float32')
label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
predict = paddle.static.nn.fc(image, 10, activation='softmax')

loss = paddle.nn.functional.cross_entropy(predict, label)

exe = paddle.static.Executor(paddle.CPUPlace())
exe.run(paddle.static.default_startup_program())

# serialize the default main program to bytes.
serialized_program = paddle.static.serialize_program([image], [predict])

# deserialize bytes to program
deserialized_program = paddle.static.deserialize_program(serialized_program)