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

# serialize parameters to bytes.
serialized_params = paddle.static.serialize_persistables([image], [predict], exe)

# deserialize bytes to parameters.
main_program = paddle.static.default_main_program()
deserialized_params = paddle.static.deserialize_persistables(main_program, serialized_params, exe)