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

# Feed data and train process

# Save inference model. Note we don't save label and loss in this example
paddle.static.save_inference_model(path_prefix, [image], [predict], exe)

# In this example, the save_inference_mode inference will prune the default
# main program according to the network's input node (img) and output node(predict).
# The pruned inference program is going to be saved in file "./infer_model.pdmodel"
# and parameters are going to be saved in file "./infer_model.pdiparams".