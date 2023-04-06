import paddle
import numpy as np

paddle.enable_static()

# Build the model
startup_prog = paddle.static.default_startup_program()
main_prog = paddle.static.default_main_program()
with paddle.static.program_guard(main_prog, startup_prog):
    image = paddle.static.data(name="img", shape=[64, 784])
    w = paddle.create_parameter(shape=[784, 200], dtype='float32')
    b = paddle.create_parameter(shape=[200], dtype='float32')
    hidden_w = paddle.matmul(x=image, y=w)
    hidden_b = paddle.add(hidden_w, b)
exe = paddle.static.Executor(paddle.CPUPlace())
exe.run(startup_prog)

# Save the inference model
path_prefix = "./infer_model"
paddle.static.save_inference_model(path_prefix, [image], [hidden_b], exe)

[inference_program, feed_target_names, fetch_targets] = (
    paddle.static.load_inference_model(path_prefix, exe))
tensor_img = np.array(np.random.random((64, 784)), dtype=np.float32)
results = exe.run(inference_program,
              feed={feed_target_names[0]: tensor_img},
              fetch_list=fetch_targets)

# In this example, the inference program was saved in file
# "./infer_model.pdmodel" and parameters were saved in file
# " ./infer_model.pdiparams".
# By the inference program, feed_target_names and
# fetch_targets, we can use an executor to run the inference
# program to get the inference result.