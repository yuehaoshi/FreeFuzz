import paddle
import numpy as np

# First create the Executor.
paddle.enable_static()
place = paddle.CUDAPlace(0)
exe = paddle.static.Executor(place)

data = paddle.static.data(name='X', shape=[None, 1], dtype='float32')
class_dim = 2
prediction = paddle.static.nn.fc(data, class_dim)
loss = paddle.mean(prediction)
adam = paddle.optimizer.Adam()
adam.minimize(loss)

# Run the startup program once and only once.
exe.run(paddle.static.default_startup_program())
build_strategy = paddle.static.BuildStrategy()
binary = paddle.static.CompiledProgram(
    paddle.static.default_main_program()).with_data_parallel(
        loss_name=loss.name, build_strategy=build_strategy)
batch_size = 6
x = np.random.random(size=(batch_size, 1)).astype('float32')

# Set return_merged as False to fetch unmerged results:
unmerged_prediction, = exe.run(binary,
                               feed={'X': x},
                               fetch_list=[prediction.name],
                               return_merged=False)
# If the user uses two GPU cards to run this python code, the printed result will be
# (2, 3, class_dim). The first dimension value of the printed result is the number of used
# GPU cards, and the second dimension value is the quotient of batch_size and the
# number of used GPU cards.
print("The unmerged prediction shape: {}".format(
    np.array(unmerged_prediction).shape))
print(unmerged_prediction)

# Set return_merged as True to fetch merged results:
merged_prediction, = exe.run(binary,
                             feed={'X': x},
                             fetch_list=[prediction.name],
                             return_merged=True)
# If the user uses two GPU cards to run this python code, the printed result will be
# (6, class_dim). The first dimension value of the printed result is the batch_size.
print("The merged prediction shape: {}".format(
    np.array(merged_prediction).shape))
print(merged_prediction)

# Out:
# The unmerged prediction shape: (2, 3, 2)
# [array([[-0.37620035, -0.19752218],
#        [-0.3561043 , -0.18697084],
#        [-0.24129935, -0.12669306]], dtype=float32), array([[-0.24489994, -0.12858354],
#        [-0.49041364, -0.25748932],
#        [-0.44331917, -0.23276259]], dtype=float32)]
# The merged prediction shape: (6, 2)
# [[-0.37789783 -0.19921964]
#  [-0.3577645  -0.18863106]
#  [-0.24274671 -0.12814042]
#  [-0.24635398 -0.13003758]
#  [-0.49232286 -0.25939852]
#  [-0.44514108 -0.2345845 ]]