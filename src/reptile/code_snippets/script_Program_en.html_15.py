import paddle
import paddle.static as static

paddle.enable_static()

program = static.default_main_program()
data = static.data(name='x', shape=[None, 13], dtype='float32')
hidden = static.nn.fc(x=data, size=10)
loss = paddle.mean(hidden)
paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)

for param in program.all_parameters():
    print(param)

# Here will print all parameters in current program, in this example,
# the result is like:
#
# persist trainable param fc_0.w_0 : LOD_TENSOR.shape(13, 10).dtype(float32).stop_gradient(False)
# persist trainable param fc_0.b_0 : LOD_TENSOR.shape(10,).dtype(float32).stop_gradient(False)
#
# Here print(param) will print out all the properties of a parameter,
# including name, type and persistable, you can access to specific
# property of a parameter, such as param.name, param.type