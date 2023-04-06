import paddle
paddle.enable_static()

x = paddle.ones(shape=[2, 3, 5])
x_T = x.T

exe = paddle.static.Executor()
x_T_np = exe.run(paddle.static.default_main_program(), fetch_list=[x_T])[0]
print(x_T_np.shape)
# (5, 3, 2)