import paddle
linear = paddle.nn.Linear(10, 10)

adam = paddle.optimizer.Adam(0.1, parameters=linear.parameters())

# set learning rate manually by python float value
lr_list = [0.2, 0.3, 0.4, 0.5, 0.6]
for i in range(5):
    adam.set_lr(lr_list[i])
    lr = adam.get_lr()
    print("current lr is {}".format(lr))
# Print:
#    current lr is 0.2
#    current lr is 0.3
#    current lr is 0.4
#    current lr is 0.5
#    current lr is 0.6