import paddle

paddle.enable_static()
# Sample Network:
x = paddle.static.data(name='x', shape=[100, 100], dtype='float32')
y = paddle.static.data(name='x', shape=[100, 100], dtype='float32')
out = paddle.add(x, y)

#print the number of blocks in the program, 1 in this case
print(paddle.static.default_main_program().num_blocks) # 1
#print the default_main_program
print(paddle.static.default_main_program())