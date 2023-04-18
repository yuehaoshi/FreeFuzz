import paddle
import numpy as np
paddle.enable_static()

data = paddle.static.data(name="input", shape=[-1, 32,32], dtype="float32")
label = paddle.static.data(name="label", shape=[-1], dtype="int")
fc_out = paddle.static.nn.fc(x=data, size=2)
predict = paddle.nn.functional.softmax(x=fc_out)
result=paddle.static.auc(input=predict, label=label)

place = paddle.CPUPlace()
exe = paddle.static.Executor(place)

exe.run(paddle.static.default_startup_program())
x = np.random.rand(3,32,32).astype("float32")
y = np.array([1,0,1])
output= exe.run(feed={"input": x,"label": y},
                 fetch_list=[result[0]])
print(output)

#you can learn the usage of ins_tag_weight by the following code.
'''
import paddle
import numpy as np
paddle.enable_static()

data = paddle.static.data(name="input", shape=[-1, 32,32], dtype="float32")
label = paddle.static.data(name="label", shape=[-1], dtype="int")
ins_tag_weight = paddle.static.data(name='ins_tag', shape=[-1,16], lod_level=0, dtype='float64')
fc_out = paddle.static.nn.fc(x=data, size=2)
predict = paddle.nn.functional.softmax(x=fc_out)
result=paddle.static.auc(input=predict, label=label, ins_tag_weight=ins_tag_weight)

place = paddle.CPUPlace()
exe = paddle.static.Executor(place)

exe.run(paddle.static.default_startup_program())
x = np.random.rand(3,32,32).astype("float32")
y = np.array([1,0,1])
z = np.array([1,0,1])
output= exe.run(feed={"input": x,"label": y, "ins_tag_weight":z},
                 fetch_list=[result[0]])
print(output)
'''