# required: gpu
# Demo1: single model and optimizer:
import paddle

model = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
optimzier = paddle.optimizer.SGD(parameters=model.parameters())

model, optimizer = paddle.amp.decorate(models=model, optimizers=optimzier, level='O2')

data = paddle.rand([10, 3, 32, 32])

with paddle.amp.auto_cast(enable=True, custom_white_list=None, custom_black_list=None, level='O2'):
    output = model(data)
    print(output.dtype) # FP16

# required: gpu
# Demo2: multi models and optimizers:
model2 = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
optimizer2 = paddle.optimizer.Adam(parameters=model2.parameters())

models, optimizers = paddle.amp.decorate(models=[model, model2], optimizers=[optimzier, optimizer2], level='O2')

data = paddle.rand([10, 3, 32, 32])

with paddle.amp.auto_cast(enable=True, custom_white_list=None, custom_black_list=None, level='O2'):
    output = models[0](data)
    output2 = models[1](data)
    print(output.dtype) # FP16
    print(output2.dtype) # FP16