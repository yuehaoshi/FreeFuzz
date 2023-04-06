import paddle

paddle.enable_static()

data = paddle.static.data(name="data", shape=[3, 32, 32], dtype="float32")

fc = paddle.static.nn.fc(x=data,
                         size=1000,
                         weight_attr=paddle.static.WeightNormParamAttr(
                             dim=None,
                             name='weight_norm_param',
                             initializer=paddle.nn.initializer.Constant(1.0),
                             learning_rate=1.0,
                             regularizer=paddle.regularizer.L2Decay(0.1),
                             trainable=True,
                             do_model_average=False,
                             need_clip=True))