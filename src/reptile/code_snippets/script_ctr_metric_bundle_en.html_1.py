import paddle
paddle.enable_static()
data = paddle.static.data(name="data", shape=[32, 32], dtype="float32")
label = paddle.static.data(name="label", shape=[-1, 1], dtype="int32")
predict = paddle.nn.functional.sigmoid(paddle.static.nn.fc(input=data, size=1))
auc_out = paddle.static.ctr_metric_bundle(input=predict, label=label)