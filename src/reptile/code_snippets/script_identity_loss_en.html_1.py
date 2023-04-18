import paddle.fluid as fluid
import paddle
paddle.enable_static()
loss = fluid.data(name="loss", shape=[-1, 1], dtype="float32")
out = paddle.incubate.identity_loss(loss, reduction=1)