import paddle
from paddle.text.datasets import WMT14

class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()

    def forward(self, src_ids, trg_ids, trg_ids_next):
        return paddle.sum(src_ids), paddle.sum(trg_ids), paddle.sum(trg_ids_next)

wmt14 = WMT14(mode='train', dict_size=50)

for i in range(10):
    src_ids, trg_ids, trg_ids_next = wmt14[i]
    src_ids = paddle.to_tensor(src_ids)
    trg_ids = paddle.to_tensor(trg_ids)
    trg_ids_next = paddle.to_tensor(trg_ids_next)

    model = SimpleNet()
    src_ids, trg_ids, trg_ids_next = model(src_ids, trg_ids, trg_ids_next)
    print(src_ids.numpy(), trg_ids.numpy(), trg_ids_next.numpy())