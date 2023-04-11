import paddle

x = paddle.to_tensor([[0, 2, 3], [1, 4, 5], [2, 6, 7]], dtype="float32")
indexes = paddle.to_tensor([[0, 1], [1, 2], [2, 1], [0, 0]], dtype="int32")
src_index = indexes[:, 0]
dst_index = indexes[:, 1]
out = paddle.incubate.graph_send_recv(x, src_index, dst_index, pool_type="sum")
# Outputs: [[0., 2., 3.], [2., 8., 10.], [1., 4., 5.]]