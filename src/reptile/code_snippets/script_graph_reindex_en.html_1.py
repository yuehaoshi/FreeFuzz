import paddle

x = [0, 1, 2]
neighbors_e1 = [8, 9, 0, 4, 7, 6, 7]
count_e1 = [2, 3, 2]
x = paddle.to_tensor(x, dtype="int64")
neighbors_e1 = paddle.to_tensor(neighbors_e1, dtype="int64")
count_e1 = paddle.to_tensor(count_e1, dtype="int32")

reindex_src, reindex_dst, out_nodes =                 paddle.incubate.graph_reindex(x, neighbors_e1, count_e1)
# reindex_src: [3, 4, 0, 5, 6, 7, 6]
# reindex_dst: [0, 0, 1, 1, 1, 2, 2]
# out_nodes: [0, 1, 2, 8, 9, 4, 7, 6]

neighbors_e2 = [0, 2, 3, 5, 1]
count_e2 = [1, 3, 1]
neighbors_e2 = paddle.to_tensor(neighbors_e2, dtype="int64")
count_e2 = paddle.to_tensor(count_e2, dtype="int32")

neighbors = paddle.concat([neighbors_e1, neighbors_e2])
count = paddle.concat([count_e1, count_e2])
reindex_src, reindex_dst, out_nodes =                 paddle.incubate.graph_reindex(x, neighbors, count)
# reindex_src: [3, 4, 0, 5, 6, 7, 6, 0, 2, 8, 9, 1]
# reindex_dst: [0, 0, 1, 1, 1, 2, 2, 0, 1, 1, 1, 2]
# out_nodes: [0, 1, 2, 8, 9, 4, 7, 6, 3, 5]