import paddle

row = [3, 7, 0, 9, 1, 4, 2, 9, 3, 9, 1, 9, 7]
colptr = [0, 2, 4, 5, 6, 7, 9, 11, 11, 13, 13]
nodes = [0, 8, 1, 2]
sample_sizes = [2, 2]
row = paddle.to_tensor(row, dtype="int64")
colptr = paddle.to_tensor(colptr, dtype="int64")
nodes = paddle.to_tensor(nodes, dtype="int64")

edge_src, edge_dst, sample_index, reindex_nodes = paddle.incubate.graph_khop_sampler(row, colptr, nodes, sample_sizes, False)