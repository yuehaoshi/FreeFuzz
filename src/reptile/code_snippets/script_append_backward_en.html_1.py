import paddle
import paddle.nn.functional as F

paddle.enable_static()

x = paddle.static.data(name='x', shape=[None, 13], dtype='int64')
y = paddle.static.data(name='y', shape=[None, 1], dtype='float32')
x_emb = paddle.static.nn.embedding(x, size=[100, 256])
y_predict = paddle.static.nn.fc(x=x_emb, size=1, activation=None, name='my_fc')
loss = F.square_error_cost(input=y_predict, label=y)
avg_loss = paddle.mean(loss)

# Get all weights in main_program, not include bias.
all_weights = [param for param in paddle.static.default_main_program().block(0).all_parameters() if 'w_' in param.name]
all_weights_name = [w.name for w in all_weights]

# return all param_grads needed to be updated if parameter_list set default None.
p_g_list1 = paddle.static.append_backward(loss=avg_loss)
# output: [(embedding_0.w_0, embedding_0.w_0@GRAD), (my_fc.w_0, my_fc.w_0@GRAD), (my_fc.b_0, my_fc.b_0@GRAD)]

# return the param_grads corresponding to parameter_list that can be list of param (Tensor).
p_g_list2 = paddle.static.append_backward(loss=avg_loss, parameter_list=all_weights)
# output: [(embedding_0.w_0, embedding_0.w_0@GRAD), (my_fc.w_0, my_fc.w_0@GRAD)]

# parameter_list can be list of param.name (str).
p_g_list3 = paddle.static.append_backward(loss=avg_loss, parameter_list=all_weights_name)
# output: [(embedding_0.w_0, embedding_0.w_0@GRAD), (my_fc.w_0, my_fc.w_0@GRAD)]

# no_grad_set can be set of Tensors that means grad will be cut off from these Tensors.
p_g_list4 = paddle.static.append_backward(loss=avg_loss, no_grad_set=set([x_emb]))
# output: [(my_fc.w_0, my_fc.w_0@GRAD), (my_fc.b_0, my_fc.b_0@GRAD)]

# no_grad_set can be set of Tensor.name when the Tensor is created inside layers and can't be specified explicitly.
p_g_list5 = paddle.static.append_backward(loss=avg_loss, no_grad_set=set(['my_fc.b_0']))
# output: [(embedding_0.w_0, embedding_0.w_0@GRAD), (my_fc.w_0, my_fc.w_0@GRAD)]

# return [] because all param_grads are filtered by no_grad_set.
p_g_list6 = paddle.static.append_backward(loss=avg_loss, parameter_list=all_weights, no_grad_set=set(all_weights))