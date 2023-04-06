# from __future__ import print_function
# import math
# import multiprocessing
# import numpy
# import numpy as np
# import os
# import paddle
# import paddle.distributed as dist
# import paddle.distributed.fleet as fleet
# import paddle.fluid as fluid
# import paddle.nn as nn
# import paddle.nn.functional as F
# import paddle.optimizer as opt
# import paddle.static as static
# import paddle.utils as utils
# import paddle.vision.transforms as T
# import shutil
# import six
# import socket
# import tempfile
# from PIL import Image
# from collections import OrderedDict
# from contextlib import closing
# from io import BytesIO
# from paddle import Model
# from paddle import ParamAttr
# from paddle import nn
# from paddle.autograd import PyLayer
# from paddle.distributed import ReduceOp
# from paddle.distributed import init_parallel_env
# from paddle.distributed.fleet.utils.hybrid_parallel_util import fused_allreduce_gradients
# from paddle.distribution import Categorical
# from paddle.distribution import Normal
# from paddle.distribution import Uniform
# from paddle.fluid.dygraph import Linear
# from paddle.fluid.dygraph.base import to_variable
# from paddle.io import Dataset
# from paddle.io import Dataset, BatchSampler, DataLoader
# from paddle.io import Dataset, ComposeDataset
# from paddle.io import Dataset, DistributedBatchSampler
# from paddle.io import Dataset, RandomSampler
# from paddle.io import Dataset, Sampler
# from paddle.io import Dataset, SequenceSampler
# from paddle.io import IterableDataset
# from paddle.io import IterableDataset, ChainDataset
# from paddle.io import IterableDataset, DataLoader, get_worker_info
# from paddle.io import RandomSampler, BatchSampler, Dataset
# from paddle.io import Subset
# from paddle.io import TensorDataset
# from paddle.io import WeightedRandomSampler
# from paddle.io import random_split
# from paddle.jit import to_static
# from paddle.metric import Accuracy
# from paddle.nn import BeamSearchDecoder, dynamic_decode
# from paddle.nn import Conv1D
# from paddle.nn import Conv1DTranspose
# from paddle.nn import Conv2D
# from paddle.nn import CrossEntropyLoss
# from paddle.nn import GRUCell, Linear, Embedding
# from paddle.nn import Linear
# from paddle.nn import Transformer
# from paddle.nn import TransformerDecoderLayer
# from paddle.nn import TransformerDecoderLayer, TransformerDecoder
# from paddle.nn import TransformerEncoderLayer
# from paddle.nn import TransformerEncoderLayer, TransformerEncoder
# from paddle.nn.layer.loss import CrossEntropyLoss
# from paddle.nn.layer.transformer import Transformer
# from paddle.optimizer import Adam
# from paddle.regularizer import L1Decay
# from paddle.regularizer import L2Decay
# from paddle.signal import stft
# from paddle.signal import stft, istft
# from paddle.static import ExponentialMovingAverage
# from paddle.static import InputSpec
# from paddle.text.datasets import Conll05st
# from paddle.text.datasets import Imdb
# from paddle.text.datasets import Imikolov
# from paddle.text.datasets import Movielens
# from paddle.text.datasets import UCIHousing
# from paddle.text.datasets import WMT14
# from paddle.text.datasets import WMT16
# from paddle.vision import DatasetFolder
# from paddle.vision import get_image_backend
# from paddle.vision import image_load, set_image_backend
# from paddle.vision import set_image_backend
# from paddle.vision.datasets import MNIST
# from paddle.vision.models import LeNet

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.abs(x)
# print(out)

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.acos(x)
# print(out)

# x = paddle.to_tensor([2, 3, 4], 'float64')
# y = paddle.to_tensor([1, 5, 2], 'float64')
# z = paddle.add(x, y)
# print(z)  

# input0 = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='float32')
# input1 = paddle.to_tensor([[7, 8, 9], [10, 11, 12]], dtype='float32')
# output = paddle.add_n([input0, input1])

# x = paddle.ones([2,2])
# y = paddle.ones([2,2])
# input = paddle.ones([2,2])

# out = paddle.addmm( input=input, x=x, y=y, beta=0.5, alpha=5.0 )

# print(out)

# x = paddle.assign(np.array([[1, 0], [1, 1]], dtype='int32'))
# print(x)
# x = paddle.cast(x, 'bool')

# out1 = paddle.all(x)  
# print(out1)

# out2 = paddle.all(x, axis=0)  
# print(out2)

# out3 = paddle.all(x, axis=-1)  
# print(out3)

# out4 = paddle.all(x, axis=1, keepdim=True)
# out4 = paddle.cast(out4, 'int32')  
# print(out4)

# x = paddle.to_tensor([10000., 1e-07])
# y = paddle.to_tensor([10000.1, 1e-08])
# result1 = paddle.allclose(x, y, rtol=1e-05, atol=1e-08,
#                         equal_nan=False, name="ignore_nan")
# np_result1 = result1.numpy()

# result2 = paddle.allclose(x, y, rtol=1e-05, atol=1e-08,
#                             equal_nan=True, name="equal_nan")
# np_result2 = result2.numpy()

# x = paddle.to_tensor([1.0, float('nan')])
# y = paddle.to_tensor([1.0, float('nan')])
# result1 = paddle.allclose(x, y, rtol=1e-05, atol=1e-08,
#                         equal_nan=False, name="ignore_nan")
# np_result1 = result1.numpy()

# result2 = paddle.allclose(x, y, rtol=1e-05, atol=1e-08,
#                             equal_nan=True, name="equal_nan")
# np_result2 = result2.numpy()

# x = paddle.assign(np.array([[1, 0], [1, 1]], dtype='int32'))
# print(x)
# x = paddle.cast(x, 'bool')

# out1 = paddle.any(x)  
# print(out1)

# out2 = paddle.any(x, axis=0)  
# print(out2)

# out3 = paddle.any(x, axis=-1)  
# print(out3)

# out4 = paddle.any(x, axis=1, keepdim=True)
# out4 = paddle.cast(out4, 'int32')  
# print(out4)

# out1 = paddle.arange(5)

# out2 = paddle.arange(3, 9, 2.0)

# out3 = paddle.arange(4.999, dtype='float32')

# start_var = paddle.to_tensor([3])
# out4 = paddle.arange(start_var, 7)

# x =  paddle.to_tensor([[5,8,9,5],
#                          [0,0,1,7],
#                          [6,9,2,4]])
# out1 = paddle.argmax(x)
# print(out1) 
# out2 = paddle.argmax(x, axis=1)
# print(out2)

# out3 = paddle.argmax(x, axis=-1)
# print(out3)

# x =  paddle.to_tensor([[5,8,9,5],
#                          [0,0,1,7],
#                          [6,9,2,4]])
# out1 = paddle.argmin(x)
# print(out1) 
# out2 = paddle.argmin(x, axis=1)
# print(out2)

# out3 = paddle.argmin(x, axis=-1)
# print(out3)

# x = paddle.to_tensor([[[5,8,9,5],
#                        [0,0,1,7],
#                        [6,9,2,4]],
#                       [[5,2,4,2],
#                        [4,7,7,9],
#                        [1,7,0,6]]],
#                     dtype='float32')
# out1 = paddle.argsort(x=x, axis=-1)
# out2 = paddle.argsort(x=x, axis=0)
# out3 = paddle.argsort(x=x, axis=1)
# print(out1)

# print(out2)

# print(out3)

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.asin(x)
# print(out)

# data = paddle.full(shape=[3, 2], fill_value=2.5, dtype='float64') 
# array = np.array([[1, 1],
#                   [3, 4],
#                   [1, 3]]).astype(np.int64)
# result1 = paddle.zeros(shape=[3, 3], dtype='float32')
# paddle.assign(array, result1) 
# result2 = paddle.assign(data)  
# result3 = paddle.assign(np.array([[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]], dtype='float32')) 

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.atan(x)
# print(out)

# x = paddle.to_tensor([-1, +1, +1, -1]).astype('float32')

# y = paddle.to_tensor([-1, -1, +1, +1]).astype('float32')

# out = paddle.atan2(x, y)

# def reader():
#     for i in range(10):
#         yield i
# batch_reader = paddle.batch(reader, batch_size=2)

# for data in batch_reader():
#     print(data)

# paddle.set_device('cpu')  
# paddle.seed(100)

# x = paddle.rand([2,3])
# print(x)

# out = paddle.bernoulli(x)
# print(out)

# x = paddle.to_tensor([1, 2, 1, 4, 5])
# result1 = paddle.bincount(x)
# print(result1) 

# w = paddle.to_tensor([2.1, 0.4, 0.1, 0.5, 0.5])
# result2 = paddle.bincount(x, weights=w)
# print(result2) 

# x = paddle.to_tensor([-5, -1, 1])
# y = paddle.to_tensor([4,  2, -3])
# res = paddle.bitwise_and(x, y)
# print(res)  

# x = paddle.to_tensor([-5, -1, 1])
# res = paddle.bitwise_not(x)
# print(res) 

# x = paddle.to_tensor([-5, -1, 1])
# y = paddle.to_tensor([4,  2, -3])
# res = paddle.bitwise_or(x, y)
# print(res)  

# x = paddle.to_tensor([-5, -1, 1])
# y = paddle.to_tensor([4,  2, -3])
# res = paddle.bitwise_xor(x, y)
# print(res) 

# x = paddle.to_tensor([[[1.0, 1.0, 1.0],
#                     [2.0, 2.0, 2.0]],
#                     [[3.0, 3.0, 3.0],
#                     [4.0, 4.0, 4.0]]])
# y = paddle.to_tensor([[[1.0, 1.0],[2.0, 2.0],[3.0, 3.0]],
#                     [[4.0, 4.0],[5.0, 5.0],[6.0, 6.0]]])
# out = paddle.bmm(x, y)

# out_np = out.numpy()

# shape = paddle.broadcast_shape([2, 1, 3], [1, 3, 1])

# x1 = paddle.rand([1, 2, 3, 4]).astype('float32')
# x2 = paddle.rand([1, 2, 1, 4]).astype('float32')
# x3 = paddle.rand([1, 1, 3, 1]).astype('float32')
# out1, out2, out3 = paddle.broadcast_tensors(input=[x1, x2, x3])

# data = paddle.to_tensor([1, 2, 3], dtype='int32')
# out = paddle.broadcast_to(data, shape=[2, 3])
# print(out)

# x = paddle.to_tensor([2, 3, 4], 'float64')
# y = paddle.cast(x, 'uint8')

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.ceil(x)
# print(out)

# a = np.random.rand(3, 3)
# a_t = np.transpose(a, [1, 0])
# x_data = np.matmul(a, a_t) + 1e-03
# x = paddle.to_tensor(x_data)
# out = paddle.cholesky(x, upper=False)
# print(out)

# x_np = np.random.random([3, 9, 5]).astype("int32")
# x = paddle.to_tensor(x_np)

# out0, out1, out2 = paddle.chunk(x, chunks=3, axis=1)

# out0, out1, out2 = paddle.chunk(x, chunks=3, axis=-2)

# x1 = paddle.to_tensor([[1.2, 3.5], [4.5, 6.4]], 'float32')
# out1 = paddle.clip(x1, min=3.5, max=5.0)
# out2 = paddle.clip(x1, min=2.5)
# print(out1)

# print(out2)

# x1 = paddle.to_tensor([[1, 2, 3],
#                        [4, 5, 6]])
# x2 = paddle.to_tensor([[11, 12, 13],
#                        [14, 15, 16]])
# x3 = paddle.to_tensor([[21, 22],
#                        [23, 24]])
# zero = paddle.full(shape=[1], dtype='int32', fill_value=0)

# out1 = paddle.concat(x=[x1, x2, x3], axis=-1)
# out2 = paddle.concat(x=[x1, x2], axis=0)
# out3 = paddle.concat(x=[x1, x2], axis=zero)

# data=paddle.to_tensor([[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j]])

# conj_data=paddle.conj(data)

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.cos(x)
# print(out)

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.cosh(x)
# print(out)

# cpu_place = paddle.CPUPlace()

# paddle.enable_static()
# W = paddle.static.create_parameter(shape=[784, 200], dtype='float32')

# paddle.disable_static()
# x = paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# shape = paddle.to_tensor([2, 2], dtype='int32')
# out = paddle.crop(x, shape)
# offsets = paddle.to_tensor([0, 1], dtype='int32')
# out = paddle.crop(x, shape, offsets)
  
  
  
  
  

# x = paddle.to_tensor([[1.0, 1.0, 1.0],
#                       [2.0, 2.0, 2.0],
#                       [3.0, 3.0, 3.0]])
# y = paddle.to_tensor([[1.0, 1.0, 1.0],
#                       [1.0, 1.0, 1.0],
#                       [1.0, 1.0, 1.0]])

# z1 = paddle.cross(x, y)

# z2 = paddle.cross(x, y, axis=1)

# # place = paddle.CUDAPinnedPlace()

# # place = paddle.CUDAPlace(0)

# data = paddle.arange(12)
# data = paddle.reshape(data, (3, 4))

# y = paddle.cumprod(data, dim=0)

# y = paddle.cumprod(data, dim=-1)

# y = paddle.cumprod(data, dim=1, dtype='float64')

# print(y.dtype)

# data = paddle.arange(12)
# data = paddle.reshape(data, (3, 4))

# y = paddle.cumsum(data)

# y = paddle.cumsum(data, axis=0)

# y = paddle.cumsum(data, axis=-1)

# y = paddle.cumsum(data, dtype='float64')
# print(y.dtype)

# class LinearNet(nn.Layer):
#     def __init__(self):
#         super(LinearNet, self).__init__()
#         self._linear1 = nn.Linear(10, 10)
#         self._linear2 = nn.Linear(10, 1)

#     def forward(self, x):
#         return self._linear2(self._linear1(x))

# def train():
    
#     dist.init_parallel_env()

    
#     layer = LinearNet()
#     dp_layer = paddle.DataParallel(layer)

#     loss_fn = nn.MSELoss()
#     adam = opt.Adam(
#         learning_rate=0.001, parameters=dp_layer.parameters())

    
#     inputs = paddle.randn([10, 10], 'float32')
#     outputs = dp_layer(inputs)
#     labels = paddle.randn([10, 1], 'float32')
#     loss = loss_fn(outputs, labels)

#     loss.backward()

#     adam.step()
#     adam.clear_grad()

# if __name__ == '__main__':
    
#     dist.spawn(train, nprocs=2)
    
    

# class cus_tanh(PyLayer):
#     @staticmethod
#     def forward(ctx, x):
#         y = paddle.tanh(x)
#         ctx.save_for_backward(y)
#         return y

#     @staticmethod
#     def backward(ctx, dy):
#         y, = ctx.saved_tensor()
#         grad = dy * (1 - paddle.square(y))
#         return grad

# class SimpleNet(paddle.nn.Layer):
#     def __init__(self):
#         super(SimpleNet, self).__init__()
#         self.linear = paddle.nn.Linear(2, 2)

#     def forward(self, inputs):
#         inputs = cus_tanh.apply(inputs)
#         return self.linear(inputs)

# if __name__ == '__main__':
#     dist.init_parallel_env()

#     model = SimpleNet()
#     model = paddle.DataParallel(model)
#     opt = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())

#     for step in range(10):
#         x_data = numpy.random.randn(2, 2).astype(numpy.float32)
#         x = paddle.to_tensor(x_data)
#         x.stop_gradient = False

        
#         with model.no_sync():
#             y_pred = model(x)
#             loss = y_pred.mean()
#             loss.backward()

        
#         fused_allreduce_gradients(list(model.parameters()), None)

#         opt.step()
#         opt.clear_grad()

# class SimpleNet(nn.Layer):
#     def __init__(self):
#         super(SimpleNet, self).__init__()
#         self._linear = nn.Linear(10, 1)

#     def forward(self, x):
#         return self._linear(x)

# dist.init_parallel_env()
# model = SimpleNet()
# dp_model = paddle.DataParallel(model)

# inputs_1 = paddle.randn([10, 10], 'float32')
# inputs_2 = paddle.ones([10, 10], 'float32')

# with dp_model.no_sync():
    
#     dp_model(inputs_1).backward()

# dp_model(inputs_2).backward()

# dist.init_parallel_env()

# emb = fluid.dygraph.Embedding([10, 10])
# emb = fluid.dygraph.DataParallel(emb)

# state_dict = emb.state_dict()
# paddle.save(state_dict, "paddle_dy.pdparams")

# class MyLayer(paddle.nn.Layer):
#     def __init__(self):
#         super(MyLayer, self).__init__()
#         self._linear = paddle.nn.Linear(1, 1)
#         w_tmp = self.create_parameter([1,1])
#         self.add_parameter("w_tmp", w_tmp)

#     def forward(self, input):
#         return self._linear(input)

# mylayer = MyLayer()
# for name, param in mylayer.named_parameters():
#     print(name, param)      

# class MySequential(paddle.nn.Layer):
#     def __init__(self, *layers):
#         super(MySequential, self).__init__()
#         if len(layers) > 0 and isinstance(layers[0], tuple):
#             for name, layer in layers:
#                 self.add_sublayer(name, layer)
#         else:
#             for idx, layer in enumerate(layers):
#                 self.add_sublayer(str(idx), layer)

#     def forward(self, input):
#         for layer in self._sub_layers.values():
#             input = layer(input)
#         return input

# fc1 = paddle.nn.Linear(10, 3)
# fc2 = paddle.nn.Linear(3, 10, bias_attr=False)
# model = MySequential(fc1, fc2)
# for prefix, layer in model.named_sublayers():
#     print(prefix, layer)

# net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))

# def init_weights(layer):
#     if type(layer) == nn.Linear:
#         print('before init weight:', layer.weight.numpy())
#         new_weight = paddle.full(shape=layer.weight.shape, dtype=layer.weight.dtype, fill_value=0.9)
#         layer.weight.set_value(new_weight)
#         print('after init weight:', layer.weight.numpy())

# net.apply(init_weights)

# print(net.state_dict())

# linear = paddle.nn.Linear(10, 3)
# value = np.array([0]).astype("float32")
# buffer = paddle.to_tensor(value)
# linear.register_buffer("buf_name", buffer, persistable=True)

# print(linear.buffers())     

# linear1 = paddle.nn.Linear(10, 3)
# linear2 = paddle.nn.Linear(3, 10, bias_attr=False)
# model = paddle.nn.Sequential(linear1, linear2)

# layer_list = list(model.children())

# print(layer_list)   

# value = np.arange(26).reshape(2, 13).astype("float32")
# a = paddle.to_tensor(value)
# linear = paddle.nn.Linear(13, 5)
# adam = paddle.optimizer.Adam(learning_rate=0.01,
#                             parameters=linear.parameters())
# out = linear(a)
# out.backward()
# adam.step()
# linear.clear_gradients()

# class MyLayer(paddle.nn.Layer):
#     def __init__(self):
#         super(MyLayer, self).__init__()
#         self._linear = paddle.nn.Linear(1, 1)
#         w_tmp = self.create_parameter([1,1])
#         self.add_parameter("w_tmp", w_tmp)

#     def forward(self, input):
#         return self._linear(input)

# mylayer = MyLayer()
# for name, param in mylayer.named_parameters():
#     print(name, param)      

# class MyLinear(paddle.nn.Layer):
#     def __init__(self,
#                 in_features,
#                 out_features):
#         super(MyLinear, self).__init__()
#         self.linear = paddle.nn.Linear( 10, 10)

#         self.back_var = self.create_tensor(name = "linear_tmp_0", dtype=self._dtype)

#     def forward(self, input):
#         out = self.linear(input)
#         paddle.assign( out, self.back_var)

#         return out

# class MyLinear(paddle.nn.Layer):
#     def __init__(self,
#                 in_features,
#                 out_features):
#         super(MyLinear, self).__init__()
#         self.linear = paddle.nn.Linear( 10, 10)

#         self.back_var = self.create_variable(name = "linear_tmp_0", dtype=self._dtype)

#     def forward(self, input):
#         out = self.linear(input)
#         paddle.assign( out, self.back_var)

#         return out

# class MyLayer(paddle.nn.Layer):
#     def __init__(self):
#         super(MyLayer, self).__init__()
#         self._linear = paddle.nn.Linear(1, 1)
#         self._dropout = paddle.nn.Dropout(p=0.5)

#     def forward(self, input):
#         temp = self._linear(input)
#         temp = self._dropout(temp)
#         return temp

# x = paddle.randn([10, 1], 'float32')
# mylayer = MyLayer()
# mylayer.eval()  
# out = mylayer(x)
# print(out)

# class LinearNet(paddle.nn.Layer):
#     def __init__(self):
#         super(LinearNet, self).__init__(name_scope = "demo_linear_net")
#         self._linear = paddle.nn.Linear(1, 1)

#     def forward(self, x):
#         return self._linear(x)

# linear_net = LinearNet()
# print(linear_net.full_name())   

# fc1 = paddle.nn.Linear(10, 3)
# buffer1 = paddle.to_tensor(np.array([0]).astype("float32"))

# fc1.register_buffer("buf_name_1", buffer1, persistable=True)

# fc2 = paddle.nn.Linear(3, 10)
# buffer2 = paddle.to_tensor(np.array([1]).astype("float32"))

# fc2.buf_name_2 = buffer2

# model = paddle.nn.Sequential(fc1, fc2)

# for name, buffer in model.named_buffers():
#     print(name, buffer)

# linear1 = paddle.nn.Linear(10, 3)
# linear2 = paddle.nn.Linear(3, 10, bias_attr=False)
# model = paddle.nn.Sequential(linear1, linear2)
# for prefix, layer in model.named_children():
#     print(prefix, layer)
    
    

# fc1 = paddle.nn.Linear(10, 3)
# fc2 = paddle.nn.Linear(3, 10, bias_attr=False)
# model = paddle.nn.Sequential(fc1, fc2)
# for name, param in model.named_parameters():
#     print(name, param)

# fc1 = paddle.nn.Linear(10, 3)
# fc2 = paddle.nn.Linear(3, 10, bias_attr=False)
# model = paddle.nn.Sequential(fc1, fc2)
# for prefix, layer in model.named_sublayers():
#     print(prefix, layer)

# linear = paddle.nn.Linear(10, 3)
# value = np.array([0]).astype("float32")
# buffer = paddle.to_tensor(value)
# linear.register_buffer("buf_name", buffer, persistable=True)

# print(linear.buf_name)

# def forward_post_hook(layer, input, output):
    

    
#     return output * 2

# linear = paddle.nn.Linear(13, 5)

# forward_post_hook_handle = linear.register_forward_post_hook(forward_post_hook)

# value1 = np.arange(26).reshape(2, 13).astype("float32")
# in1 = paddle.to_tensor(value1)

# out0 = linear(in1)

# forward_post_hook_handle.remove()

# out1 = linear(in1)

# assert (out0.numpy() == (out1.numpy()) * 2).any()

# def forward_pre_hook(layer, input):
    

    
#     input_return = (input[0] * 2)
#     return input_return

# linear = paddle.nn.Linear(13, 5)

# forward_pre_hook_handle = linear.register_forward_pre_hook(forward_pre_hook)

# value0 = np.arange(26).reshape(2, 13).astype("float32")
# in0 = paddle.to_tensor(value0)
# out0 = linear(in0)

# forward_pre_hook_handle.remove()

# value1 = value0 * 2
# in1 = paddle.to_tensor(value1)
# out1 = linear(in1)

# assert (out0.numpy() == out1.numpy()).any()

# dist.init_parallel_env()

# emb = paddle.nn.Embedding(10, 10)
# emb = fluid.dygraph.DataParallel(emb)

# state_dict = emb.state_dict()
# paddle.save(state_dict, "paddle_dy.pdparams")

# para_state_dict = paddle.load("paddle_dy.pdparams")
# emb.set_state_dict(para_state_dict)

# class MyLayer(paddle.nn.Layer):
#     def __init__(self):
#         super(MyLayer, self).__init__()
#         self._linear = paddle.nn.Linear(1, 1)
#         self._dropout = paddle.nn.Dropout(p=0.5)

#     def forward(self, input):
#         temp = self._linear(input)
#         temp = self._dropout(temp)
#         return temp

# mylayer = MyLayer()
# print(mylayer.sublayers())  

# linear=paddle.nn.Linear(2, 2)
# linear.weight

# linear.to(dtype='float64')
# linear.weight

# linear.to(device='cpu')
# linear.weight

# linear.to(device=paddle.CUDAPinnedPlace(), blocking=False)
# linear.weight

# emb = paddle.nn.Embedding(10, 10)

# state_dict = emb.to_static_state_dict()
# paddle.save( state_dict, "paddle_dy.pdparams")

# class MyLayer(paddle.nn.Layer):
#     def __init__(self):
#         super(MyLayer, self).__init__()
#         self._linear = paddle.nn.Linear(1, 1)
#         self._dropout = paddle.nn.Dropout(p=0.5)

#     def forward(self, input):
#         temp = self._linear(input)
#         temp = self._dropout(temp)
#         return temp

# x = paddle.randn([10, 1], 'float32')
# mylayer = MyLayer()
# mylayer.eval()  
# out = mylayer(x)
# mylayer.train()  
# out = mylayer(x)

# dist.init_parallel_env()

# emb = paddle.nn.Embedding(10, 10)
# emb = fluid.dygraph.DataParallel(emb)

# state_dict = emb.state_dict()
# paddle.save(state_dict, "paddle_dy.pdparams")

# para_state_dict = paddle.load("paddle_dy.pdparams")
# emb.set_state_dict(para_state_dict)

# dist.init_parallel_env()

# emb = paddle.nn.Embedding(10, 10)
# emb = fluid.dygraph.DataParallel(emb)

# state_dict = emb.state_dict()
# paddle.save(state_dict, "paddle_dy.pdparams")

# para_state_dict = paddle.load("paddle_dy.pdparams")
# emb.set_state_dict(para_state_dict)

# paddle.disable_static()
# x = paddle.to_tensor([1, 2, 3])
# y = paddle.diag(x)
# print(y.numpy())

# y = paddle.diag(x, offset=1)
# print(y.numpy())

# y = paddle.diag(x, padding_value=6)
# print(y.numpy())

# paddle.disable_static()
# x = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])
# y = paddle.diag(x)
# print(y.numpy())

# y = paddle.diag(x, offset=1)
# print(y.numpy())

# y = paddle.diag(x, offset=-1)
# print(y.numpy())

# x = paddle.to_tensor([1, 2, 3])
# y = paddle.diagflat(x)
# print(y.numpy())

# y = paddle.diagflat(x, offset=1)
# print(y.numpy())

# y = paddle.diagflat(x, offset=-1)
# print(y.numpy())

# x = paddle.to_tensor([[1, 2], [3, 4]])
# y = paddle.diagflat(x)
# print(y.numpy())

# y = paddle.diagflat(x, offset=1)
# print(y.numpy())

# y = paddle.diagflat(x, offset=-1)
# print(y.numpy())

# x = paddle.rand([2,2,3],'float32')
# print(x)

# out1 = paddle.diagonal(x)
# print(out1)

# out2 = paddle.diagonal(x, offset=0, axis1=2, axis2=1)
# print(out2)

# out3 = paddle.diagonal(x, offset=1, axis1=0, axis2=1)
# print(out3)

# out4 = paddle.diagonal(x, offset=0, axis1=1, axis2=2)
# print(out4)

# data = paddle.to_tensor([[1, 1.5], [0, -2.2]], dtype='float32')
# res = paddle.digamma(data)
# print(res)

# paddle.disable_signal_handler()

# print(paddle.in_dynamic_mode())  

# paddle.enable_static()
# print(paddle.in_dynamic_mode())  

# paddle.disable_static()
# print(paddle.in_dynamic_mode())  

# x = paddle.to_tensor(np.array([[3, 3],[3, 3]]), "float32")
# y = paddle.to_tensor(np.array([[3, 3],[3, 1]]), "float32")
# out = paddle.dist(x, y, 0)
# print(out) 

# out = paddle.dist(x, y, 2)
# print(out) 

# out = paddle.dist(x, y, float("inf"))
# print(out) 

# out = paddle.dist(x, y, float("-inf"))
# print(out) 

# x = paddle.to_tensor([2, 3, 4], dtype='float64')
# y = paddle.to_tensor([1, 5, 2], dtype='float64')
# z = paddle.divide(x, y)
# print(z)  

# x_data = np.random.uniform(0.1, 1, [10]).astype(np.float32)
# y_data = np.random.uniform(1, 3, [10]).astype(np.float32)
# x = paddle.to_tensor(x_data)
# y = paddle.to_tensor(y_data)
# z = paddle.dot(x, y)
# print(z)

# paddle.set_device("cpu")  

# data1 = paddle.empty(shape=[2,3], dtype='float32')

# shape_data = np.array([2, 3]).astype('int32')
# shape = paddle.to_tensor(shape_data)
# data2 = paddle.empty(shape=shape, dtype='float32')

# dim2_data = np.array([3]).astype('int32')
# dim2 = paddle.to_tensor(dim2_data)
# data3 = paddle.empty(shape=[2, dim2], dtype='float32')

# paddle.set_device("cpu")  

# x = paddle.randn([2, 3], 'float32')
# output = paddle.empty_like(x)

# print(paddle.in_dynamic_mode())  

# paddle.enable_static()
# print(paddle.in_dynamic_mode())  

# paddle.disable_static()
# print(paddle.in_dynamic_mode())  

# x = paddle.to_tensor([1, 2, 3])
# y = paddle.to_tensor([1, 3, 2])
# result1 = paddle.equal(x, y)
# print(result1)  

# x = paddle.to_tensor([1, 2, 3])
# y = paddle.to_tensor([1, 2, 3])
# z = paddle.to_tensor([1, 4, 3])
# result1 = paddle.equal_all(x, y)
# print(result1) 
# result2 = paddle.equal_all(x, z)
# print(result2) 

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.erf(x)
# print(out)

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.exp(x)
# print(out)

# data = paddle.to_tensor([1, 2, 3], dtype='int32')
# out = paddle.expand(data, shape=[2, 3])
# print(out)

# data_x = paddle.to_tensor([1, 2, 3], 'int32')
# data_y = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], 'int32')
# out = paddle.expand_as(data_x, data_y)
# np_out = out.numpy()

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.expm1(x)
# print(out)

# data = paddle.eye(3, dtype='int32')

# data = paddle.eye(2, 3, dtype='int32')

# image_shape=(2, 3, 4, 4)

# x = paddle.arange(end=image_shape[0] * image_shape[1] * image_shape[2] * image_shape[3])
# img = paddle.reshape(x, image_shape)

# out = paddle.flatten(img, start_axis=1, stop_axis=2)

# img[0, 0, 0, 0] = -1
# print(out[0, 0, 0]) 

# image_shape=(3, 2, 2)
# x = np.arange(image_shape[0] * image_shape[1] * image_shape[2]).reshape(image_shape)
# x = x.astype('float32')
# img = paddle.to_tensor(x)
# tmp = paddle.flip(img, [0,1])
# print(tmp) 

# out = paddle.flip(tmp,-1)
# print(out) 

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.floor(x)
# print(out)

# x = paddle.to_tensor([2, 3, 8, 7])
# y = paddle.to_tensor([1, 5, 3, 3])
# z = paddle.floor_divide(x, y)
# print(z)  

# x = paddle.to_tensor([2, 3, 8, 7])
# y = paddle.to_tensor([1, 5, 3, 3])
# z = paddle.remainder(x, y)
# print(z)  

# class LeNet(nn.Layer):
#     def __init__(self, num_classes=10):
#         super(LeNet, self).__init__()
#         self.num_classes = num_classes
#         self.features = nn.Sequential(
#             nn.Conv2D(
#                 1, 6, 3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2D(2, 2),
#             nn.Conv2D(
#                 6, 16, 5, stride=1, padding=0),
#             nn.ReLU(),
#             nn.MaxPool2D(2, 2))

#         if num_classes > 0:
#             self.fc = nn.Sequential(
#                 nn.Linear(400, 120),
#                 nn.Linear(120, 84),
#                 nn.Linear(
#                     84, 10))

#     def forward(self, inputs):
#         x = self.features(inputs)

#         if self.num_classes > 0:
#             x = paddle.flatten(x, 1)
#             x = self.fc(x)
#         return x

# lenet = LeNet()

# def count_leaky_relu(m, x, y):
#     x = x[0]
#     nelements = x.numel()
#     m.total_ops += int(nelements)

# FLOPs = paddle.flops(lenet, [1, 1, 28, 28], custom_ops= {nn.LeakyReLU: count_leaky_relu},
#                     print_detail=True)
# print(FLOPs)

# data1 = paddle.full(shape=[2,1], fill_value=0, dtype='int64')

# positive_2 = paddle.full([1], 2, "int32")
# data3 = paddle.full(shape=[1, positive_2], dtype='float32', fill_value=1.5)

# shape = paddle.full([2], 2, "int32")
# data4 = paddle.full(shape=shape, dtype='bool', fill_value=True)

# val = paddle.full([1], 2.0, "float32")
# data5 = paddle.full(shape=[2,1], fill_value=val, dtype='float32')

# input = paddle.full(shape=[2, 3], fill_value=0.0, dtype='float32', name='input')
# output = paddle.full_like(input, 2.0)

# input = paddle.to_tensor([[1,2],[3,4],[5,6]])
# index = paddle.to_tensor([0,1])
# output = paddle.gather(input, index, axis=0)

# x = paddle.to_tensor([[[1, 2], [3, 4], [5, 6]],
#                       [[7, 8], [9, 10], [11, 12]]])
# index = paddle.to_tensor([[0, 1]])

# output = paddle.gather_nd(x, index) 

# sts = paddle.get_cuda_rng_state()

# paddle.get_default_dtype()

# flags = ['FLAGS_eager_delete_tensor_gb', 'FLAGS_check_nan_inf']
# res = paddle.get_flags(flags)
# print(res)

# def test_dygraph_grad(create_graph):
#     x = paddle.ones(shape=[1], dtype='float32')
#     x.stop_gradient = False
#     y = x * x

    
#     dx = paddle.grad(
#             outputs=[y],
#             inputs=[x],
#             create_graph=create_graph,
#             retain_graph=True)[0]

#     z = y + dx

    
    
    

    
    
    
    

#     z.backward()
#     return x.gradient()

# print(test_dygraph_grad(create_graph=False)) 
# print(test_dygraph_grad(create_graph=True)) 

# def test_dygraph_grad(grad_outputs=None):
#     x = paddle.to_tensor(2.0)
#     x.stop_gradient = False

#     y1 = x * x
#     y2 = x * 3

    
    
    
    

    
    
    
    

#     dx = paddle.grad(
#         outputs=[y1, y2],
#         inputs=[x],
#         grad_outputs=grad_outputs)[0]

#     return dx.numpy()

# grad_value = paddle.to_tensor(4.0)

# print(test_dygraph_grad(None)) 

# print(test_dygraph_grad([None, grad_value])) 

# print(test_dygraph_grad([grad_value, None])) 

# grad_y1 = paddle.to_tensor(3.0)
# print(test_dygraph_grad([grad_y1, grad_value])) 

# x = paddle.to_tensor([1, 2, 3])
# y = paddle.to_tensor([1, 3, 2])
# result1 = paddle.greater_equal(x, y)
# print(result1)  

# x = paddle.to_tensor([1, 2, 3])
# y = paddle.to_tensor([1, 3, 2])
# result1 = paddle.greater_than(x, y)
# print(result1)  

# inputs = paddle.to_tensor([1, 2, 1])
# result = paddle.histogram(inputs, bins=4, min=0, max=3)
# print(result) 

# x = paddle.to_tensor(
#     [[1 + 6j, 2 + 5j, 3 + 4j], [4 + 3j, 5 + 2j, 6 + 1j]])

# imag_res = paddle.imag(x)

# imag_t = x.imag()

# print(paddle.in_dynamic_mode())  

# paddle.enable_static()
# print(paddle.in_dynamic_mode())  

# paddle.disable_static()
# print(paddle.in_dynamic_mode())  

# data = paddle.zeros(shape=[1], dtype='float32')
# counter = paddle.increment(data)

# x = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0],
#                       [5.0, 6.0, 7.0, 8.0],
#                       [9.0, 10.0, 11.0, 12.0]], dtype='float32')
# index = paddle.to_tensor([[0, 1, 2],
#                           [1, 2, 3],
#                           [0, 0, 0]], dtype='int32')
# target = paddle.to_tensor([[100, 200, 300, 400],
#                            [500, 600, 700, 800],
#                            [900, 1000, 1100, 1200]], dtype='int32')
# out_z1 = paddle.index_sample(x, index)
# print(out_z1)

# top_value, top_index = paddle.topk(x, k=2)
# out_z2 = paddle.index_sample(target, top_index)
# print(top_value)

# print(top_index)

# print(out_z2)

# x = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0],
#                       [5.0, 6.0, 7.0, 8.0],
#                       [9.0, 10.0, 11.0, 12.0]])
# index = paddle.to_tensor([0, 1, 1], dtype='int32')
# out_z1 = paddle.index_select(x=x, index=index)

# out_z2 = paddle.index_select(x=x, index=index, axis=1)

# mat = paddle.to_tensor([[2, 0], [0, 2]], dtype='float32')
# inv = paddle.inverse(mat)
# print(inv) 

# input = paddle.rand(shape=[4, 32, 32], dtype='float32')
# res = paddle.is_empty(x=input)
# print("res:", res)

# input1 = paddle.rand(shape=[2, 3, 5], dtype='float32')
# check = paddle.is_tensor(input1)
# print(check)  

# input3 = [1, 4]
# check = paddle.is_tensor(input3)
# print(check)  

# x = paddle.to_tensor([float('-inf'), -2, 3.6, float('inf'), 0, float('-nan'), float('nan')])
# out = paddle.tensor.isfinite(x)
# print(out)  

# x = paddle.to_tensor([float('-inf'), -2, 3.6, float('inf'), 0, float('-nan'), float('nan')])
# out = paddle.tensor.isinf(x)
# print(out)  

# x = paddle.to_tensor([float('-inf'), -2, 3.6, float('inf'), 0, float('-nan'), float('nan')])
# out = paddle.tensor.isnan(x)
# print(out)  

# x = paddle.to_tensor([[1, 2], [3, 4]], dtype='int64')
# y = paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='int64')
# out = paddle.kron(x, y)
# print(out)

# x = paddle.to_tensor([1, 2, 3])
# y = paddle.to_tensor([1, 3, 2])
# result1 = paddle.less_equal(x, y)
# print(result1)  

# x = paddle.to_tensor([1, 2, 3])
# y = paddle.to_tensor([1, 3, 2])
# result1 = paddle.less_than(x, y)
# print(result1)  

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.lgamma(x)
# print(out)

# data = paddle.linspace(0, 10, 5, 'float32') 
# data = paddle.linspace(0, 10, 1, 'float32') 

# emb = paddle.nn.Embedding(10, 10)
# layer_state_dict = emb.state_dict()

# paddle.save(layer_state_dict, "emb.pdparams")

# scheduler = paddle.optimizer.lr.NoamDecay(
#     d_model=0.01, warmup_steps=100, verbose=True)
# adam = paddle.optimizer.Adam(
#     learning_rate=scheduler,
#     parameters=emb.parameters())
# opt_state_dict = adam.state_dict()

# paddle.save(opt_state_dict, "adam.pdopt")

# paddle.save(emb.weight, "emb.weight.pdtensor")

# load_layer_state_dict = paddle.load("emb.pdparams")

# load_opt_state_dict = paddle.load("adam.pdopt")

# load_weight = paddle.load("emb.weight.pdtensor")

# layer = paddle.nn.Linear(3, 4)
# adam = Adam(learning_rate=0.001, parameters=layer.parameters())
# obj = {'model': layer.state_dict(), 'opt': adam.state_dict(), 'epoch': 100}
# path = 'example/model.pdparams'
# paddle.save(obj, path)
# obj_load = paddle.load(path)

# paddle.enable_static()

# x = paddle.static.data(name="x", shape=[None, 224], dtype='float32')
# z = paddle.static.nn.fc(x, 10)

# place = paddle.CPUPlace()
# exe = paddle.static.Executor(place)
# exe.run(paddle.static.default_startup_program())
# prog = paddle.static.default_main_program()
# for var in prog.list_vars():
#     if list(var.shape) == [224, 10]:
#         tensor = var.get_value()
#         break

# path_tensor = 'temp/tensor.pdtensor'
# paddle.save(tensor, path_tensor)
# load_tensor = paddle.load(path_tensor)

# path_state_dict = 'temp/model.pdparams'
# paddle.save(prog.state_dict("param"), path_tensor)
# load_state_dict = paddle.load(path_tensor)

# paddle.enable_static()

# data = paddle.static.data(
#     name='x_static_save', shape=(None, 224), dtype='float32')
# y_static = z = paddle.static.nn.fc(data, 10)
# main_program = paddle.static.default_main_program()
# path = "example/main_program.pdmodel"
# paddle.save(main_program, path)
# load_main = paddle.load(path)
# print(load_main)

# paddle.disable_static()

# linear = Linear(5, 10)
# state_dict = linear.state_dict()
# byio = BytesIO()
# paddle.save(state_dict, byio)
# tensor = paddle.randn([2, 3], dtype='float32')
# paddle.save(tensor, byio)
# byio.seek(0)

# dict_load = paddle.load(byio)

# x = [[2,3,4], [7,8,9]]
# x = paddle.to_tensor(x, dtype='float32')
# res = paddle.log(x)

# x_i = paddle.to_tensor([[1.0], [10.0]])
# res = paddle.log10(x_i) 

# x_i = paddle.full(shape=[1], fill_value=10, dtype='float32')
# paddle.to_tensor(x_i)
# res = paddle.log10(x_i)
# print(res) 

# x_i = paddle.full(shape=[1], fill_value=10, dtype='float64')
# paddle.to_tensor(x_i)
# res = paddle.log10(x_i)
# print(res) 

# data = paddle.to_tensor([[0], [1]], dtype='float32')
# res = paddle.log1p(data)

# x_i = paddle.to_tensor([[1.0], [2.0]])
# res = paddle.log2(x_i) 

# x_i = paddle.full(shape=[1], fill_value=2, dtype='float32')
# paddle.to_tensor(x_i)
# res = paddle.log2(x_i)
# print(res) 

# x_i = paddle.full(shape=[1], fill_value=2, dtype='float64')
# paddle.to_tensor(x_i)
# res = paddle.log2(x_i)
# print(res) 

# x = paddle.to_tensor([True])
# y = paddle.to_tensor([True, False, True, False])
# res = paddle.logical_and(x, y)
# print(res) 

# x = paddle.to_tensor([True, False, True, False])
# res = paddle.logical_not(x)
# print(res) 

# x_data = np.array([True, False], dtype=np.bool).reshape(2, 1)
# y_data = np.array([True, False, True, False], dtype=np.bool).reshape(2, 2)
# x = paddle.to_tensor(x_data)
# y = paddle.to_tensor(y_data)
# res = paddle.logical_or(x, y)
# print(res) 

# x_data = np.array([True, False], dtype=np.bool).reshape([2, 1])
# y_data = np.array([True, False, True, False], dtype=np.bool).reshape([2, 2])
# x = paddle.to_tensor(x_data)
# y = paddle.to_tensor(y_data)
# res = paddle.logical_xor(x, y)
# print(res) 

# x = paddle.to_tensor([[-1.5, 0., 2.], [3., 1.2, -2.4]])
# out1 = paddle.logsumexp(x) 
# out2 = paddle.logsumexp(x, 1) 

# x = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0],
#                       [5.0, 6.0, 7.0, 8.0],
#                       [9.0, 10.0, 11.0, 12.0]])
# mask = paddle.to_tensor([[True, False, False, False],
#                          [True, True, False, False],
#                          [True, False, False, False]])
# out = paddle.masked_select(x, mask)

# x_data = np.random.random([10]).astype(np.float32)
# y_data = np.random.random([10]).astype(np.float32)
# x = paddle.to_tensor(x_data)
# y = paddle.to_tensor(y_data)
# z = paddle.matmul(x, y)
# print(z.numpy().shape)

# x_data = np.random.random([10, 5]).astype(np.float32)
# y_data = np.random.random([5]).astype(np.float32)
# x = paddle.to_tensor(x_data)
# y = paddle.to_tensor(y_data)
# z = paddle.matmul(x, y)
# print(z.numpy().shape)

# x_data = np.random.random([10, 5, 2]).astype(np.float32)
# y_data = np.random.random([2]).astype(np.float32)
# x = paddle.to_tensor(x_data)
# y = paddle.to_tensor(y_data)
# z = paddle.matmul(x, y)
# print(z.numpy().shape)

# x_data = np.random.random([10, 5, 2]).astype(np.float32)
# y_data = np.random.random([10, 2, 5]).astype(np.float32)
# x = paddle.to_tensor(x_data)
# y = paddle.to_tensor(y_data)
# z = paddle.matmul(x, y)
# print(z.numpy().shape)

# x_data = np.random.random([10, 1, 5, 2]).astype(np.float32)
# y_data = np.random.random([1, 3, 2, 5]).astype(np.float32)
# x = paddle.to_tensor(x_data)
# y = paddle.to_tensor(y_data)
# z = paddle.matmul(x, y)
# print(z.numpy().shape)

# x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
#                       [0.1, 0.2, 0.6, 0.7]])
# result1 = paddle.max(x)
# print(result1)

# result2 = paddle.max(x, axis=0)
# print(result2)

# result3 = paddle.max(x, axis=-1)
# print(result3)

# result4 = paddle.max(x, axis=1, keepdim=True)
# print(result4)

# y = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]],
#                       [[5.0, 6.0], [7.0, 8.0]]])
# result5 = paddle.max(y, axis=[1, 2])
# print(result5)

# result6 = paddle.max(y, axis=[0, 1])
# print(result6)

# x = paddle.to_tensor([[1, 2], [7, 8]])
# y = paddle.to_tensor([[3, 4], [5, 6]])
# res = paddle.maximum(x, y)
# print(res)

# x = paddle.to_tensor([[1, 2, 3], [1, 2, 3]])
# y = paddle.to_tensor([3, 0, 4])
# res = paddle.maximum(x, y)
# print(res)

# x = paddle.to_tensor([2, 3, 5], dtype='float32')
# y = paddle.to_tensor([1, np.nan, np.nan], dtype='float32')
# res = paddle.maximum(x, y)
# print(res)

# x = paddle.to_tensor([5, 3, np.inf], dtype='float32')
# y = paddle.to_tensor([1, -np.inf, 5], dtype='float32')
# res = paddle.maximum(x, y)
# print(res)

# x = paddle.to_tensor([[[1., 2., 3., 4.],
#                        [5., 6., 7., 8.],
#                        [9., 10., 11., 12.]],
#                       [[13., 14., 15., 16.],
#                        [17., 18., 19., 20.],
#                        [21., 22., 23., 24.]]])
# out1 = paddle.mean(x)

# out2 = paddle.mean(x, axis=-1)

# out3 = paddle.mean(x, axis=-1, keepdim=True)

# out4 = paddle.mean(x, axis=[0, 2])

# x = paddle.arange(12).reshape([3, 4])

# y1 = paddle.median(x)

# y2 = paddle.median(x, axis=0)

# y3 = paddle.median(x, axis=1)

# y4 = paddle.median(x, axis=0, keepdim=True)

# x = paddle.randint(low=0, high=100, shape=[100])
# y = paddle.randint(low=0, high=100, shape=[200])

# grid_x, grid_y = paddle.meshgrid(x, y)

# print(grid_x.shape)
# print(grid_y.shape)

# x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
#                       [0.1, 0.2, 0.6, 0.7]])
# result1 = paddle.min(x)
# print(result1)

# result2 = paddle.min(x, axis=0)
# print(result2)

# result3 = paddle.min(x, axis=-1)
# print(result3)

# result4 = paddle.min(x, axis=1, keepdim=True)
# print(result4)

# y = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]],
#                       [[5.0, 6.0], [7.0, 8.0]]])
# result5 = paddle.min(y, axis=[1, 2])
# print(result5)

# result6 = paddle.min(y, axis=[0, 1])
# print(result6)

# x = paddle.to_tensor([[1, 2], [7, 8]])
# y = paddle.to_tensor([[3, 4], [5, 6]])
# res = paddle.minimum(x, y)
# print(res)

# x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
# y = paddle.to_tensor([3, 0, 4])
# res = paddle.minimum(x, y)
# print(res)

# x = paddle.to_tensor([2, 3, 5], dtype='float32')
# y = paddle.to_tensor([1, np.nan, np.nan], dtype='float32')
# res = paddle.minimum(x, y)
# print(res)

# x = paddle.to_tensor([5, 3, np.inf], dtype='float64')
# y = paddle.to_tensor([1, -np.inf, 5], dtype='float64')
# res = paddle.minimum(x, y)
# print(res)

# input = paddle.arange(1, 7).reshape((3, 2)).astype('float32')
# mat2 = paddle.arange(1, 9).reshape((2, 4)).astype('float32')
# out = paddle.mm(input, mat2)
# print(out)

# device = paddle.set_device('cpu') 

# net = nn.Sequential(
#     nn.Flatten(1),
#     nn.Linear(784, 200),
#     nn.Tanh(),
#     nn.Linear(200, 10))

# input = InputSpec([None, 784], 'float32', 'x')
# label = InputSpec([None, 1], 'int64', 'label')

# model = paddle.Model(net, input, label)
# optim = paddle.optimizer.SGD(learning_rate=1e-3,
#     parameters=model.parameters())

# model.prepare(optim,
#               paddle.nn.CrossEntropyLoss(),
#               paddle.metric.Accuracy())

# transform = T.Compose([
#     T.Transpose(),
#     T.Normalize([127.5], [127.5])
# ])
# data = paddle.vision.datasets.MNIST(mode='train', transform=transform)
# model.fit(data, epochs=2, batch_size=32, verbose=1)

# def run_example_code():
#   device = paddle.set_device('gpu')

#   net = nn.Sequential(nn.Flatten(1), nn.Linear(784, 200), nn.Tanh(),
#                       nn.Linear(200, 10))

#   model = paddle.Model(net)
#   optim = paddle.optimizer.SGD(learning_rate=1e-3, parameters=model.parameters())

#   amp_configs = {
#       "level": "O1",
#       "custom_white_list": {'conv2d'},
#       "use_dynamic_loss_scaling": True
#   }
#   model.prepare(optim,
#       paddle.nn.CrossEntropyLoss(),
#       paddle.metric.Accuracy(),
#       amp_configs=amp_configs)

#   transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
#   data = paddle.vision.datasets.MNIST(mode='train', transform=transform)
#   model.fit(data, epochs=2, batch_size=32, verbose=1)

# if paddle.is_compiled_with_cuda():
#   run_example_code()

# device = paddle.set_device('cpu') 

# net = nn.Sequential(
#     nn.Linear(784, 200),
#     nn.Tanh(),
#     nn.Linear(200, 10))

# input = InputSpec([None, 784], 'float32', 'x')
# label = InputSpec([None, 1], 'int64', 'label')
# model = paddle.Model(net, input, label)
# optim = paddle.optimizer.SGD(learning_rate=1e-3,
#     parameters=model.parameters())
# model.prepare(optim, paddle.nn.CrossEntropyLoss())
# data = np.random.random(size=(4,784)).astype(np.float32)
# label = np.random.randint(0, 10, size=(4, 1)).astype(np.int64)
# loss = model.train_batch([data], [label])
# print(loss)

# device = paddle.set_device('cpu') 

# net = nn.Sequential(
#     nn.Linear(784, 200),
#     nn.Tanh(),
#     nn.Linear(200, 10))

# input = InputSpec([None, 784], 'float32', 'x')
# label = InputSpec([None, 1], 'int64', 'label')
# model = paddle.Model(net, input, label)
# optim = paddle.optimizer.SGD(learning_rate=1e-3,
#     parameters=model.parameters())
# model.prepare(optim,
#               paddle.nn.CrossEntropyLoss())
# data = np.random.random(size=(4,784)).astype(np.float32)
# label = np.random.randint(0, 10, size=(4, 1)).astype(np.int64)
# loss = model.eval_batch([data], [label])
# print(loss)

# device = paddle.set_device('cpu') 

# input = InputSpec([None, 784], 'float32', 'x')
# label = InputSpec([None, 1], 'int64', 'label')

# net = nn.Sequential(
#     nn.Linear(784, 200),
#     nn.Tanh(),
#     nn.Linear(200, 10),
#     nn.Softmax())

# model = paddle.Model(net, input, label)
# model.prepare()
# data = np.random.random(size=(4,784)).astype(np.float32)
# out = model.predict_batch([data])
# print(out)

# class Mnist(nn.Layer):
#     def __init__(self):
#         super(Mnist, self).__init__()
#         self.net = nn.Sequential(
#             nn.Flatten(1),
#             nn.Linear(784, 200),
#             nn.Tanh(),
#             nn.Linear(200, 10),
#             nn.Softmax())

#     def forward(self, x):
#         return self.net(x)

# dynamic = True  

# if not dynamic:
#     paddle.enable_static()

# input = InputSpec([None, 784], 'float32', 'x')
# label = InputSpec([None, 1], 'int64', 'label')
# model = paddle.Model(Mnist(), input, label)
# optim = paddle.optimizer.SGD(learning_rate=1e-3,
#     parameters=model.parameters())
# model.prepare(optim, paddle.nn.CrossEntropyLoss())

# transform = T.Compose([
#     T.Transpose(),
#     T.Normalize([127.5], [127.5])
# ])
# data = paddle.vision.datasets.MNIST(mode='train', transform=transform)

# model.fit(data, epochs=1, batch_size=32, verbose=0)
# model.save('checkpoint/test')  
# model.save('inference_model', False)  

# device = paddle.set_device('cpu')

# input = InputSpec([None, 784], 'float32', 'x')

# model = paddle.Model(nn.Sequential(
#     nn.Linear(784, 200),
#     nn.Tanh(),
#     nn.Linear(200, 10),
#     nn.Softmax()), input)

# model.save('checkpoint/test')
# model.load('checkpoint/test')

# input = InputSpec([None, 784], 'float32', 'x')

# model = paddle.Model(nn.Sequential(
#     nn.Linear(784, 200),
#     nn.Tanh(),
#     nn.Linear(200, 10)), input)

# params = model.parameters()

# dynamic = True
# if not dynamic:
#     paddle.enable_static()

# transform = T.Compose([
#     T.Transpose(),
#     T.Normalize([127.5], [127.5])
# ])
# train_dataset = MNIST(mode='train', transform=transform)
# val_dataset = MNIST(mode='test', transform=transform)

# input = InputSpec([None, 1, 28, 28], 'float32', 'image')
# label = InputSpec([None, 1], 'int64', 'label')

# model = paddle.Model(
#     paddle.vision.models.LeNet(),
#     input, label)
# optim = paddle.optimizer.Adam(
#     learning_rate=0.001, parameters=model.parameters())
# model.prepare(
#     optim,
#     paddle.nn.CrossEntropyLoss(),
#     paddle.metric.Accuracy(topk=(1, 2)))
# model.fit(train_dataset,
#           val_dataset,
#           epochs=2,
#           batch_size=64,
#           save_dir='mnist_checkpoint')

# dynamic = True
# if not dynamic:
#     paddle.enable_static()

# transform = T.Compose([
#       T.Transpose(),
#       T.Normalize([127.5], [127.5])
#   ])
# train_dataset = MNIST(mode='train', transform=transform)
# train_loader = paddle.io.DataLoader(train_dataset,
#     batch_size=64)
# val_dataset = MNIST(mode='test', transform=transform)
# val_loader = paddle.io.DataLoader(val_dataset,
#     batch_size=64)

# input = InputSpec([None, 1, 28, 28], 'float32', 'image')
# label = InputSpec([None, 1], 'int64', 'label')

# model = paddle.Model(
#     paddle.vision.models.LeNet(), input, label)
# optim = paddle.optimizer.Adam(
#     learning_rate=0.001, parameters=model.parameters())
# model.prepare(
#     optim,
#     paddle.nn.CrossEntropyLoss(),
#     paddle.metric.Accuracy(topk=(1, 2)))
# model.fit(train_loader,
#           val_loader,
#           epochs=2,
#           save_dir='mnist_checkpoint')

# transform = T.Compose([
#         T.Transpose(),
#         T.Normalize([127.5], [127.5])
#     ])
# val_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)

# input = InputSpec([-1, 1, 28, 28], 'float32', 'image')
# label = InputSpec([None, 1], 'int64', 'label')
# model = paddle.Model(paddle.vision.models.LeNet(), input, label)
# model.prepare(metrics=paddle.metric.Accuracy())
# result = model.evaluate(val_dataset, batch_size=64)
# print(result)

# class MnistDataset(paddle.vision.datasets.MNIST):
#     def __init__(self, mode, return_label=True):
#         super(MnistDataset, self).__init__(mode=mode)
#         self.return_label = return_label

#     def __getitem__(self, idx):
#         img = np.reshape(self.images[idx], [1, 28, 28])
#         if self.return_label:
#             return img, np.array(self.labels[idx]).astype('int64')
#         return img,

#     def __len__(self):
#         return len(self.images)

# test_dataset = MnistDataset(mode='test', return_label=False)

# input = InputSpec([-1, 1, 28, 28], 'float32', 'image')
# model = paddle.Model(paddle.vision.models.LeNet(), input)
# model.prepare()
# result = model.predict(test_dataset, batch_size=64)
# print(len(result[0]), result[0][0].shape)

# device = paddle.set_device('cpu')
# paddle.enable_static()
# input = InputSpec([-1, 1, 28, 28], 'float32', 'image')
# model = paddle.Model(paddle.vision.models.LeNet(), input)
# model.prepare()

# result = model.predict(test_dataset, batch_size=64)
# print(len(result[0]), result[0][0].shape)

# input = InputSpec([None, 1, 28, 28], 'float32', 'image')
# label = InputSpec([None, 1], 'int64', 'label')

# model = paddle.Model(paddle.vision.models.LeNet(),
#     input, label)
# optim = paddle.optimizer.Adam(
#     learning_rate=0.001, parameters=model.parameters())
# model.prepare(
#     optim,
#     paddle.nn.CrossEntropyLoss())

# params_info = model.summary()
# print(params_info)

# paddle.seed(100) 
# x = paddle.rand([2,4])
# print(x)

# paddle.seed(200) 
# out1 = paddle.multinomial(x, num_samples=5, replacement=True)
# print(out1)

# paddle.seed(300) 
# out3 = paddle.multinomial(x, num_samples=3)
# print(out3)

# img1 = np.array([[1, 2], [3, 4]]).astype(np.float32)
# img2 = np.array([[5, 6], [7, 8]]).astype(np.float32)
# inputs = [paddle.to_tensor(img1), paddle.to_tensor(img2)]
# index = paddle.to_tensor(np.array([[1], [0]]).astype(np.int32))
# res = paddle.multiplex(inputs, index)
# print(res) 

# x = paddle.to_tensor([[1, 2], [3, 4]])
# y = paddle.to_tensor([[5, 6], [7, 8]])
# res = paddle.multiply(x, y)
# print(res) 

# x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
# y = paddle.to_tensor([2])
# res = paddle.multiply(x, y)
# print(res) 

# x_data = np.array([[2, 1, 3], [3, 0, 1]]).astype("float64")
# x = paddle.to_tensor(x_data)
# vec_data = np.array([3, 5, 1])
# vec = paddle.to_tensor(vec_data).astype("float64")
# out = paddle.mv(x, vec)

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.neg(x)
# print(out)

# x1 = paddle.to_tensor([[1.0, 0.0, 0.0],
#                        [0.0, 2.0, 0.0],
#                        [0.0, 0.0, 3.0]])
# x2 = paddle.to_tensor([0.0, 1.0, 0.0, 3.0])
# out_z1 = paddle.nonzero(x1)
# print(out_z1)

# out_z1_tuple = paddle.nonzero(x1, as_tuple=True)
# for out in out_z1_tuple:
#     print(out)

# out_z2 = paddle.nonzero(x2)
# print(out_z2)

# out_z2_tuple = paddle.nonzero(x2, as_tuple=True)
# for out in out_z2_tuple:
#     print(out)

# shape=[2, 3, 4]
# np_input = np.arange(24).astype('float32') - 12
# np_input = np_input.reshape(shape)
# x = paddle.to_tensor(np_input)

# out_fro = paddle.norm(x, p='fro', axis=[0,1])

# out_pnorm = paddle.norm(x, p=2, axis=-1)

# out_pnorm = paddle.norm(x, p=2, axis=[0,1])

# out_pnorm = paddle.norm(x, p=np.inf)

# out_pnorm = paddle.norm(x, p=np.inf, axis=0)

# out_pnorm = paddle.norm(x, p=-np.inf)

# out_pnorm = paddle.norm(x, p=-np.inf, axis=0)

# out1 = paddle.normal(shape=[2, 3])

# mean_tensor = paddle.to_tensor([1.0, 2.0, 3.0])
# out2 = paddle.normal(mean=mean_tensor)

# std_tensor = paddle.to_tensor([1.0, 2.0, 3.0])
# out3 = paddle.normal(mean=mean_tensor, std=std_tensor)

# x = paddle.to_tensor([1, 2, 3])
# y = paddle.to_tensor([1, 3, 2])
# result1 = paddle.not_equal(x, y)
# print(result1)  

# import paddle
# npu_place = paddle.NPUPlace(0)

# x = paddle.full(shape=[4, 5, 7], fill_value=0, dtype='int32')
# numel = paddle.numel(x) 

# data1 = paddle.ones(shape=[3, 2])

# data2 = paddle.ones(shape=[2, 2], dtype='int32')

# shape = paddle.full(shape=[2], dtype='int32', fill_value=2)
# data3 = paddle.ones(shape=shape, dtype='int32')

# x = paddle.to_tensor([1,2,3])
# out1 = paddle.ones_like(x) 
# out2 = paddle.ones_like(x, dtype='int32') 

# weight_attr = paddle.ParamAttr(name="weight",
#                                learning_rate=0.5,
#                                regularizer=paddle.regularizer.L2Decay(1.0),
#                                trainable=True)
# print(weight_attr.name) 
# paddle.nn.Linear(3, 4, weight_attr=weight_attr)

# x = paddle.to_tensor([1, 2, 3], dtype='float32')

# res = paddle.pow(x, 2)
# print(res)

# res = paddle.pow(x, 2.5)
# print(res)

# y = paddle.to_tensor([2], dtype='float32')
# res = paddle.pow(x, y)
# print(res)

# x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
#                       [0.1, 0.2, 0.6, 0.7]])
# out1 = paddle.prod(x)

# out2 = paddle.prod(x, -1)

# out3 = paddle.prod(x, 0)

# out4 = paddle.prod(x, 0, keepdim=True)

# out5 = paddle.prod(x, 0, dtype='int64')

# y = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]],
#                       [[5.0, 6.0], [7.0, 8.0]]])
# out6 = paddle.prod(y, [0, 1])

# out7 = paddle.prod(y, (1, 2))

# out1 = paddle.rand(shape=[2, 3])

# dim1 = paddle.to_tensor([2], 'int64')
# dim2 = paddle.to_tensor([3], 'int32')
# out2 = paddle.rand(shape=[dim1, dim2, 2])

# shape_tensor = paddle.to_tensor([2, 3])
# out3 = paddle.rand(shape_tensor)

# out1 = paddle.randint(low=-5, high=5, shape=[3])

# dim1 = paddle.to_tensor([2], 'int64')
# dim2 = paddle.to_tensor([3], 'int32')
# out2 = paddle.randint(low=-5, high=5, shape=[dim1, dim2])

# shape_tensor = paddle.to_tensor(3)
# out3 = paddle.randint(low=-5, high=5, shape=shape_tensor)

# out4 = paddle.randint(low=-5, high=5, shape=[3], dtype='int32')

# out5 = paddle.randint(10)

# out1 = paddle.randn(shape=[2, 3])

# dim1 = paddle.to_tensor([2], 'int64')
# dim2 = paddle.to_tensor([3], 'int32')
# out2 = paddle.randn(shape=[dim1, dim2, 2])

# shape_tensor = paddle.to_tensor([2, 3])
# out3 = paddle.randn(shape_tensor)

# out1 = paddle.randperm(5)

# out2 = paddle.randperm(7, 'int32')

# input = paddle.rand((3, 100, 100))
# rank = paddle.rank(input)
# print(rank)

# x = paddle.to_tensor(
#     [[1 + 6j, 2 + 5j, 3 + 4j], [4 + 3j, 5 + 2j, 6 + 1j]])

# real_res = paddle.real(x)

# real_t = x.real()

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.reciprocal(x)
# print(out)

# x = paddle.rand([2, 4, 6], dtype="float32")
# positive_four = paddle.full([1], 4, "int32")

# out = paddle.reshape(x, [-1, 0, 3, 2])
# print(out)

# out = paddle.reshape(x, shape=[positive_four, 12])
# print(out)

# shape_tensor = paddle.to_tensor(np.array([8, 6]).astype("int32"))
# out = paddle.reshape(x, shape=shape_tensor)
# print(out)

# x[0, 0, 0] = 10.
# print(out[0, 0])

# x = paddle.to_tensor([[1.0, 2.0, 3.0],
#                       [4.0, 5.0, 6.0],
#                       [7.0, 8.0, 9.0]])
# out_z1 = paddle.roll(x, shifts=1)
# print(out_z1)

# out_z2 = paddle.roll(x, shifts=1, axis=0)
# print(out_z2)

# x = paddle.to_tensor([-0.5, -0.2, 0.6, 1.5])
# out = paddle.round(x)
# print(out)

# x = paddle.to_tensor([0.1, 0.2, 0.3, 0.4])
# out = paddle.rsqrt(x)
# print(out)

# emb = paddle.nn.Embedding(10, 10)
# layer_state_dict = emb.state_dict()

# paddle.save(layer_state_dict, "emb.pdparams")

# scheduler = paddle.optimizer.lr.NoamDecay(
#     d_model=0.01, warmup_steps=100, verbose=True)
# adam = paddle.optimizer.Adam(
#     learning_rate=scheduler,
#     parameters=emb.parameters())
# opt_state_dict = adam.state_dict()

# paddle.save(opt_state_dict, "adam.pdopt")

# paddle.save(emb.weight, "emb.weight.pdtensor")

# layer = paddle.nn.Linear(3, 4)
# adam = Adam(learning_rate=0.001, parameters=layer.parameters())
# obj = {'model': layer.state_dict(), 'opt': adam.state_dict(), 'epoch': 100}
# path = 'example/model.pdparams'
# paddle.save(obj, path)

# paddle.enable_static()

# x = paddle.static.data(name="x", shape=[None, 224], dtype='float32')
# z = paddle.static.nn.fc(x, 10)

# place = paddle.CPUPlace()
# exe = paddle.static.Executor(place)
# exe.run(paddle.static.default_startup_program())
# prog = paddle.static.default_main_program()
# for var in prog.list_vars():
#     if list(var.shape) == [224, 10]:
#         tensor = var.get_value()
#         break

# path_tensor = 'temp/tensor.pdtensor'
# paddle.save(tensor, path_tensor)

# path_state_dict = 'temp/model.pdparams'
# paddle.save(prog.state_dict("param"), path_tensor)

# paddle.enable_static()

# data = paddle.static.data(
#     name='x_static_save', shape=(None, 224), dtype='float32')
# y_static = z = paddle.static.nn.fc(data, 10)
# main_program = paddle.static.default_main_program()
# path = "example/main_program.pdmodel"
# paddle.save(main_program, path)

# paddle.disable_static()

# linear = Linear(5, 10)
# state_dict = linear.state_dict()
# byio = BytesIO()
# paddle.save(state_dict, byio)
# tensor = paddle.randn([2, 3], dtype='float32')
# paddle.save(tensor, byio)

# data = paddle.randn(shape=[2,3], dtype='float32')
# res = paddle.scale(data, scale=2.0, bias=1.0)

# data = paddle.randn(shape=[2, 3], dtype='float32')
# factor = paddle.to_tensor([2], dtype='float32')
# res = paddle.scale(data, scale=factor, bias=1.0)

# x = np.array([[1, 1], [2, 2], [3, 3]])
# index = np.array([2, 1, 0, 1])

# updates = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
# overwrite = False

# if not overwrite:
#     for i in range(len(index)):
#         x[index[i]] = np.zeros((2))
# for i in range(len(index)):
#     if (overwrite):
#         x[index[i]] = updates[i]
#     else:
#         x[index[i]] += updates[i]

# out = np.array([[3, 3], [6, 6], [1, 1]])
# out.shape 

# x = paddle.to_tensor([[1, 1], [2, 2], [3, 3]], dtype='float32')
# index = paddle.to_tensor([2, 1, 0, 1], dtype='int64')
# updates = paddle.to_tensor([[1, 1], [2, 2], [3, 3], [4, 4]], dtype='float32')

# output1 = paddle.scatter(x, index, updates, overwrite=False)

# output2 = paddle.scatter(x, index, updates, overwrite=True)

# index_data = np.array([[1, 1],
#                        [0, 1],
#                        [1, 3]]).astype(np.int64)
# index = paddle.to_tensor(index_data)
# updates = paddle.rand(shape=[3, 9, 10], dtype='float32')
# shape = [3, 5, 9, 10]

# output = paddle.scatter_nd(index, updates, shape)

# x = paddle.rand(shape=[3, 5, 9, 10], dtype='float32')
# updates = paddle.rand(shape=[3, 9, 10], dtype='float32')
# index_data = np.array([[1, 1],
#                        [0, 1],
#                        [1, 3]]).astype(np.int64)
# index = paddle.to_tensor(index_data)
# output = paddle.scatter_nd_add(x, index, updates)

# sorted_sequence = paddle.to_tensor([[1, 3, 5, 7, 9, 11],
#                                     [2, 4, 6, 8, 10, 12]], dtype='int32')
# values = paddle.to_tensor([[3, 6, 9, 10], [3, 6, 9, 10]], dtype='int32')
# out1 = paddle.searchsorted(sorted_sequence, values)
# print(out1)

# out2 = paddle.searchsorted(sorted_sequence, values, right=True)
# print(out2)

# sorted_sequence_1d = paddle.to_tensor([1, 3, 5, 7, 9, 11, 13])
# out3 = paddle.searchsorted(sorted_sequence_1d, values)
# print(out3)

# gen = paddle.seed(102)

# sts = paddle.get_cuda_rng_state()
# paddle.set_cuda_rng_state(sts)

# paddle.set_default_dtype("float32")

# paddle.set_flags({'FLAGS_eager_delete_tensor_gb': 1.0})

# x = paddle.ones([3, 2])
# x.stop_gradient = False
# with paddle.set_grad_enabled(False):
#     y = x * 2
#     with paddle.set_grad_enabled(True):
#         z = x * 2
# print(y.stop_gradient)   
# print(z.stop_gradient)   

# paddle.seed(10)
# a = paddle.rand([10, 20])
# paddle.set_printoptions(4, 100, 3)
# print(a)

# '''
# Tensor(shape=[10, 20], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#        [[0.0002, 0.8503, 0.0135, ..., 0.9508, 0.2621, 0.6661],
#         [0.9710, 0.2605, 0.9950, ..., 0.4427, 0.9241, 0.9363],
#         [0.0948, 0.3226, 0.9955, ..., 0.1198, 0.0889, 0.9231],
#         ...,
#         [0.7206, 0.0941, 0.5292, ..., 0.4856, 0.1379, 0.0351],
#         [0.1745, 0.5621, 0.3602, ..., 0.2998, 0.4011, 0.1764],
#         [0.0728, 0.7786, 0.0314, ..., 0.2583, 0.1654, 0.0637]])
# '''

# paddle.enable_static()

# inputs = fluid.data(name="x", shape=[3, 100, 100], dtype="float32")
# output = fluid.layers.shape(inputs)

# exe = fluid.Executor(fluid.CPUPlace())
# exe.run(fluid.default_startup_program())

# img = np.ones((3, 100, 100)).astype(np.float32)

# res = exe.run(fluid.default_main_program(), feed={'x':img}, fetch_list=[output])
# print(res) 
# shard_size = (index_num + nshards - 1) // nshards
# v = v - shard_id * shard_size if shard_id * shard_size <= v < (shard_id+1) * shard_size else ignore_value

# label = paddle.to_tensor([[16], [1]], "int64")
# shard_label = paddle.shard_index(input=label,
#                                  index_num=20,
#                                  nshards=2,
#                                  shard_id=0)
# print(shard_label)

# x = paddle.to_tensor([3.0, 0.0, -2.0, 1.7], dtype='float32')
# out = paddle.sign(x=x)
# print(out)  

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.sin(x)
# print(out)

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.sinh(x)
# print(out)

# input = paddle.rand(shape=[4, 5, 6], dtype='float32')

# axes = [0, 1, 2]
# starts = [-3, 0, 2]
# ends = [3, 2, 4]
# sliced_1 = paddle.slice(input, axes=axes, starts=starts, ends=ends)

# minus_3 = paddle.full([1], -3, "int32")
# sliced_2 = paddle.slice(input, axes=axes, starts=[minus_3, 0, 2], ends=ends)

# x = paddle.to_tensor([[[5,8,9,5],
#                        [0,0,1,7],
#                        [6,9,2,4]],
#                       [[5,2,4,2],
#                        [4,7,7,9],
#                        [1,7,0,6]]],
#                      dtype='float32')
# out1 = paddle.sort(x=x, axis=-1)
# out2 = paddle.sort(x=x, axis=0)
# out3 = paddle.sort(x=x, axis=1)
# print(out1)

# print(out2)

# print(out3)

# x = paddle.rand([3, 9, 5])

# out0, out1, out2 = paddle.split(x, num_or_sections=3, axis=1)
# print(out0.shape)  
# print(out1.shape)  
# print(out2.shape)  

# out0, out1, out2 = paddle.split(x, num_or_sections=[2, 3, 4], axis=1)
# print(out0.shape)  
# print(out1.shape)  
# print(out2.shape)  

# out0, out1, out2 = paddle.split(x, num_or_sections=[2, 3, -1], axis=1)
# print(out0.shape)  
# print(out1.shape)  
# print(out2.shape)  

# out0, out1, out2 = paddle.split(x, num_or_sections=3, axis=-2)
# print(out0.shape)  
# print(out1.shape)  
# print(out2.shape)  

# x = paddle.to_tensor([0.1, 0.2, 0.3, 0.4])
# out = paddle.sqrt(x)
# print(out)

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.square(x)
# print(out)

# x = paddle.rand([5, 1, 10])
# output = paddle.squeeze(x, axis=1)

# print(x.shape)  
# print(output.shape)  

# x[0, 0, 0] = 10.
# print(output[0, 0]) 

# x1 = paddle.to_tensor([[1.0, 2.0]])
# x2 = paddle.to_tensor([[3.0, 4.0]])
# x3 = paddle.to_tensor([[5.0, 6.0]])
# out = paddle.stack([x1, x2, x3], axis=0)
# print(out.shape)  
# print(out)

# out1 = paddle.standard_normal(shape=[2, 3])

# dim1 = paddle.to_tensor([2], 'int64')
# dim2 = paddle.to_tensor([3], 'int32')
# out2 = paddle.standard_normal(shape=[dim1, dim2, 2])

# shape_tensor = paddle.to_tensor([2, 3])
# out3 = paddle.standard_normal(shape_tensor)

# x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
# out = paddle.stanh(x, scale_a=0.67, scale_b=1.72) 

# x = paddle.to_tensor([[1.0, 2.0, 3.0], [1.0, 4.0, 5.0]])
# out1 = paddle.std(x)

# out2 = paddle.std(x, axis=1)

# x = paddle.zeros(shape=[3,4,5,6], dtype="float32")

# axes = [1, 2, 3]
# starts = [-3, 0, 2]
# ends = [3, 2, 4]
# strides_1 = [1, 1, 1]
# strides_2 = [1, 1, 2]
# sliced_1 = paddle.strided_slice(x, axes=axes, starts=starts, ends=ends, strides=strides_1)

# minus_3 = paddle.full(shape=[1], fill_value=-3, dtype='int32')
# sliced_2 = paddle.strided_slice(x, axes=axes, starts=[minus_3, 0, 2], ends=ends, strides=strides_2)

# x = paddle.to_tensor([[1, 2], [7, 8]])
# y = paddle.to_tensor([[5, 6], [3, 4]])
# res = paddle.subtract(x, y)
# print(res)

# x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
# y = paddle.to_tensor([1, 0, 4])
# res = paddle.subtract(x, y)
# print(res)

# x = paddle.to_tensor([2, np.nan, 5], dtype='float32')
# y = paddle.to_tensor([1, 4, np.nan], dtype='float32')
# res = paddle.subtract(x, y)
# print(res)

# x = paddle.to_tensor([5, np.inf, -np.inf], dtype='float64')
# y = paddle.to_tensor([1, 4, 5], dtype='float64')
# res = paddle.subtract(x, y)
# print(res)

# x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
#                       [0.1, 0.2, 0.6, 0.7]])
# out1 = paddle.sum(x)  
# out2 = paddle.sum(x, axis=0)  
# out3 = paddle.sum(x, axis=-1)  
# out4 = paddle.sum(x, axis=1, keepdim=True)  

# y = paddle.to_tensor([[[1, 2], [3, 4]],
#                       [[5, 6], [7, 8]]])
# out5 = paddle.sum(y, axis=[1, 2]) 
# out6 = paddle.sum(y, axis=[0, 1]) 

# x = paddle.to_tensor([[True, True, True, True],
#                       [False, False, False, False]])
# out7 = paddle.sum(x)  
# out8 = paddle.sum(x, axis=0)  
# out9 = paddle.sum(x, axis=1)  

# class LeNet(nn.Layer):
#     def __init__(self, num_classes=10):
#         super(LeNet, self).__init__()
#         self.num_classes = num_classes
#         self.features = nn.Sequential(
#             nn.Conv2D(
#                 1, 6, 3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2D(2, 2),
#             nn.Conv2D(
#                 6, 16, 5, stride=1, padding=0),
#             nn.ReLU(),
#             nn.MaxPool2D(2, 2))

#         if num_classes > 0:
#             self.fc = nn.Sequential(
#                 nn.Linear(400, 120),
#                 nn.Linear(120, 84),
#                 nn.Linear(
#                     84, 10))

#     def forward(self, inputs):
#         x = self.features(inputs)

#         if self.num_classes > 0:
#             x = paddle.flatten(x, 1)
#             x = self.fc(x)
#         return x

# lenet = LeNet()

# params_info = paddle.summary(lenet, (1, 1, 28, 28))
# print(params_info)

# class LeNetMultiInput(LeNet):

#     def forward(self, inputs, y):
#         x = self.features(inputs)

#         if self.num_classes > 0:
#             x = paddle.flatten(x, 1)
#             x = self.fc(x + y)
#         return x

# lenet_multi_input = LeNetMultiInput()

# params_info = paddle.summary(lenet_multi_input, [(1, 1, 28, 28), (1, 400)],
#                             dtypes=['float32', 'float32'])
# print(params_info)

# class LeNetListInput(LeNet):

#     def forward(self, inputs):
#         x = self.features(inputs[0])

#         if self.num_classes > 0:
#             x = paddle.flatten(x, 1)
#             x = self.fc(x + inputs[1])
#         return x

# lenet_list_input = LeNetListInput()
# input_data = [paddle.rand([1, 1, 28, 28]), paddle.rand([1, 400])]
# params_info = paddle.summary(lenet_list_input, input=input_data)
# print(params_info)

# class LeNetDictInput(LeNet):

#     def forward(self, inputs):
#         x = self.features(inputs['x1'])

#         if self.num_classes > 0:
#             x = paddle.flatten(x, 1)
#             x = self.fc(x + inputs['x2'])
#         return x

# lenet_dict_input = LeNetDictInput()
# input_data = {'x1': paddle.rand([1, 1, 28, 28]),
#               'x2': paddle.rand([1, 400])}
# params_info = paddle.summary(lenet_dict_input, input=input_data)
# print(params_info)

# x = paddle.ones(shape=[2, 3], dtype='int32')
# x_transposed = paddle.t(x)
# print(x_transposed.shape)

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.tan(x)
# print(out)

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.tanh(x)
# print(out)

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.abs(x)
# print(out)

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.acos(x)
# print(out)

# x = paddle.to_tensor([2, 3, 4], 'float64')
# y = paddle.to_tensor([1, 5, 2], 'float64')
# z = paddle.add(x, y)
# print(z)  

# input0 = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='float32')
# input1 = paddle.to_tensor([[7, 8, 9], [10, 11, 12]], dtype='float32')
# output = paddle.add_n([input0, input1])

# x = paddle.ones([2,2])
# y = paddle.ones([2,2])
# input = paddle.ones([2,2])

# out = paddle.addmm( input=input, x=x, y=y, beta=0.5, alpha=5.0 )

# print(out)

# x = paddle.assign(np.array([[1, 0], [1, 1]], dtype='int32'))
# print(x)
# x = paddle.cast(x, 'bool')

# out1 = paddle.all(x)  
# print(out1)

# out2 = paddle.all(x, axis=0)  
# print(out2)

# out3 = paddle.all(x, axis=-1)  
# print(out3)

# out4 = paddle.all(x, axis=1, keepdim=True)
# out4 = paddle.cast(out4, 'int32')  
# print(out4)

# x = paddle.to_tensor([10000., 1e-07])
# y = paddle.to_tensor([10000.1, 1e-08])
# result1 = paddle.allclose(x, y, rtol=1e-05, atol=1e-08,
#                         equal_nan=False, name="ignore_nan")
# np_result1 = result1.numpy()

# result2 = paddle.allclose(x, y, rtol=1e-05, atol=1e-08,
#                             equal_nan=True, name="equal_nan")
# np_result2 = result2.numpy()

# x = paddle.to_tensor([1.0, float('nan')])
# y = paddle.to_tensor([1.0, float('nan')])
# result1 = paddle.allclose(x, y, rtol=1e-05, atol=1e-08,
#                         equal_nan=False, name="ignore_nan")
# np_result1 = result1.numpy()

# result2 = paddle.allclose(x, y, rtol=1e-05, atol=1e-08,
#                             equal_nan=True, name="equal_nan")
# np_result2 = result2.numpy()

# x = paddle.assign(np.array([[1, 0], [1, 1]], dtype='int32'))
# print(x)
# x = paddle.cast(x, 'bool')

# out1 = paddle.any(x)  
# print(out1)

# out2 = paddle.any(x, axis=0)  
# print(out2)

# out3 = paddle.any(x, axis=-1)  
# print(out3)

# out4 = paddle.any(x, axis=1, keepdim=True)
# out4 = paddle.cast(out4, 'int32')  
# print(out4)

# x =  paddle.to_tensor([[5,8,9,5],
#                          [0,0,1,7],
#                          [6,9,2,4]])
# out1 = paddle.argmax(x)
# print(out1) 
# out2 = paddle.argmax(x, axis=1)
# print(out2)

# out3 = paddle.argmax(x, axis=-1)
# print(out3)

# x =  paddle.to_tensor([[5,8,9,5],
#                          [0,0,1,7],
#                          [6,9,2,4]])
# out1 = paddle.argmin(x)
# print(out1) 
# out2 = paddle.argmin(x, axis=1)
# print(out2)

# out3 = paddle.argmin(x, axis=-1)
# print(out3)

# x = paddle.to_tensor([[[5,8,9,5],
#                        [0,0,1,7],
#                        [6,9,2,4]],
#                       [[5,2,4,2],
#                        [4,7,7,9],
#                        [1,7,0,6]]],
#                     dtype='float32')
# out1 = paddle.argsort(x=x, axis=-1)
# out2 = paddle.argsort(x=x, axis=0)
# out3 = paddle.argsort(x=x, axis=1)
# print(out1)

# print(out2)

# print(out3)

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.asin(x)
# print(out)

# original_tensor = paddle.ones([2, 2])
# print("original tensor's dtype is: {}".format(original_tensor.dtype))
# new_tensor = original_tensor.astype('float32')
# print("new tensor's dtype is: {}".format(new_tensor.dtype))

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.atan(x)
# print(out)

# x = paddle.to_tensor(5., stop_gradient=False)
# for i in range(5):
#     y = paddle.pow(x, 4.0)
#     y.backward()
#     print("{}: {}".format(i, x.grad))

# x.clear_grad()
# print("{}".format(x.grad))

# grad_tensor=paddle.to_tensor(2.)
# for i in range(5):
#     y = paddle.pow(x, 4.0)
#     y.backward(grad_tensor)
#     print("{}: {}".format(i, x.grad))

# x = paddle.to_tensor([1, 2, 1, 4, 5])
# result1 = paddle.bincount(x)
# print(result1) 

# w = paddle.to_tensor([2.1, 0.4, 0.1, 0.5, 0.5])
# result2 = paddle.bincount(x, weights=w)
# print(result2) 

# x = paddle.to_tensor([-5, -1, 1])
# y = paddle.to_tensor([4,  2, -3])
# res = paddle.bitwise_and(x, y)
# print(res)  

# x = paddle.to_tensor([-5, -1, 1])
# res = paddle.bitwise_not(x)
# print(res) 

# x = paddle.to_tensor([-5, -1, 1])
# y = paddle.to_tensor([4,  2, -3])
# res = paddle.bitwise_or(x, y)
# print(res)  

# x = paddle.to_tensor([-5, -1, 1])
# y = paddle.to_tensor([4,  2, -3])
# res = paddle.bitwise_xor(x, y)
# print(res) 

# x = paddle.to_tensor([[[1.0, 1.0, 1.0],
#                     [2.0, 2.0, 2.0]],
#                     [[3.0, 3.0, 3.0],
#                     [4.0, 4.0, 4.0]]])
# y = paddle.to_tensor([[[1.0, 1.0],[2.0, 2.0],[3.0, 3.0]],
#                     [[4.0, 4.0],[5.0, 5.0],[6.0, 6.0]]])
# out = paddle.bmm(x, y)

# out_np = out.numpy()

# shape = paddle.broadcast_shape([2, 1, 3], [1, 3, 1])

# x1 = paddle.rand([1, 2, 3, 4]).astype('float32')
# x2 = paddle.rand([1, 2, 1, 4]).astype('float32')
# x3 = paddle.rand([1, 1, 3, 1]).astype('float32')
# out1, out2, out3 = paddle.broadcast_tensors(input=[x1, x2, x3])

# data = paddle.to_tensor([1, 2, 3], dtype='int32')
# out = paddle.broadcast_to(data, shape=[2, 3])
# print(out)

# x = paddle.to_tensor([2, 3, 4], 'float64')
# y = paddle.cast(x, 'uint8')

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.ceil(x)
# print(out)

# a = np.random.rand(3, 3)
# a_t = np.transpose(a, [1, 0])
# x_data = np.matmul(a, a_t) + 1e-03
# x = paddle.to_tensor(x_data)
# out = paddle.cholesky(x, upper=False)
# print(out)

# x_np = np.random.random([3, 9, 5]).astype("int32")
# x = paddle.to_tensor(x_np)

# out0, out1, out2 = paddle.chunk(x, chunks=3, axis=1)

# out0, out1, out2 = paddle.chunk(x, chunks=3, axis=-2)

# input = paddle.uniform([10, 2])
# linear = paddle.nn.Linear(2, 3)
# out = linear(input)
# out.backward()
# print("Before clear_gradient, linear.weight.grad: {}".format(linear.weight.grad))
# linear.weight.clear_gradient()
# print("After clear_gradient, linear.weight.grad: {}".format(linear.weight.grad))

# x1 = paddle.to_tensor([[1.2, 3.5], [4.5, 6.4]], 'float32')
# out1 = paddle.clip(x1, min=3.5, max=5.0)
# out2 = paddle.clip(x1, min=2.5)
# print(out1)

# print(out2)

# x = paddle.to_tensor(1.0, stop_gradient=False)
# clone_x = x.clone()
# y = clone_x**2
# y.backward()
# print(clone_x.stop_gradient) 
# print(clone_x.grad)          
# print(x.stop_gradient)       
# print(x.grad)                

# x = paddle.to_tensor(1.0)
# clone_x = x.clone()
# clone_x.stop_gradient = False
# z = clone_x**3
# z.backward()
# print(clone_x.stop_gradient) 
# print(clone_x.grad)          
# print(x.stop_gradient) 
# print(x.grad)          

# x1 = paddle.to_tensor([[1, 2, 3],
#                        [4, 5, 6]])
# x2 = paddle.to_tensor([[11, 12, 13],
#                        [14, 15, 16]])
# x3 = paddle.to_tensor([[21, 22],
#                        [23, 24]])
# zero = paddle.full(shape=[1], dtype='int32', fill_value=0)

# out1 = paddle.concat(x=[x1, x2, x3], axis=-1)
# out2 = paddle.concat(x=[x1, x2], axis=0)
# out3 = paddle.concat(x=[x1, x2], axis=zero)

# x = paddle.to_tensor([[1., 0, -1], [0, 1, 0], [1, 0, 1]])

# out = paddle.linalg.cond(x)

# out_fro = paddle.linalg.cond(x, p='fro')

# out_nuc = paddle.linalg.cond(x, p='nuc')

# out_1 = paddle.linalg.cond(x, p=1)

# out_minus_1 = paddle.linalg.cond(x, p=-1)

# out_2 = paddle.linalg.cond(x, p=2)

# out_minus_2 = paddle.linalg.cond(x, p=-2)

# out_inf = paddle.linalg.cond(x, p=np.inf)

# out_minus_inf = paddle.linalg.cond(x, p=-np.inf)

# a = paddle.to_tensor(np.random.randn(2, 4, 4).astype('float32'))

# a_cond_fro = paddle.linalg.cond(a, p='fro')

# b = paddle.to_tensor(np.random.randn(2, 3, 4).astype('float64'))

# b_cond_2 = paddle.linalg.cond(b, p=2)

# data=paddle.to_tensor([[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j]])

# conj_data=paddle.conj(data)

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.cos(x)
# print(out)

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.cosh(x)
# print(out)

# x = paddle.to_tensor(1.0, place=paddle.CUDAPlace(0))
# print(x.place)    

# y = x.cpu()
# print(y.place)    

# x = paddle.to_tensor([[1.0, 1.0, 1.0],
#                       [2.0, 2.0, 2.0],
#                       [3.0, 3.0, 3.0]])
# y = paddle.to_tensor([[1.0, 1.0, 1.0],
#                       [1.0, 1.0, 1.0],
#                       [1.0, 1.0, 1.0]])

# z1 = paddle.cross(x, y)

# z2 = paddle.cross(x, y, axis=1)

# x = paddle.to_tensor(1.0, place=paddle.CPUPlace())
# print(x.place)        

# y = x.cuda()
# print(y.place)        

# y = x.cuda(None)
# print(y.place)        

# y = x.cuda(1)
# print(y.place)        

# data = paddle.arange(12)
# data = paddle.reshape(data, (3, 4))

# y = paddle.cumprod(data, dim=0)

# y = paddle.cumprod(data, dim=-1)

# y = paddle.cumprod(data, dim=1, dtype='float64')

# print(y.dtype)

# data = paddle.arange(12)
# data = paddle.reshape(data, (3, 4))

# y = paddle.cumsum(data)

# y = paddle.cumsum(data, axis=0)

# y = paddle.cumsum(data, axis=-1)

# y = paddle.cumsum(data, dtype='float64')
# print(y.dtype)

# x = paddle.to_tensor(1.0, stop_gradient=False)
# detach_x = x.detach()
# detach_x[:] = 10.0
# print(x)  
          
# y = x**2
# y.backward()
# print(x.grad)         
# print(detach_x.grad)  

# detach_x.stop_gradient = False 
# z = detach_x**3
# z.backward()

# print(x.grad)         
# print(detach_x.grad)  

# y = 2 * x
# detach_x[:] = 5.0
# y.backward()

# x = paddle.rand([2,2,3],'float32')
# print(x)

# out1 = paddle.diagonal(x)
# print(out1)

# out2 = paddle.diagonal(x, offset=0, axis1=2, axis2=1)
# print(out2)

# out3 = paddle.diagonal(x, offset=1, axis1=0, axis2=1)
# print(out3)

# out4 = paddle.diagonal(x, offset=0, axis1=1, axis2=2)
# print(out4)

# data = paddle.to_tensor([[1, 1.5], [0, -2.2]], dtype='float32')
# res = paddle.digamma(data)
# print(res)

# x = paddle.to_tensor(np.array([[3, 3],[3, 3]]), "float32")
# y = paddle.to_tensor(np.array([[3, 3],[3, 1]]), "float32")
# out = paddle.dist(x, y, 0)
# print(out) 

# out = paddle.dist(x, y, 2)
# print(out) 

# out = paddle.dist(x, y, float("inf"))
# print(out) 

# out = paddle.dist(x, y, float("-inf"))
# print(out) 

# x = paddle.to_tensor([2, 3, 4], dtype='float64')
# y = paddle.to_tensor([1, 5, 2], dtype='float64')
# z = paddle.divide(x, y)
# print(z)  

# x_data = np.random.uniform(0.1, 1, [10]).astype(np.float32)
# y_data = np.random.uniform(1, 3, [10]).astype(np.float32)
# x = paddle.to_tensor(x_data)
# y = paddle.to_tensor(y_data)
# z = paddle.dot(x, y)
# print(z)

# paddle.device.set_device("cpu")

# x_data = np.array([[1.6707249, 7.2249975, 6.5045543],
#                    [9.956216,  8.749598,  6.066444 ],
#                    [4.4251957, 1.7983172, 0.370647 ]]).astype("float32")
# x = paddle.to_tensor(x_data)
# w, v = paddle.linalg.eig(x)
# print(w)

# print(v)

# paddle.set_device("cpu")
# paddle.seed(1234)

# x = paddle.rand(shape=[3, 3], dtype='float64')

# print(paddle.linalg.eigvals(x))

# x_data = np.array([[1, -2j], [2j, 5]])
# x = paddle.to_tensor(x_data)
# out_value = paddle.eigvalsh(x, UPLO='L')
# print(out_value)

# x = paddle.to_tensor([1, 2, 3])
# y = paddle.to_tensor([1, 3, 2])
# result1 = paddle.equal(x, y)
# print(result1)  

# x = paddle.to_tensor([1, 2, 3])
# y = paddle.to_tensor([1, 2, 3])
# z = paddle.to_tensor([1, 4, 3])
# result1 = paddle.equal_all(x, y)
# print(result1) 
# result2 = paddle.equal_all(x, z)
# print(result2) 

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.erf(x)
# print(out)

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.exp(x)
# print(out)

# data = paddle.to_tensor([1, 2, 3], dtype='int32')
# out = paddle.expand(data, shape=[2, 3])
# print(out)

# data_x = paddle.to_tensor([1, 2, 3], 'int32')
# data_y = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], 'int32')
# out = paddle.expand_as(data_x, data_y)
# np_out = out.numpy()

# tensor = paddle.to_tensor([0, 1, 2, 3, 4])

# tensor.fill_(0)
# print(tensor.tolist())   

# x = paddle.ones((4, 3)) * 2
# x.fill_diagonal_(1.0)
# print(x.tolist())   

# x = paddle.ones((4, 3)) * 2
# y = paddle.ones((3,))
# nx = x.fill_diagonal_tensor(y)
# print(nx.tolist())   

# x = paddle.ones((4, 3)) * 2
# y = paddle.ones((3,))
# x.fill_diagonal_tensor_(y)
# print(x.tolist())   

# image_shape=(2, 3, 4, 4)

# x = paddle.arange(end=image_shape[0] * image_shape[1] * image_shape[2] * image_shape[3])
# img = paddle.reshape(x, image_shape)

# out = paddle.flatten(img, start_axis=1, stop_axis=2)

# img[0, 0, 0, 0] = -1
# print(out[0, 0, 0]) 

# image_shape=(3, 2, 2)
# x = np.arange(image_shape[0] * image_shape[1] * image_shape[2]).reshape(image_shape)
# x = x.astype('float32')
# img = paddle.to_tensor(x)
# tmp = paddle.flip(img, [0,1])
# print(tmp) 

# out = paddle.flip(tmp,-1)
# print(out) 

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.floor(x)
# print(out)

# x = paddle.to_tensor([2, 3, 8, 7])
# y = paddle.to_tensor([1, 5, 3, 3])
# z = paddle.floor_divide(x, y)
# print(z)  

# x = paddle.to_tensor([2, 3, 8, 7])
# y = paddle.to_tensor([1, 5, 3, 3])
# z = paddle.remainder(x, y)
# print(z)  

# input = paddle.to_tensor([[1,2],[3,4],[5,6]])
# index = paddle.to_tensor([0,1])
# output = paddle.gather(input, index, axis=0)

# x = paddle.to_tensor([[[1, 2], [3, 4], [5, 6]],
#                       [[7, 8], [9, 10], [11, 12]]])
# index = paddle.to_tensor([[0, 1]])

# output = paddle.gather_nd(x, index) 

# x = paddle.to_tensor(5., stop_gradient=False)
# y = paddle.pow(x, 4.0)
# y.backward()
# print("grad of x: {}".format(x.grad))

# x = paddle.to_tensor(5., stop_gradient=False)
# y = paddle.pow(x, 4.0)
# y.backward()
# print("grad of x: {}".format(x.gradient()))

# x = paddle.to_tensor([1, 2, 3])
# y = paddle.to_tensor([1, 3, 2])
# result1 = paddle.greater_equal(x, y)
# print(result1)  

# x = paddle.to_tensor([1, 2, 3])
# y = paddle.to_tensor([1, 3, 2])
# result1 = paddle.greater_than(x, y)
# print(result1)  

# inputs = paddle.to_tensor([1, 2, 1])
# result = paddle.histogram(inputs, bins=4, min=0, max=3)
# print(result) 

# x = paddle.to_tensor(
#     [[1 + 6j, 2 + 5j, 3 + 4j], [4 + 3j, 5 + 2j, 6 + 1j]])

# imag_res = paddle.imag(x)

# imag_t = x.imag()

# data = paddle.zeros(shape=[1], dtype='float32')
# counter = paddle.increment(data)

# x = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0],
#                       [5.0, 6.0, 7.0, 8.0],
#                       [9.0, 10.0, 11.0, 12.0]], dtype='float32')
# index = paddle.to_tensor([[0, 1, 2],
#                           [1, 2, 3],
#                           [0, 0, 0]], dtype='int32')
# target = paddle.to_tensor([[100, 200, 300, 400],
#                            [500, 600, 700, 800],
#                            [900, 1000, 1100, 1200]], dtype='int32')
# out_z1 = paddle.index_sample(x, index)
# print(out_z1)

# top_value, top_index = paddle.topk(x, k=2)
# out_z2 = paddle.index_sample(target, top_index)
# print(top_value)

# print(top_index)

# print(out_z2)

# x = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0],
#                       [5.0, 6.0, 7.0, 8.0],
#                       [9.0, 10.0, 11.0, 12.0]])
# index = paddle.to_tensor([0, 1, 1], dtype='int32')
# out_z1 = paddle.index_select(x=x, index=index)

# out_z2 = paddle.index_select(x=x, index=index, axis=1)

# var = paddle.ones(shape=[4, 2, 3], dtype="float32")
# print(var.inplace_version)  

# var[1] = 2.2
# print(var.inplace_version)  

# mat = paddle.to_tensor([[2, 0], [0, 2]], dtype='float32')
# inv = paddle.inverse(mat)
# print(inv) 

# input = paddle.rand(shape=[4, 32, 32], dtype='float32')
# res = paddle.is_empty(x=input)
# print("res:", res)

# x = paddle.to_tensor(1.)
# print(x.is_leaf) 

# x = paddle.to_tensor(1., stop_gradient=True)
# y = x + 1
# print(x.is_leaf) 
# print(y.is_leaf) 

# x = paddle.to_tensor(1., stop_gradient=False)
# y = x + 1
# print(x.is_leaf) 
# print(y.is_leaf) 

# input1 = paddle.rand(shape=[2, 3, 5], dtype='float32')
# check = paddle.is_tensor(input1)
# print(check)  

# input3 = [1, 4]
# check = paddle.is_tensor(input3)
# print(check)  

# x = paddle.to_tensor([float('-inf'), -2, 3.6, float('inf'), 0, float('-nan'), float('nan')])
# out = paddle.tensor.isfinite(x)
# print(out)  

# x = paddle.to_tensor([float('-inf'), -2, 3.6, float('inf'), 0, float('-nan'), float('nan')])
# out = paddle.tensor.isinf(x)
# print(out)  

# x = paddle.to_tensor([float('-inf'), -2, 3.6, float('inf'), 0, float('-nan'), float('nan')])
# out = paddle.tensor.isnan(x)
# print(out)  

# x = paddle.to_tensor(1)
# print(x.item())             
# print(type(x.item()))       

# x = paddle.to_tensor(1.0)
# print(x.item())             
# print(type(x.item()))       

# x = paddle.to_tensor(True)
# print(x.item())             
# print(type(x.item()))       

# x = paddle.to_tensor(1+1j)
# print(x.item())             
# print(type(x.item()))       

# x = paddle.to_tensor([[1.1, 2.2, 3.3]])
# print(x.item(2))            
# print(x.item(0, 2))         

# x = paddle.to_tensor([[1, 2], [3, 4]], dtype='int64')
# y = paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='int64')
# out = paddle.kron(x, y)
# print(out)

# x = paddle.to_tensor([1, 2, 3])
# y = paddle.to_tensor([1, 3, 2])
# result1 = paddle.less_equal(x, y)
# print(result1)  

# x = paddle.to_tensor([1, 2, 3])
# y = paddle.to_tensor([1, 3, 2])
# result1 = paddle.less_than(x, y)
# print(result1)  

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.lgamma(x)
# print(out)

# x = [[2,3,4], [7,8,9]]
# x = paddle.to_tensor(x, dtype='float32')
# res = paddle.log(x)

# x_i = paddle.to_tensor([[1.0], [10.0]])
# res = paddle.log10(x_i) 

# x_i = paddle.full(shape=[1], fill_value=10, dtype='float32')
# paddle.to_tensor(x_i)
# res = paddle.log10(x_i)
# print(res) 

# x_i = paddle.full(shape=[1], fill_value=10, dtype='float64')
# paddle.to_tensor(x_i)
# res = paddle.log10(x_i)
# print(res) 

# data = paddle.to_tensor([[0], [1]], dtype='float32')
# res = paddle.log1p(data)

# x_i = paddle.to_tensor([[1.0], [2.0]])
# res = paddle.log2(x_i) 

# x_i = paddle.full(shape=[1], fill_value=2, dtype='float32')
# paddle.to_tensor(x_i)
# res = paddle.log2(x_i)
# print(res) 

# x_i = paddle.full(shape=[1], fill_value=2, dtype='float64')
# paddle.to_tensor(x_i)
# res = paddle.log2(x_i)
# print(res) 

# x = paddle.to_tensor([True])
# y = paddle.to_tensor([True, False, True, False])
# res = paddle.logical_and(x, y)
# print(res) 

# x = paddle.to_tensor([True, False, True, False])
# res = paddle.logical_not(x)
# print(res) 

# x_data = np.array([True, False], dtype=np.bool).reshape(2, 1)
# y_data = np.array([True, False, True, False], dtype=np.bool).reshape(2, 2)
# x = paddle.to_tensor(x_data)
# y = paddle.to_tensor(y_data)
# res = paddle.logical_or(x, y)
# print(res) 

# x_data = np.array([True, False], dtype=np.bool).reshape([2, 1])
# y_data = np.array([True, False, True, False], dtype=np.bool).reshape([2, 2])
# x = paddle.to_tensor(x_data)
# y = paddle.to_tensor(y_data)
# res = paddle.logical_xor(x, y)
# print(res) 

# x = paddle.to_tensor([[-1.5, 0., 2.], [3., 1.2, -2.4]])
# out1 = paddle.logsumexp(x) 
# out2 = paddle.logsumexp(x, 1) 

# x = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0],
#                       [5.0, 6.0, 7.0, 8.0],
#                       [9.0, 10.0, 11.0, 12.0]])
# mask = paddle.to_tensor([[True, False, False, False],
#                          [True, True, False, False],
#                          [True, False, False, False]])
# out = paddle.masked_select(x, mask)

# x_data = np.random.random([10]).astype(np.float32)
# y_data = np.random.random([10]).astype(np.float32)
# x = paddle.to_tensor(x_data)
# y = paddle.to_tensor(y_data)
# z = paddle.matmul(x, y)
# print(z.numpy().shape)

# x_data = np.random.random([10, 5]).astype(np.float32)
# y_data = np.random.random([5]).astype(np.float32)
# x = paddle.to_tensor(x_data)
# y = paddle.to_tensor(y_data)
# z = paddle.matmul(x, y)
# print(z.numpy().shape)

# x_data = np.random.random([10, 5, 2]).astype(np.float32)
# y_data = np.random.random([2]).astype(np.float32)
# x = paddle.to_tensor(x_data)
# y = paddle.to_tensor(y_data)
# z = paddle.matmul(x, y)
# print(z.numpy().shape)

# x_data = np.random.random([10, 5, 2]).astype(np.float32)
# y_data = np.random.random([10, 2, 5]).astype(np.float32)
# x = paddle.to_tensor(x_data)
# y = paddle.to_tensor(y_data)
# z = paddle.matmul(x, y)
# print(z.numpy().shape)

# x_data = np.random.random([10, 1, 5, 2]).astype(np.float32)
# y_data = np.random.random([1, 3, 2, 5]).astype(np.float32)
# x = paddle.to_tensor(x_data)
# y = paddle.to_tensor(y_data)
# z = paddle.matmul(x, y)
# print(z.numpy().shape)

# x = paddle.to_tensor([[1, 2, 3],
#                       [1, 4, 9],
#                       [1, 8, 27]], dtype='float64')
# print(paddle.linalg.matrix_power(x, 2))

# print(paddle.linalg.matrix_power(x, 0))

# print(paddle.linalg.matrix_power(x, -2))

# x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
#                       [0.1, 0.2, 0.6, 0.7]])
# result1 = paddle.max(x)
# print(result1)

# result2 = paddle.max(x, axis=0)
# print(result2)

# result3 = paddle.max(x, axis=-1)
# print(result3)

# result4 = paddle.max(x, axis=1, keepdim=True)
# print(result4)

# y = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]],
#                       [[5.0, 6.0], [7.0, 8.0]]])
# result5 = paddle.max(y, axis=[1, 2])
# print(result5)

# result6 = paddle.max(y, axis=[0, 1])
# print(result6)

# x = paddle.to_tensor([[1, 2], [7, 8]])
# y = paddle.to_tensor([[3, 4], [5, 6]])
# res = paddle.maximum(x, y)
# print(res)

# x = paddle.to_tensor([[1, 2, 3], [1, 2, 3]])
# y = paddle.to_tensor([3, 0, 4])
# res = paddle.maximum(x, y)
# print(res)

# x = paddle.to_tensor([2, 3, 5], dtype='float32')
# y = paddle.to_tensor([1, np.nan, np.nan], dtype='float32')
# res = paddle.maximum(x, y)
# print(res)

# x = paddle.to_tensor([5, 3, np.inf], dtype='float32')
# y = paddle.to_tensor([1, -np.inf, 5], dtype='float32')
# res = paddle.maximum(x, y)
# print(res)

# x = paddle.to_tensor([[[1., 2., 3., 4.],
#                        [5., 6., 7., 8.],
#                        [9., 10., 11., 12.]],
#                       [[13., 14., 15., 16.],
#                        [17., 18., 19., 20.],
#                        [21., 22., 23., 24.]]])
# out1 = paddle.mean(x)

# out2 = paddle.mean(x, axis=-1)

# out3 = paddle.mean(x, axis=-1, keepdim=True)

# out4 = paddle.mean(x, axis=[0, 2])

# x = paddle.arange(12).reshape([3, 4])

# y1 = paddle.median(x)

# y2 = paddle.median(x, axis=0)

# y3 = paddle.median(x, axis=1)

# y4 = paddle.median(x, axis=0, keepdim=True)

# x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
#                       [0.1, 0.2, 0.6, 0.7]])
# result1 = paddle.min(x)
# print(result1)

# result2 = paddle.min(x, axis=0)
# print(result2)

# result3 = paddle.min(x, axis=-1)
# print(result3)

# result4 = paddle.min(x, axis=1, keepdim=True)
# print(result4)

# y = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]],
#                       [[5.0, 6.0], [7.0, 8.0]]])
# result5 = paddle.min(y, axis=[1, 2])
# print(result5)

# result6 = paddle.min(y, axis=[0, 1])
# print(result6)

# x = paddle.to_tensor([[1, 2], [7, 8]])
# y = paddle.to_tensor([[3, 4], [5, 6]])
# res = paddle.minimum(x, y)
# print(res)

# x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
# y = paddle.to_tensor([3, 0, 4])
# res = paddle.minimum(x, y)
# print(res)

# x = paddle.to_tensor([2, 3, 5], dtype='float32')
# y = paddle.to_tensor([1, np.nan, np.nan], dtype='float32')
# res = paddle.minimum(x, y)
# print(res)

# x = paddle.to_tensor([5, 3, np.inf], dtype='float64')
# y = paddle.to_tensor([1, -np.inf, 5], dtype='float64')
# res = paddle.minimum(x, y)
# print(res)

# input = paddle.arange(1, 7).reshape((3, 2)).astype('float32')
# mat2 = paddle.arange(1, 9).reshape((2, 4)).astype('float32')
# out = paddle.mm(input, mat2)
# print(out)

# x = paddle.to_tensor([2, 3, 8, 7])
# y = paddle.to_tensor([1, 5, 3, 3])
# z = paddle.remainder(x, y)
# print(z)  

# A_data = np.random.random([3, 4]).astype(np.float32)
# B_data = np.random.random([4, 5]).astype(np.float32)
# A = paddle.to_tensor(A_data)
# B = paddle.to_tensor(B_data)
# out = paddle.linalg.multi_dot([A, B])
# print(out.numpy().shape)

# A_data = np.random.random([10, 5]).astype(np.float32)
# B_data = np.random.random([5, 8]).astype(np.float32)
# C_data = np.random.random([8, 7]).astype(np.float32)
# A = paddle.to_tensor(A_data)
# B = paddle.to_tensor(B_data)
# C = paddle.to_tensor(C_data)
# out = paddle.linalg.multi_dot([A, B, C])
# print(out.numpy().shape)

# img1 = np.array([[1, 2], [3, 4]]).astype(np.float32)
# img2 = np.array([[5, 6], [7, 8]]).astype(np.float32)
# inputs = [paddle.to_tensor(img1), paddle.to_tensor(img2)]
# index = paddle.to_tensor(np.array([[1], [0]]).astype(np.int32))
# res = paddle.multiplex(inputs, index)
# print(res) 

# x = paddle.to_tensor([[1, 2], [3, 4]])
# y = paddle.to_tensor([[5, 6], [7, 8]])
# res = paddle.multiply(x, y)
# print(res) 

# x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
# y = paddle.to_tensor([2])
# res = paddle.multiply(x, y)
# print(res) 

# x_data = np.array([[2, 1, 3], [3, 0, 1]]).astype("float64")
# x = paddle.to_tensor(x_data)
# vec_data = np.array([3, 5, 1])
# vec = paddle.to_tensor(vec_data).astype("float64")
# out = paddle.mv(x, vec)

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.neg(x)
# print(out)

# x1 = paddle.to_tensor([[1.0, 0.0, 0.0],
#                        [0.0, 2.0, 0.0],
#                        [0.0, 0.0, 3.0]])
# x2 = paddle.to_tensor([0.0, 1.0, 0.0, 3.0])
# out_z1 = paddle.nonzero(x1)
# print(out_z1)

# out_z1_tuple = paddle.nonzero(x1, as_tuple=True)
# for out in out_z1_tuple:
#     print(out)

# out_z2 = paddle.nonzero(x2)
# print(out_z2)

# out_z2_tuple = paddle.nonzero(x2, as_tuple=True)
# for out in out_z2_tuple:
#     print(out)

# shape=[2, 3, 4]
# np_input = np.arange(24).astype('float32') - 12
# np_input = np_input.reshape(shape)
# x = paddle.to_tensor(np_input)

# out_fro = paddle.norm(x, p='fro', axis=[0,1])

# out_pnorm = paddle.norm(x, p=2, axis=-1)

# out_pnorm = paddle.norm(x, p=2, axis=[0,1])

# out_pnorm = paddle.norm(x, p=np.inf)

# out_pnorm = paddle.norm(x, p=np.inf, axis=0)

# out_pnorm = paddle.norm(x, p=-np.inf)

# out_pnorm = paddle.norm(x, p=-np.inf, axis=0)

# x = paddle.to_tensor([1, 2, 3])
# y = paddle.to_tensor([1, 3, 2])
# result1 = paddle.not_equal(x, y)
# print(result1)  

# x = paddle.full(shape=[4, 5, 7], fill_value=0, dtype='int32')
# numel = paddle.numel(x) 

# data = np.random.uniform(-1, 1, [30, 10, 32]).astype('float32')
# linear = paddle.nn.Linear(32, 64)
# data = paddle.to_tensor(data)
# x = linear(data)
# print(x.numpy())

# x = paddle.to_tensor(1.0, place=paddle.CUDAPlace(0))
# print(x.place)      

# y = x.pin_memory()
# print(y.place)      

# x = paddle.to_tensor([1, 2, 3], dtype='float32')

# res = paddle.pow(x, 2)
# print(res)

# res = paddle.pow(x, 2.5)
# print(res)

# y = paddle.to_tensor([2], dtype='float32')
# res = paddle.pow(x, y)
# print(res)

# x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
#                       [0.1, 0.2, 0.6, 0.7]])
# out1 = paddle.prod(x)

# out2 = paddle.prod(x, -1)

# out3 = paddle.prod(x, 0)

# out4 = paddle.prod(x, 0, keepdim=True)

# out5 = paddle.prod(x, 0, dtype='int64')

# y = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]],
#                       [[5.0, 6.0], [7.0, 8.0]]])
# out6 = paddle.prod(y, [0, 1])

# out7 = paddle.prod(y, (1, 2))

# x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).astype('float64')
# q, r = paddle.linalg.qr(x)
# print (q)
# print (r)

# input = paddle.rand((3, 100, 100))
# rank = paddle.rank(input)
# print(rank)

# x = paddle.to_tensor(
#     [[1 + 6j, 2 + 5j, 3 + 4j], [4 + 3j, 5 + 2j, 6 + 1j]])

# real_res = paddle.real(x)

# real_t = x.real()

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.reciprocal(x)
# print(out)

# def print_hook_fn(grad):
#     print(grad)

# def double_hook_fn(grad):
#     grad = grad * 2
#     return grad

# x = paddle.to_tensor([0., 1., 2., 3.], stop_gradient=False)
# y = paddle.to_tensor([4., 5., 6., 7.], stop_gradient=False)
# z = paddle.to_tensor([1., 2., 3., 4.])

# h = x.register_hook(print_hook_fn)
# x.register_hook(double_hook_fn)

# w = x + y

# w.register_hook(lambda grad: grad * 2)

# o = z.matmul(w)
# o.backward()

# print("w.grad:", w.grad) 
# print("x.grad:", x.grad) 
# print("y.grad:", y.grad) 

# h.remove()

# x = paddle.to_tensor([2, 3, 8, 7])
# y = paddle.to_tensor([1, 5, 3, 3])
# z = paddle.remainder(x, y)
# print(z)  

# x = paddle.rand([2, 4, 6], dtype="float32")
# positive_four = paddle.full([1], 4, "int32")

# out = paddle.reshape(x, [-1, 0, 3, 2])
# print(out)

# out = paddle.reshape(x, shape=[positive_four, 12])
# print(out)

# shape_tensor = paddle.to_tensor(np.array([8, 6]).astype("int32"))
# out = paddle.reshape(x, shape=shape_tensor)
# print(out)

# x[0, 0, 0] = 10.
# print(out[0, 0])

# image_shape=(3, 2, 2)
# x = np.arange(image_shape[0] * image_shape[1] * image_shape[2]).reshape(image_shape)
# x = x.astype('float32')
# img = paddle.to_tensor(x)
# tmp = paddle.flip(img, [0,1])
# print(tmp) 

# out = paddle.flip(tmp,-1)
# print(out) 

# x = paddle.to_tensor([[1.0, 2.0, 3.0],
#                       [4.0, 5.0, 6.0],
#                       [7.0, 8.0, 9.0]])
# out_z1 = paddle.roll(x, shifts=1)
# print(out_z1)

# out_z2 = paddle.roll(x, shifts=1, axis=0)
# print(out_z2)

# x = paddle.to_tensor([-0.5, -0.2, 0.6, 1.5])
# out = paddle.round(x)
# print(out)

# x = paddle.to_tensor([0.1, 0.2, 0.3, 0.4])
# out = paddle.rsqrt(x)
# print(out)

# data = paddle.randn(shape=[2,3], dtype='float32')
# res = paddle.scale(data, scale=2.0, bias=1.0)

# data = paddle.randn(shape=[2, 3], dtype='float32')
# factor = paddle.to_tensor([2], dtype='float32')
# res = paddle.scale(data, scale=factor, bias=1.0)

# x = np.array([[1, 1], [2, 2], [3, 3]])
# index = np.array([2, 1, 0, 1])

# updates = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
# overwrite = False

# if not overwrite:
#     for i in range(len(index)):
#         x[index[i]] = np.zeros((2))
# for i in range(len(index)):
#     if (overwrite):
#         x[index[i]] = updates[i]
#     else:
#         x[index[i]] += updates[i]

# out = np.array([[3, 3], [6, 6], [1, 1]])
# out.shape 

# x = paddle.to_tensor([[1, 1], [2, 2], [3, 3]], dtype='float32')
# index = paddle.to_tensor([2, 1, 0, 1], dtype='int64')
# updates = paddle.to_tensor([[1, 1], [2, 2], [3, 3], [4, 4]], dtype='float32')

# output1 = paddle.scatter(x, index, updates, overwrite=False)

# output2 = paddle.scatter(x, index, updates, overwrite=True)

# index_data = np.array([[1, 1],
#                        [0, 1],
#                        [1, 3]]).astype(np.int64)
# index = paddle.to_tensor(index_data)
# updates = paddle.rand(shape=[3, 9, 10], dtype='float32')
# shape = [3, 5, 9, 10]

# output = paddle.scatter_nd(index, updates, shape)

# x = paddle.rand(shape=[3, 5, 9, 10], dtype='float32')
# updates = paddle.rand(shape=[3, 9, 10], dtype='float32')
# index_data = np.array([[1, 1],
#                        [0, 1],
#                        [1, 3]]).astype(np.int64)
# index = paddle.to_tensor(index_data)
# output = paddle.scatter_nd_add(x, index, updates)

# data = np.ones([3, 1024], dtype='float32')
# with fluid.dygraph.guard():
#     linear = fluid.dygraph.Linear(1024, 4)
#     t = to_variable(data)
#     linear(t)  
#     custom_weight = np.random.randn(1024, 4).astype("float32")
#     linear.weight.set_value(custom_weight)  
#     out = linear(t)  
# shard_size = (index_num + nshards - 1) // nshards
# v = v - shard_id * shard_size if shard_id * shard_size <= v < (shard_id+1) * shard_size else ignore_value

# label = paddle.to_tensor([[16], [1]], "int64")
# shard_label = paddle.shard_index(input=label,
#                                  index_num=20,
#                                  nshards=2,
#                                  shard_id=0)
# print(shard_label)

# x = paddle.to_tensor([3.0, 0.0, -2.0, 1.7], dtype='float32')
# out = paddle.sign(x=x)
# print(out)  

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.sin(x)
# print(out)

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.sinh(x)
# print(out)

# input = paddle.rand(shape=[4, 5, 6], dtype='float32')

# axes = [0, 1, 2]
# starts = [-3, 0, 2]
# ends = [3, 2, 4]
# sliced_1 = paddle.slice(input, axes=axes, starts=starts, ends=ends)

# minus_3 = paddle.full([1], -3, "int32")
# sliced_2 = paddle.slice(input, axes=axes, starts=[minus_3, 0, 2], ends=ends)

# x = paddle.to_tensor([[[5,8,9,5],
#                        [0,0,1,7],
#                        [6,9,2,4]],
#                       [[5,2,4,2],
#                        [4,7,7,9],
#                        [1,7,0,6]]],
#                      dtype='float32')
# out1 = paddle.sort(x=x, axis=-1)
# out2 = paddle.sort(x=x, axis=0)
# out3 = paddle.sort(x=x, axis=1)
# print(out1)

# print(out2)

# print(out3)

# x = paddle.rand([3, 9, 5])

# out0, out1, out2 = paddle.split(x, num_or_sections=3, axis=1)
# print(out0.shape)  
# print(out1.shape)  
# print(out2.shape)  

# out0, out1, out2 = paddle.split(x, num_or_sections=[2, 3, 4], axis=1)
# print(out0.shape)  
# print(out1.shape)  
# print(out2.shape)  

# out0, out1, out2 = paddle.split(x, num_or_sections=[2, 3, -1], axis=1)
# print(out0.shape)  
# print(out1.shape)  
# print(out2.shape)  

# out0, out1, out2 = paddle.split(x, num_or_sections=3, axis=-2)
# print(out0.shape)  
# print(out1.shape)  
# print(out2.shape)  

# x = paddle.to_tensor([0.1, 0.2, 0.3, 0.4])
# out = paddle.sqrt(x)
# print(out)

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.square(x)
# print(out)

# x = paddle.rand([5, 1, 10])
# output = paddle.squeeze(x, axis=1)

# print(x.shape)  
# print(output.shape)  

# x[0, 0, 0] = 10.
# print(output[0, 0]) 

# x1 = paddle.to_tensor([[1.0, 2.0]])
# x2 = paddle.to_tensor([[3.0, 4.0]])
# x3 = paddle.to_tensor([[5.0, 6.0]])
# out = paddle.stack([x1, x2, x3], axis=0)
# print(out.shape)  
# print(out)

# x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
# out = paddle.stanh(x, scale_a=0.67, scale_b=1.72) 

# x = paddle.to_tensor([[1.0, 2.0, 3.0], [1.0, 4.0, 5.0]])
# out1 = paddle.std(x)

# out2 = paddle.std(x, axis=1)

# x = paddle.zeros(shape=[3,4,5,6], dtype="float32")

# axes = [1, 2, 3]
# starts = [-3, 0, 2]
# ends = [3, 2, 4]
# strides_1 = [1, 1, 1]
# strides_2 = [1, 1, 2]
# sliced_1 = paddle.strided_slice(x, axes=axes, starts=starts, ends=ends, strides=strides_1)

# minus_3 = paddle.full(shape=[1], fill_value=-3, dtype='int32')
# sliced_2 = paddle.strided_slice(x, axes=axes, starts=[minus_3, 0, 2], ends=ends, strides=strides_2)

# x = paddle.to_tensor([[1, 2], [7, 8]])
# y = paddle.to_tensor([[5, 6], [3, 4]])
# res = paddle.subtract(x, y)
# print(res)

# x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
# y = paddle.to_tensor([1, 0, 4])
# res = paddle.subtract(x, y)
# print(res)

# x = paddle.to_tensor([2, np.nan, 5], dtype='float32')
# y = paddle.to_tensor([1, 4, np.nan], dtype='float32')
# res = paddle.subtract(x, y)
# print(res)

# x = paddle.to_tensor([5, np.inf, -np.inf], dtype='float64')
# y = paddle.to_tensor([1, 4, 5], dtype='float64')
# res = paddle.subtract(x, y)
# print(res)

# x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
#                       [0.1, 0.2, 0.6, 0.7]])
# out1 = paddle.sum(x)  
# out2 = paddle.sum(x, axis=0)  
# out3 = paddle.sum(x, axis=-1)  
# out4 = paddle.sum(x, axis=1, keepdim=True)  

# y = paddle.to_tensor([[[1, 2], [3, 4]],
#                       [[5, 6], [7, 8]]])
# out5 = paddle.sum(y, axis=[1, 2]) 
# out6 = paddle.sum(y, axis=[0, 1]) 

# x = paddle.to_tensor([[True, True, True, True],
#                       [False, False, False, False]])
# out7 = paddle.sum(x)  
# out8 = paddle.sum(x, axis=0)  
# out9 = paddle.sum(x, axis=1)  

# x = paddle.ones(shape=[2, 3], dtype='int32')
# x_transposed = paddle.t(x)
# print(x_transposed.shape)

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.tanh(x)
# print(out)

# data_type = 'float64'

# x = paddle.arange(4, dtype=data_type).reshape([2, 2])
# y = paddle.arange(4, dtype=data_type).reshape([2, 2])
# z = paddle.tensordot(x, y, axes=0)

# x = paddle.arange(10, dtype=data_type)
# y = paddle.arange(10, dtype=data_type)
# z1 = paddle.tensordot(x, y, axes=1)
# z2 = paddle.dot(x, y)

# x = paddle.arange(6, dtype=data_type).reshape([2, 3])
# y = paddle.arange(12, dtype=data_type).reshape([3, 4])
# z1 = paddle.tensordot(x, y, axes=1)
# z2 = paddle.matmul(x, y)

# x = paddle.arange(24, dtype=data_type).reshape([2, 3, 4])
# y = paddle.arange(36, dtype=data_type).reshape([3, 3, 4])
# z = paddle.tensordot(x, y, axes=[1, 2])

# x = paddle.arange(60, dtype=data_type).reshape([3, 4, 5])
# y = paddle.arange(24, dtype=data_type).reshape([4, 3, 2])
# z = paddle.tensordot(x, y, axes=([1, 0], [0, 1]))

# x = paddle.arange(720, dtype=data_type).reshape([2, 3, 4, 5, 6])
# y = paddle.arange(720, dtype=data_type).reshape([3, 2, 4, 5, 6])
# z = paddle.tensordot(x, y, axes=[[0, 1, 3, 4], [1, 0]])

# data = paddle.to_tensor([1, 2, 3], dtype='int32')
# out = paddle.tile(data, repeat_times=[2, 1])
# np_out = out.numpy()

# out = paddle.tile(data, repeat_times=[2, 2])
# np_out = out.numpy()

# repeat_times = paddle.to_tensor([2, 1], dtype='int32')
# out = paddle.tile(data, repeat_times=repeat_times)
# np_out = out.numpy()

# t = paddle.to_tensor([0,1,2,3,4])
# expectlist = t.tolist()
# print(expectlist)   

# expectlist = paddle.tolist(t)
# print(expectlist)   

# tensor_1 = paddle.to_tensor([1, 4, 5, 7])
# value_1, indices_1 = paddle.topk(tensor_1, k=1)
# print(value_1)

# print(indices_1)

# tensor_2 = paddle.to_tensor([[1, 4, 5, 7], [2, 6, 2, 5]])
# value_2, indices_2 = paddle.topk(tensor_2, k=1)
# print(value_2)

# print(indices_2)

# value_3, indices_3 = paddle.topk(tensor_2, k=1, axis=-1)
# print(value_3)

# print(indices_3)

# value_4, indices_4 = paddle.topk(tensor_2, k=1, axis=0)
# print(value_4)

# print(indices_4)

# case1 = paddle.randn([2, 3])
# case2 = paddle.randn([3, 10, 10])
# case3 = paddle.randn([3, 10, 5, 10])
# data1 = paddle.trace(case1) 
# data2 = paddle.trace(case2, offset=1, axis1=1, axis2=2) 
# data3 = paddle.trace(case3, offset=-3, axis1=1, axis2=-1) 

# x = paddle.randn([2, 3, 4])
# x_transposed = paddle.transpose(x, perm=[1, 0, 2])
# print(x_transposed.shape)

# input = paddle.rand([2,2],'float32')
# print(input)

# output = paddle.trunc(input)
# print(output)

# np_input = np.random.rand(3, 4, 5).astype('float32')
# input = paddle.to_tensor(np_input)
# [x0, x1, x2] = paddle.unbind(input, axis=0)

# [x0, x1, x2, x3] = paddle.unbind(input, axis=1)

# x = paddle.ones(shape=[3, 4])
# x.uniform_()
# print(x)

# x = paddle.to_tensor([2, 3, 3, 1, 5, 3])
# unique = paddle.unique(x)
# np_unique = unique.numpy() 
# _, indices, inverse, counts = paddle.unique(x, return_index=True, return_inverse=True, return_counts=True)
# np_indices = indices.numpy() 
# np_inverse = inverse.numpy() 
# np_counts = counts.numpy() 

# x = paddle.to_tensor([[2, 1, 3], [3, 0, 1], [2, 1, 3]])
# unique = paddle.unique(x)
# np_unique = unique.numpy() 

# unique = paddle.unique(x, axis=0)
# np_unique = unique.numpy()

# x = paddle.to_tensor([1, 1, 2, 2, 3, 1, 1, 2])
# output = paddle.unique_consecutive(x) 
# np_output = output.numpy() 
# _, inverse, counts = paddle.unique_consecutive(x, return_inverse=True, return_counts=True)
# np_inverse = inverse.numpy() 
# np_counts = inverse.numpy() 

# x = paddle.to_tensor([[2, 1, 3], [3, 0, 1], [2, 1, 3], [2, 1, 3]])
# output = paddle.unique_consecutive(x, axis=0) 
# np_output = output.numpy() 

# x = paddle.to_tensor([[2, 1, 3], [3, 0, 1], [2, 1, 3], [2, 1, 3]])
# output = paddle.unique_consecutive(x, axis=0) 
# np_output = output.numpy()

# x = paddle.rand([5, 10])
# print(x.shape)  

# out1 = paddle.unsqueeze(x, axis=0)
# print(out1.shape)  

# out2 = paddle.unsqueeze(x, axis=[0, 2])
# print(out2.shape)  

# axis = paddle.to_tensor([0, 1, 2])
# out3 = paddle.unsqueeze(x, axis=axis)
# print(out3.shape)  

# x[0, 0] = 10.
# print(out1[0, 0, 0]) 
# print(out2[0, 0, 0, 0]) 
# print(out3[0, 0, 0, 0, 0]) 

# x = paddle.ones(name='x', shape=[2, 3, 5], dtype='float32')  
# y = paddle.unstack(x, axis=1)  

# x = paddle.to_tensor([[1.0, 2.0, 3.0], [1.0, 4.0, 5.0]])
# out1 = paddle.var(x)

# out2 = paddle.var(x, axis=1)

# x = paddle.to_tensor([0.9383, 0.1983, 3.2, 1.2])
# y = paddle.to_tensor([1.0, 1.0, 1.0, 1.0])
# out = paddle.where(x>1, x, y)

# print(out)

# tensor = paddle.to_tensor([0, 1, 2, 3, 4])

# tensor.zero_()
# print(tensor.tolist())   

# data_type = 'float64'

# x = paddle.arange(4, dtype=data_type).reshape([2, 2])
# y = paddle.arange(4, dtype=data_type).reshape([2, 2])
# z = paddle.tensordot(x, y, axes=0)

# x = paddle.arange(10, dtype=data_type)
# y = paddle.arange(10, dtype=data_type)
# z1 = paddle.tensordot(x, y, axes=1)
# z2 = paddle.dot(x, y)

# x = paddle.arange(6, dtype=data_type).reshape([2, 3])
# y = paddle.arange(12, dtype=data_type).reshape([3, 4])
# z1 = paddle.tensordot(x, y, axes=1)
# z2 = paddle.matmul(x, y)

# x = paddle.arange(24, dtype=data_type).reshape([2, 3, 4])
# y = paddle.arange(36, dtype=data_type).reshape([3, 3, 4])
# z = paddle.tensordot(x, y, axes=[1, 2])

# x = paddle.arange(60, dtype=data_type).reshape([3, 4, 5])
# y = paddle.arange(24, dtype=data_type).reshape([4, 3, 2])
# z = paddle.tensordot(x, y, axes=([1, 0], [0, 1]))

# x = paddle.arange(720, dtype=data_type).reshape([2, 3, 4, 5, 6])
# y = paddle.arange(720, dtype=data_type).reshape([3, 2, 4, 5, 6])
# z = paddle.tensordot(x, y, axes=[[0, 1, 3, 4], [1, 0]])

# data = paddle.to_tensor([1, 2, 3], dtype='int32')
# out = paddle.tile(data, repeat_times=[2, 1])
# np_out = out.numpy()

# out = paddle.tile(data, repeat_times=[2, 2])
# np_out = out.numpy()

# repeat_times = paddle.to_tensor([2, 1], dtype='int32')
# out = paddle.tile(data, repeat_times=repeat_times)
# np_out = out.numpy()

# type(paddle.to_tensor(1))

# paddle.to_tensor(1)

# x = paddle.to_tensor(1, stop_gradient=False)
# print(x)

# paddle.to_tensor(x)  

# paddle.to_tensor([[0.1, 0.2], [0.3, 0.4]], place=paddle.CPUPlace(), stop_gradient=False)

# type(paddle.to_tensor([[1+1j, 2], [3+2j, 4]], dtype='complex64'))

# paddle.to_tensor([[1+1j, 2], [3+2j, 4]], dtype='complex64')

# t = paddle.to_tensor([0,1,2,3,4])
# expectlist = t.tolist()
# print(expectlist)   

# expectlist = paddle.tolist(t)
# print(expectlist)   

# tensor_1 = paddle.to_tensor([1, 4, 5, 7])
# value_1, indices_1 = paddle.topk(tensor_1, k=1)
# print(value_1)

# print(indices_1)

# tensor_2 = paddle.to_tensor([[1, 4, 5, 7], [2, 6, 2, 5]])
# value_2, indices_2 = paddle.topk(tensor_2, k=1)
# print(value_2)

# print(indices_2)

# value_3, indices_3 = paddle.topk(tensor_2, k=1, axis=-1)
# print(value_3)

# print(indices_3)

# value_4, indices_4 = paddle.topk(tensor_2, k=1, axis=0)
# print(value_4)

# print(indices_4)

# case1 = paddle.randn([2, 3])
# case2 = paddle.randn([3, 10, 10])
# case3 = paddle.randn([3, 10, 5, 10])
# data1 = paddle.trace(case1) 
# data2 = paddle.trace(case2, offset=1, axis1=1, axis2=2) 
# data3 = paddle.trace(case3, offset=-3, axis1=1, axis2=-1) 

# x = paddle.randn([2, 3, 4])
# x_transposed = paddle.transpose(x, perm=[1, 0, 2])
# print(x_transposed.shape)

# data = np.arange(1, 13, dtype="int64").reshape(3,-1)

# x = paddle.to_tensor(data)

# tril1 = paddle.tensor.tril(x)

# tril2 = paddle.tensor.tril(x, diagonal=2)

# tril3 = paddle.tensor.tril(x, diagonal=-1)

# data = np.arange(1, 13, dtype="int64").reshape(3,-1)

# x = paddle.to_tensor(data)
# triu1 = paddle.tensor.triu(x)

# triu2 = paddle.tensor.triu(x, diagonal=2)

# triu3 = paddle.tensor.triu(x, diagonal=-1)

# input = paddle.rand([2,2],'float32')
# print(input)

# output = paddle.trunc(input)
# print(output)

# np_input = np.random.rand(3, 4, 5).astype('float32')
# input = paddle.to_tensor(np_input)
# [x0, x1, x2] = paddle.unbind(input, axis=0)

# [x0, x1, x2, x3] = paddle.unbind(input, axis=1)

# out1 = paddle.uniform(shape=[3, 4])

# dim1 = paddle.to_tensor([2], 'int64')
# dim2 = paddle.to_tensor([3], 'int32')
# out2 = paddle.uniform(shape=[dim1, dim2])

# shape_tensor = paddle.to_tensor([2, 3])
# out3 = paddle.uniform(shape_tensor)

# x = paddle.to_tensor([2, 3, 3, 1, 5, 3])
# unique = paddle.unique(x)
# np_unique = unique.numpy() 
# _, indices, inverse, counts = paddle.unique(x, return_index=True, return_inverse=True, return_counts=True)
# np_indices = indices.numpy() 
# np_inverse = inverse.numpy() 
# np_counts = counts.numpy() 

# x = paddle.to_tensor([[2, 1, 3], [3, 0, 1], [2, 1, 3]])
# unique = paddle.unique(x)
# np_unique = unique.numpy() 

# unique = paddle.unique(x, axis=0)
# np_unique = unique.numpy()

# x = paddle.to_tensor([1, 1, 2, 2, 3, 1, 1, 2])
# output = paddle.unique_consecutive(x) 
# np_output = output.numpy() 
# _, inverse, counts = paddle.unique_consecutive(x, return_inverse=True, return_counts=True)
# np_inverse = inverse.numpy() 
# np_counts = inverse.numpy() 

# x = paddle.to_tensor([[2, 1, 3], [3, 0, 1], [2, 1, 3], [2, 1, 3]])
# output = paddle.unique_consecutive(x, axis=0) 
# np_output = output.numpy() 

# x = paddle.to_tensor([[2, 1, 3], [3, 0, 1], [2, 1, 3], [2, 1, 3]])
# output = paddle.unique_consecutive(x, axis=0) 
# np_output = output.numpy()

# x = paddle.rand([5, 10])
# print(x.shape)  

# out1 = paddle.unsqueeze(x, axis=0)
# print(out1.shape)  

# out2 = paddle.unsqueeze(x, axis=[0, 2])
# print(out2.shape)  

# axis = paddle.to_tensor([0, 1, 2])
# out3 = paddle.unsqueeze(x, axis=axis)
# print(out3.shape)  

# x[0, 0] = 10.
# print(out1[0, 0, 0]) 
# print(out2[0, 0, 0, 0]) 
# print(out3[0, 0, 0, 0, 0]) 

# x = paddle.ones(name='x', shape=[2, 3, 5], dtype='float32')  
# y = paddle.unstack(x, axis=1)  

# x = paddle.to_tensor([[1.0, 2.0, 3.0], [1.0, 4.0, 5.0]])
# out1 = paddle.var(x)

# out2 = paddle.var(x, axis=1)

# x = paddle.to_tensor([0.9383, 0.1983, 3.2, 1.2])
# y = paddle.to_tensor([1.0, 1.0, 1.0, 1.0])
# out = paddle.where(x>1, x, y)

# print(out)

# data = paddle.zeros(shape=[3, 2], dtype='float32')

# data = paddle.zeros(shape=[2, 2])

# shape = paddle.full(shape=[2], dtype='int32', fill_value=2)
# data3 = paddle.zeros(shape=shape, dtype='int32')

# x = paddle.to_tensor([1, 2, 3])
# out1 = paddle.zeros_like(x) 
# out2 = paddle.zeros_like(x, dtype='int32') 

# conv2d = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
# data = paddle.rand([10, 3, 32, 32])

# with paddle.amp.auto_cast():
#     conv = conv2d(data)
#     print(conv.dtype) 

# with paddle.amp.auto_cast(enable=False):
#     conv = conv2d(data)
#     print(conv.dtype) 

# with paddle.amp.auto_cast(custom_black_list={'conv2d'}):
#     conv = conv2d(data)
#     print(conv.dtype) 

# a = paddle.rand([2,3])
# b = paddle.rand([2,3])
# with paddle.amp.auto_cast(custom_white_list={'elementwise_add'}):
#     c = a + b
#     print(c.dtype) 

# with paddle.amp.auto_cast(custom_white_list={'elementwise_add'}, level='O2'):
#     d = a + b
#     print(d.dtype) 

# model = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
# optimzier = paddle.optimizer.SGD(parameters=model.parameters())

# model, optimizer = paddle.amp.decorate(models=model, optimizers=optimzier, level='O2')

# data = paddle.rand([10, 3, 32, 32])

# with paddle.amp.auto_cast(enable=True, custom_white_list=None, custom_black_list=None, level='O2'):
#     output = model(data)
#     print(output.dtype) 

# model2 = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
# optimizer2 = paddle.optimizer.Adam(parameters=model2.parameters())

# models, optimizers = paddle.amp.decorate(models=[model, model2], optimizers=[optimzier, optimizer2], level='O2')

# data = paddle.rand([10, 3, 32, 32])

# with paddle.amp.auto_cast(enable=True, custom_white_list=None, custom_black_list=None, level='O2'):
#     output = models[0](data)
#     output2 = models[1](data)
#     print(output.dtype) 
#     print(output2.dtype) 

# model = paddle.nn.Conv2D(3, 2, 3, bias_attr=True)
# optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
# scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
# data = paddle.rand([10, 3, 32, 32])

# with paddle.amp.auto_cast():
#     conv = model(data)
#     loss = paddle.mean(conv)

# scaled = scaler.scale(loss)  
# scaled.backward()            
# scaler.minimize(optimizer, scaled)  
# optimizer.clear_grad()

# model = paddle.nn.Conv2D(3, 2, 3, bias_attr=True)
# optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
# scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
# data = paddle.rand([10, 3, 32, 32])

# with paddle.amp.auto_cast():
#     conv = model(data)
#     loss = paddle.mean(conv)

# scaled = scaler.scale(loss)  
# scaled.backward()            
# scaler.minimize(optimizer, scaled)  
# optimizer.clear_grad()

# model = paddle.nn.Conv2D(3, 2, 3, bias_attr=True)
# optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
# scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
# data = paddle.rand([10, 3, 32, 32])

# with paddle.amp.auto_cast():
#     conv = model(data)
#     loss = paddle.mean(conv)

# scaled = scaler.scale(loss)  
# scaled.backward()            
# scaler.minimize(optimizer, scaled)  
# optimizer.clear_grad()

# model = paddle.nn.Conv2D(3, 2, 3, bias_attr=True)
# optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
# scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
# data = paddle.rand([10, 3, 32, 32])
# with paddle.amp.auto_cast():
#     conv = model(data)
#     loss = paddle.mean(conv)
# scaled = scaler.scale(loss)  
# scaled.backward()            
# scaler.step(optimizer)       
# scaler.update()              
# optimizer.clear_grad()

# model = paddle.nn.Conv2D(3, 2, 3, bias_attr=True)
# optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
# scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
# data = paddle.rand([10, 3, 32, 32])
# with paddle.amp.auto_cast():
#     conv = model(data)
#     loss = paddle.mean(conv)
# scaled = scaler.scale(loss)     
# scaled.backward()               
# scaler.step(optimizer)          
# scaler.update()                 
# optimizer.clear_grad()

# model = paddle.nn.Conv2D(3, 2, 3, bias_attr=True)
# optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
# scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
# data = paddle.rand([10, 3, 32, 32])
# with paddle.amp.auto_cast():
#     conv = model(data)
#     loss = paddle.mean(conv)
# scaled = scaler.scale(loss)  
# scaled.backward()            
# scaler.unscale_(optimizer)    
# scaler.step(optimizer)
# scaler.update()
# optimizer.clear_grad()

# scaler = paddle.amp.GradScaler(enable=True,
#                                init_loss_scaling=1024,
#                                incr_ratio=2.0,
#                                decr_ratio=0.5,
#                                incr_every_n_steps=1000,
#                                decr_every_n_nan_or_inf=2,
#                                use_dynamic_loss_scaling=True)
# enable = scaler.is_enable()
# print(enable) 

# scaler = paddle.amp.GradScaler(enable=True,
#                                init_loss_scaling=1024,
#                                incr_ratio=2.0,
#                                decr_ratio=0.5,
#                                incr_every_n_steps=1000,
#                                decr_every_n_nan_or_inf=2,
#                                use_dynamic_loss_scaling=True)
# use_dynamic_loss_scaling = scaler.is_use_dynamic_loss_scaling()
# print(use_dynamic_loss_scaling) 

# scaler = paddle.amp.GradScaler(enable=True,
#                                init_loss_scaling=1024,
#                                incr_ratio=2.0,
#                                decr_ratio=0.5,
#                                incr_every_n_steps=1000,
#                                decr_every_n_nan_or_inf=2,
#                                use_dynamic_loss_scaling=True)
# init_loss_scaling = scaler.get_init_loss_scaling()
# print(init_loss_scaling) 

# scaler = paddle.amp.GradScaler(enable=True,
#                                init_loss_scaling=1024,
#                                incr_ratio=2.0,
#                                decr_ratio=0.5,
#                                incr_every_n_steps=1000,
#                                decr_every_n_nan_or_inf=2,
#                                use_dynamic_loss_scaling=True)
# print(scaler.get_init_loss_scaling()) 
# new_init_loss_scaling = 1000
# scaler.set_init_loss_scaling(new_init_loss_scaling)
# print(scaler.get_init_loss_scaling()) 

# scaler = paddle.amp.GradScaler(enable=True,
#                                init_loss_scaling=1024,
#                                incr_ratio=2.0,
#                                decr_ratio=0.5,
#                                incr_every_n_steps=1000,
#                                decr_every_n_nan_or_inf=2,
#                                use_dynamic_loss_scaling=True)
# incr_ratio = scaler.get_incr_ratio()
# print(incr_ratio) 

# scaler = paddle.amp.GradScaler(enable=True,
#                                init_loss_scaling=1024,
#                                incr_ratio=2.0,
#                                decr_ratio=0.5,
#                                incr_every_n_steps=1000,
#                                decr_every_n_nan_or_inf=2,
#                                use_dynamic_loss_scaling=True)
# print(scaler.get_incr_ratio()) 
# new_incr_ratio = 3.0
# scaler.set_incr_ratio(new_incr_ratio)
# print(scaler.get_incr_ratio()) 

# scaler = paddle.amp.GradScaler(enable=True,
#                                init_loss_scaling=1024,
#                                incr_ratio=2.0,
#                                decr_ratio=0.5,
#                                incr_every_n_steps=1000,
#                                decr_every_n_nan_or_inf=2,
#                                use_dynamic_loss_scaling=True)
# decr_ratio = scaler.get_decr_ratio()
# print(decr_ratio) 

# scaler = paddle.amp.GradScaler(enable=True,
#                                init_loss_scaling=1024,
#                                incr_ratio=2.0,
#                                decr_ratio=0.5,
#                                incr_every_n_steps=1000,
#                                decr_every_n_nan_or_inf=2,
#                                use_dynamic_loss_scaling=True)
# print(scaler.get_decr_ratio()) 
# new_decr_ratio = 0.1
# scaler.set_decr_ratio(new_decr_ratio)
# print(scaler.get_decr_ratio()) 

# scaler = paddle.amp.GradScaler(enable=True,
#                                init_loss_scaling=1024,
#                                incr_ratio=2.0,
#                                decr_ratio=0.5,
#                                incr_every_n_steps=1000,
#                                decr_every_n_nan_or_inf=2,
#                                use_dynamic_loss_scaling=True)
# incr_every_n_steps = scaler.get_incr_every_n_steps()
# print(incr_every_n_steps) 

# scaler = paddle.amp.GradScaler(enable=True,
#                                init_loss_scaling=1024,
#                                incr_ratio=2.0,
#                                decr_ratio=0.5,
#                                incr_every_n_steps=1000,
#                                decr_every_n_nan_or_inf=2,
#                                use_dynamic_loss_scaling=True)
# print(scaler.get_incr_every_n_steps()) 
# new_incr_every_n_steps = 2000
# scaler.set_incr_every_n_steps(new_incr_every_n_steps)
# print(scaler.get_incr_every_n_steps()) 

# scaler = paddle.amp.GradScaler(enable=True,
#                                init_loss_scaling=1024,
#                                incr_ratio=2.0,
#                                decr_ratio=0.5,
#                                incr_every_n_steps=1000,
#                                decr_every_n_nan_or_inf=2,
#                                use_dynamic_loss_scaling=True)
# decr_every_n_nan_or_inf = scaler.get_decr_every_n_nan_or_inf()
# print(decr_every_n_nan_or_inf) 

# scaler = paddle.amp.GradScaler(enable=True,
#                                init_loss_scaling=1024,
#                                incr_ratio=2.0,
#                                decr_ratio=0.5,
#                                incr_every_n_steps=1000,
#                                decr_every_n_nan_or_inf=2,
#                                use_dynamic_loss_scaling=True)
# print(scaler.get_decr_every_n_nan_or_inf()) 
# new_decr_every_n_nan_or_inf = 3
# scaler.set_decr_every_n_nan_or_inf(new_decr_every_n_nan_or_inf)
# print(scaler.get_decr_every_n_nan_or_inf()) 

# scaler = paddle.amp.GradScaler(enable=True,
#                                init_loss_scaling=1024,
#                                incr_ratio=2.0,
#                                decr_ratio=0.5,
#                                incr_every_n_steps=1000,
#                                decr_every_n_nan_or_inf=2,
#                                use_dynamic_loss_scaling=True)
# scaler_state = scaler.state_dict()

# scaler = paddle.amp.GradScaler(enable=True,
#                                init_loss_scaling=1024,
#                                incr_ratio=2.0,
#                                decr_ratio=0.5,
#                                incr_every_n_steps=1000,
#                                decr_every_n_nan_or_inf=2,
#                                use_dynamic_loss_scaling=True)
# scaler_state = scaler.state_dict()
# scaler.load_state_dict(scaler_state)

# x = paddle.to_tensor([[1, 2], [3, 4]], dtype='float32', stop_gradient=False)
# y = paddle.to_tensor([[3, 2], [3, 4]], dtype='float32')

# grad_tensor1 = paddle.to_tensor([[1,2], [2, 3]], dtype='float32')
# grad_tensor2 = paddle.to_tensor([[1,1], [1, 1]], dtype='float32')

# z1 = paddle.matmul(x, y)
# z2 = paddle.matmul(x, y)

# paddle.autograd.backward([z1, z2], [grad_tensor1, grad_tensor2], True)
# print(x.grad)

# x.clear_grad()

# paddle.autograd.backward([z1, z2], [grad_tensor1, None], True)
# print(x.grad)

# x.clear_grad()

# paddle.autograd.backward([z1, z2])
# print(x.grad)

# class cus_tanh(PyLayer):
#     @staticmethod
#     def forward(ctx, x, func1, func2=paddle.square):
        
#         ctx.func = func2
#         y = func1(x)
        
#         ctx.save_for_backward(y)
#         return y

#     @staticmethod
    
#     def backward(ctx, dy):
        
#         y, = ctx.saved_tensor()
#         grad = dy * (1 - ctx.func(y))
        
#         return grad

# data = paddle.randn([2, 3], dtype="float64")
# data.stop_gradient = False
# z = cus_tanh.apply(data, func1=paddle.tanh)
# z.mean().backward()

# print(data.grad)

# class cus_tanh(PyLayer):
#     @staticmethod
#     def forward(ctx, x):
#         y = paddle.tanh(x)
        
#         ctx.save_for_backward(y)
#         return y

#     @staticmethod
#     def backward(ctx, dy):
        
#         y, = ctx.saved_tensor()
#         grad = dy * (1 - paddle.square(y))
#         return grad

# class cus_tanh(PyLayer):
#     @staticmethod
#     def forward(ctx, x):
#         y = paddle.tanh(x)
        
#         ctx.save_for_backward(y)
#         return y

#     @staticmethod
#     def backward(ctx, dy):
        
#         y, = ctx.saved_tensor()
#         grad = dy * (1 - paddle.square(y))
#         return grad

# class cus_tanh(PyLayer):
#     @staticmethod
#     def forward(ctx, x, func1, func2=paddle.square):
#         ctx.func = func2
#         y = func1(x)
        
#         ctx.save_for_backward(y)
#         return y

#     @staticmethod
#     def backward(ctx, dy):
        
#         y, = ctx.saved_tensor()
#         grad = dy * (1 - ctx.func(y))
#         return grad

# data = paddle.randn([2, 3], dtype="float64")
# data.stop_gradient = False

# z = cus_tanh.apply(data, func1=paddle.tanh)

# class cus_tanh(PyLayer):
#     @staticmethod
#     def forward(ctx, x):
        
#         y = paddle.tanh(x)
#         ctx.save_for_backward(y)
#         return y

#     @staticmethod
#     def backward(ctx, dy):
        
#         y, = ctx.saved_tensor()
#         grad = dy * (1 - paddle.square(y))
#         return grad

# class cus_tanh(PyLayer):
#     @staticmethod
#     def forward(ctx, x):
        
#         y = paddle.tanh(x)
        
#         ctx.save_for_backward(y)
#         return y

#     @staticmethod
#     def backward(ctx, dy):
        
#         y, = ctx.saved_tensor()
#         grad = dy * (1 - paddle.square(y))
#         return grad

# class cus_tanh(PyLayer):
#     @staticmethod
#     def forward(ctx, x):
        
#         y = paddle.tanh(x)
        
#         ctx.save_for_backward(y)
#         return y

#     @staticmethod
#     def backward(ctx, dy):
        
#         y, = ctx.saved_tensor()
#         grad = dy * (1 - paddle.square(y))
#         return grad

# class ModelCheckpoint(paddle.callbacks.Callback):
#     def __init__(self, save_freq=1, save_dir=None):
#         self.save_freq = save_freq
#         self.save_dir = save_dir

#     def on_epoch_end(self, epoch, logs=None):
#         if self.model is not None and epoch % self.save_freq == 0:
#             path = '{}/{}'.format(self.save_dir, epoch)
#             print('save checkpoint at {}'.format(path))
#             self.model.save(path)

# device = paddle.set_device('cpu')
# sample_num = 200
# save_dir = './best_model_checkpoint'
# transform = T.Compose(
#     [T.Transpose(), T.Normalize([127.5], [127.5])])
# train_dataset = MNIST(mode='train', transform=transform)
# val_dataset = MNIST(mode='test', transform=transform)
# net = LeNet()
# optim = paddle.optimizer.Adam(
#     learning_rate=0.001, parameters=net.parameters())

# inputs = [InputSpec([None, 1, 28, 28], 'float32', 'x')]
# labels = [InputSpec([None, 1], 'int64', 'label')]

# model = Model(net, inputs=inputs, labels=labels)
# model.prepare(
#     optim,
#     loss=CrossEntropyLoss(reduction="sum"),
#     metrics=[Accuracy()])
# callbacks = paddle.callbacks.EarlyStopping(
#     'loss',
#     mode='min',
#     patience=1,
#     verbose=1,
#     min_delta=0,
#     baseline=None,
#     save_best_model=True)
# model.fit(train_dataset,
#           val_dataset,
#           batch_size=64,
#           log_freq=200,
#           save_freq=10,
#           save_dir=save_dir,
#           epochs=20,
#           callbacks=[callbacks])

# inputs = [InputSpec([-1, 1, 28, 28], 'float32', 'image')]
# labels = [InputSpec([None, 1], 'int64', 'label')]

# transform = T.Compose([
#     T.Transpose(),
#     T.Normalize([127.5], [127.5])
# ])
# train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)

# lenet = paddle.vision.models.LeNet()
# model = paddle.Model(lenet,
#     inputs, labels)

# base_lr = 1e-3
# boundaries = [5, 8]
# wamup_steps = 4

# def make_optimizer(parameters=None):
#     momentum = 0.9
#     weight_decay = 5e-4
#     values = [base_lr * (0.1**i) for i in range(len(boundaries) + 1)]
#     learning_rate = paddle.optimizer.lr.PiecewiseDecay(
#         boundaries=boundaries, values=values)
#     learning_rate = paddle.optimizer.lr.LinearWarmup(
#         learning_rate=learning_rate,
#         warmup_steps=wamup_steps,
#         start_lr=base_lr / 5.,
#         end_lr=base_lr,
#         verbose=True)
#     optimizer = paddle.optimizer.Momentum(
#         learning_rate=learning_rate,
#         weight_decay=weight_decay,
#         momentum=momentum,
#         parameters=parameters)
#     return optimizer

# optim = make_optimizer(parameters=lenet.parameters())
# model.prepare(optimizer=optim,
#             loss=paddle.nn.CrossEntropyLoss(),
#             metrics=paddle.metric.Accuracy())

# model.fit(train_dataset, batch_size=64)

# callback = paddle.callbacks.LRScheduler(by_step=False, by_epoch=True)
# model.fit(train_dataset, batch_size=64, callbacks=callback)

# inputs = [InputSpec([-1, 1, 28, 28], 'float32', 'image')]
# labels = [InputSpec([None, 1], 'int64', 'label')]

# transform = T.Compose([
#     T.Transpose(),
#     T.Normalize([127.5], [127.5])
# ])
# train_dataset = MNIST(mode='train', transform=transform)

# lenet = paddle.vision.models.LeNet()
# model = paddle.Model(lenet,
#     inputs, labels)

# optim = paddle.optimizer.Adam(0.001, parameters=lenet.parameters())
# model.prepare(optimizer=optim,
#             loss=paddle.nn.CrossEntropyLoss(),
#             metrics=paddle.metric.Accuracy())

# callback = paddle.callbacks.ModelCheckpoint(save_dir='./temp')
# model.fit(train_dataset, batch_size=64, callbacks=callback)

# inputs = [InputSpec([-1, 1, 28, 28], 'float32', 'image')]
# labels = [InputSpec([None, 1], 'int64', 'label')]

# transform = T.Compose([
#     T.Transpose(),
#     T.Normalize([127.5], [127.5])
# ])
# train_dataset = MNIST(mode='train', transform=transform)

# lenet = paddle.vision.models.LeNet()
# model = paddle.Model(lenet,
#     inputs, labels)

# optim = paddle.optimizer.Adam(0.001, parameters=lenet.parameters())
# model.prepare(optimizer=optim,
#             loss=paddle.nn.CrossEntropyLoss(),
#             metrics=paddle.metric.Accuracy())

# callback = paddle.callbacks.ProgBarLogger(log_freq=10)
# model.fit(train_dataset, batch_size=64, callbacks=callback)

# sample_num = 200
# transform = T.Compose(
#     [T.Transpose(), T.Normalize([127.5], [127.5])])
# train_dataset = MNIST(mode='train', transform=transform)
# val_dataset = MNIST(mode='test', transform=transform)
# net = LeNet()
# optim = paddle.optimizer.Adam(
#     learning_rate=0.001, parameters=net.parameters())
# inputs = [InputSpec([None, 1, 28, 28], 'float32', 'x')]
# labels = [InputSpec([None, 1], 'int64', 'label')]
# model = Model(net, inputs=inputs, labels=labels)
# model.prepare(
#     optim,
#     loss=CrossEntropyLoss(),
#     metrics=[Accuracy()])
# callbacks = paddle.callbacks.ReduceLROnPlateau(patience=3, verbose=1)
# model.fit(train_dataset,
#             val_dataset,
#             batch_size=64,
#             log_freq=200,
#             save_freq=10,
#             epochs=20,
#             callbacks=[callbacks])

# inputs = [InputSpec([-1, 1, 28, 28], 'float32', 'image')]
# labels = [InputSpec([None, 1], 'int64', 'label')]

# transform = T.Compose([
#     T.Transpose(),
#     T.Normalize([127.5], [127.5])
# ])
# train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
# eval_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)

# net = paddle.vision.models.LeNet()
# model = paddle.Model(net, inputs, labels)

# optim = paddle.optimizer.Adam(0.001, parameters=net.parameters())
# model.prepare(optimizer=optim,
#             loss=paddle.nn.CrossEntropyLoss(),
#             metrics=paddle.metric.Accuracy())

# cudnn_version = paddle.device.get_cudnn_version()

# device = paddle.device.get_device()

# support_gpu = paddle.device.is_compiled_with_cuda()

# support_npu = paddle.device.is_compiled_with_npu()

# support_gpu = paddle.device.is_compiled_with_rocm()

# support_xpu = paddle.device.is_compiled_with_xpu()

# paddle.device.set_device("cpu")
# x1 = paddle.ones(name='x1', shape=[1, 2], dtype='int32')
# x2 = paddle.zeros(name='x2', shape=[1, 2], dtype='int32')
# data = paddle.stack([x1,x2], axis=1)

# place = paddle.device.XPUPlace(0)

# paddle.set_device('gpu:%d'%paddle.distributed.ParallelEnv().dev_id)
# init_parallel_env()
# tensor_list = []
# if paddle.distributed.ParallelEnv().local_rank == 0:
#     np_data1 = np.array([[4, 5, 6], [4, 5, 6]])
#     np_data2 = np.array([[4, 5, 6], [4, 5, 6]])
#     data1 = paddle.to_tensor(np_data1)
#     data2 = paddle.to_tensor(np_data2)
#     paddle.distributed.all_gather(tensor_list, data1)
# else:
#     np_data1 = np.array([[1, 2, 3], [1, 2, 3]])
#     np_data2 = np.array([[1, 2, 3], [1, 2, 3]])
#     data1 = paddle.to_tensor(np_data1)
#     data2 = paddle.to_tensor(np_data2)
#     paddle.distributed.all_gather(tensor_list, data2)

# paddle.set_device('gpu:%d'%paddle.distributed.ParallelEnv().dev_id)
# init_parallel_env()
# if paddle.distributed.ParallelEnv().local_rank == 0:
#     np_data = np.array([[4, 5, 6], [4, 5, 6]])
# else:
#     np_data = np.array([[1, 2, 3], [1, 2, 3]])
# data = paddle.to_tensor(np_data)
# paddle.distributed.all_reduce(data)
# out = data.numpy()

# init_parallel_env()
# out_tensor_list = []
# if paddle.distributed.ParallelEnv().rank == 0:
#     np_data1 = np.array([[1, 2, 3], [4, 5, 6]])
#     np_data2 = np.array([[7, 8, 9], [10, 11, 12]])
# else:
#     np_data1 = np.array([[13, 14, 15], [16, 17, 18]])
#     np_data2 = np.array([[19, 20, 21], [22, 23, 24]])
# data1 = paddle.to_tensor(np_data1)
# data2 = paddle.to_tensor(np_data2)
# paddle.distributed.alltoall([data1, data2], out_tensor_list)

# paddle.set_device('gpu:%d'%paddle.distributed.ParallelEnv().dev_id)
# init_parallel_env()
# paddle.distributed.barrier()

# paddle.set_device('gpu:%d'%paddle.distributed.ParallelEnv().dev_id)
# init_parallel_env()
# if paddle.distributed.ParallelEnv().local_rank == 0:
#     np_data = np.array([[4, 5, 6], [4, 5, 6]])
# else:
#     np_data = np.array([[1, 2, 3], [1, 2, 3]])
# data = paddle.to_tensor(np_data)
# paddle.distributed.broadcast(data, 1)
# out = data.numpy()

# sparse_feature_dim = 1024
# embedding_size = 64

# entry = paddle.distributed.CountFilterEntry(10)

# input = paddle.static.data(name='ins', shape=[1], dtype='int64')

# emb = paddle.static.nn.sparse_embedding(
#     input=input,
#     size=[sparse_feature_dim, embedding_size],
#     is_test=False,
#     entry=entry,
#     param_attr=paddle.ParamAttr(name="SparseFeatFactors",
#                                initializer=paddle.nn.initializer.Uniform()))

# print("The rank is %d" % dist.get_rank())

# print("The world_size is %d" % dist.get_world_size())

# port_set = set()

# def find_free_port():
#     def _free_port():
#         with closing(socket.socket(socket.AF_INET,
#             socket.SOCK_STREAM)) as s:
#             s.bind(('', 0))
#             return s.getsockname()[1]
#     while True:
#         port = _free_port()
#         if port not in port_set:
#             port_set.add(port)
#             return port

# def test_gloo_barrier(id, rank_num, server_endpoint):
#     paddle.distributed.gloo_init_parallel_env(
#         id, rank_num, server_endpoint)
#     paddle.distributed.gloo_barrier()

# def test_gloo_barrier_with_multiprocess(num_of_ranks):
#     jobs = []
#     server_endpoint = "127.0.0.1:%s" % (find_free_port())
#     for id in range(num_of_ranks):
#         p = multiprocessing.Process(
#             target=test_gloo_barrier,
#             args=(id, num_of_ranks, server_endpoint))
#         jobs.append(p)
#         p.start()
#     for proc in jobs:
#         proc.join()

# if __name__ == '__main__':
    
#     test_gloo_barrier_with_multiprocess(2)

# port_set = set()

# def find_free_port():
#     def _free_port():
#         with closing(socket.socket(socket.AF_INET,
#             socket.SOCK_STREAM)) as s:
#             s.bind(('', 0))
#             return s.getsockname()[1]
#     while True:
#         port = _free_port()
#         if port not in port_set:
#             port_set.add(port)
#             return port

# def test_gloo_init(id, rank_num, server_endpoint):
#     paddle.distributed.gloo_init_parallel_env(
#         id, rank_num, server_endpoint)

# def test_gloo_init_with_multiprocess(num_of_ranks):
#     jobs = []
#     server_endpoint = "127.0.0.1:%s" % (find_free_port())
#     for id in range(num_of_ranks):
#         p = multiprocessing.Process(
#             target=test_gloo_init,
#             args=(id, num_of_ranks, server_endpoint))
#         jobs.append(p)
#         p.start()
#     for proc in jobs:
#         proc.join()

# if __name__ == '__main__':
    
#     test_gloo_init_with_multiprocess(2)

# port_set = set()

# def find_free_port():
#     def _free_port():
#         with closing(socket.socket(socket.AF_INET,
#             socket.SOCK_STREAM)) as s:
#             s.bind(('', 0))
#             return s.getsockname()[1]
#     while True:
#         port = _free_port()
#         if port not in port_set:
#             port_set.add(port)
#             return port

# def test_gloo_release(id, rank_num, server_endpoint):
#     paddle.distributed.gloo_init_parallel_env(
#         id, rank_num, server_endpoint)
#     paddle.distributed.gloo_barrier()
#     paddle.distributed.gloo_release()

# def test_gloo_release_with_multiprocess(num_of_ranks):
#     jobs = []
#     server_endpoint = "127.0.0.1:%s" % (find_free_port())
#     for id in range(num_of_ranks):
#         p = multiprocessing.Process(
#             target=test_gloo_release,
#             args=(id, num_of_ranks, server_endpoint))
#         jobs.append(p)
#         p.start()
#     for proc in jobs:
#         proc.join()
    
# test_gloo_release_with_multiprocess(2)
    
# import paddle
# import paddle.nn as nn
# import paddle.optimizer as opt
# import paddle.distributed as dist

# class LinearNet(nn.Layer):
#     def __init__(self):
#         super(LinearNet, self).__init__()
#         self._linear1 = nn.Linear(10, 10)
#         self._linear2 = nn.Linear(10, 1)

#     def forward(self, x):
#         return self._linear2(self._linear1(x))

#     def train():
        
#         dist.init_parallel_env()

        
#         layer = LinearNet()
#         dp_layer = paddle.DataParallel(layer)

#         loss_fn = nn.MSELoss()
#         adam = opt.Adam(
#             learning_rate=0.001, parameters=dp_layer.parameters())

        
#         inputs = paddle.randn([10, 10], 'float32')
#         outputs = dp_layer(inputs)
#         labels = paddle.randn([10, 1], 'float32')
#         loss = loss_fn(outputs, labels)

#         loss.backward()

#         adam.step()
#         adam.clear_grad()

#         dist.spawn(train)

# paddle.enable_static()
# dataset = paddle.distributed.InMemoryDataset()

# paddle.enable_static()

# dataset = paddle.distributed.InMemoryDataset()
# dataset.init(
#     batch_size=1,
#     thread_num=2,
#     input_type=1,
#     pipe_command="cat",
#     use_var=[])
# dataset._init_distributed_settings(
#     parse_ins_id=True,
#     parse_content=True,
#     fea_eval=True,
#     candidate_size=10000)
# dataset.update_settings(batch_size=2)

# paddle.enable_static()

# with open("test_queue_dataset_run_a.txt", "w") as f:
#     data = "2 1 2 2 5 4 2 2 7 2 1 3"
#     f.write(data)
# with open("test_queue_dataset_run_b.txt", "w") as f:
#     data = "2 1 2 2 5 4 2 2 7 2 1 3"
#     f.write(data)

# slots = ["slot1", "slot2", "slot3", "slot4"]
# slots_vars = []
# for slot in slots:
#     var = paddle.static.data(
#         name=slot, shape=[None, 1], dtype="int64", lod_level=1)
#     slots_vars.append(var)

# dataset = paddle.distributed.InMemoryDataset()
# dataset.init(
#     batch_size=1,
#     thread_num=2,
#     input_type=1,
#     pipe_command="cat",
#     use_var=slots_vars)
# dataset.set_filelist(
#     ["test_queue_dataset_run_a.txt", "test_queue_dataset_run_b.txt"])
# dataset.load_into_memory()

# place = paddle.CPUPlace()
# exe = paddle.static.Executor(place)
# startup_program = paddle.static.Program()
# main_program = paddle.static.Program()
# exe.run(startup_program)

# exe.train_from_dataset(main_program, dataset)

# os.remove("./test_queue_dataset_run_a.txt")
# os.remove("./test_queue_dataset_run_b.txt")

# paddle.enable_static()

# dataset = paddle.distributed.InMemoryDataset()
# slots = ["slot1", "slot2", "slot3", "slot4"]
# slots_vars = []
# for slot in slots:
#     var = paddle.static.data(
#         name=slot, shape=[None, 1], dtype="int64", lod_level=1)
#     slots_vars.append(var)
# dataset.init(
#     batch_size=1,
#     thread_num=2,
#     input_type=1,
#     pipe_command="cat",
#     use_var=slots_vars)
# filelist = ["a.txt", "b.txt"]
# dataset.set_filelist(filelist)
# dataset.load_into_memory()

# paddle.enable_static()

# dataset = paddle.distributed.InMemoryDataset()
# slots = ["slot1", "slot2", "slot3", "slot4"]
# slots_vars = []
# for slot in slots:
#     var = paddle.static.data(
#         name=slot, shape=[None, 1], dtype="int64", lod_level=1)
#     slots_vars.append(var)
# dataset.init(
#     batch_size=1,
#     thread_num=2,
#     input_type=1,
#     pipe_command="cat",
#     use_var=slots_vars)
# filelist = ["a.txt", "b.txt"]
# dataset.set_filelist(filelist)
# dataset.preload_into_memory()
# dataset.wait_preload_done()

# paddle.enable_static()

# dataset = paddle.distributed.InMemoryDataset()
# slots = ["slot1", "slot2", "slot3", "slot4"]
# slots_vars = []
# for slot in slots:
#     var = paddle.static.data(
#         name=slot, shape=[None, 1], dtype="int64", lod_level=1)
#     slots_vars.append(var)
# dataset.init(
#     batch_size=1,
#     thread_num=2,
#     input_type=1,
#     pipe_command="cat",
#     use_var=slots_vars)
# filelist = ["a.txt", "b.txt"]
# dataset.set_filelist(filelist)
# dataset.preload_into_memory()
# dataset.wait_preload_done()

# paddle.enable_static()

# dataset = paddle.distributed.InMemoryDataset()
# slots = ["slot1", "slot2", "slot3", "slot4"]
# slots_vars = []
# for slot in slots:
#     var = paddle.static.data(
#         name=slot, shape=[None, 1], dtype="int64", lod_level=1)
#     slots_vars.append(var)
# dataset.init(
#     batch_size=1,
#     thread_num=2,
#     input_type=1,
#     pipe_command="cat",
#     use_var=slots_vars)
# filelist = ["a.txt", "b.txt"]
# dataset.set_filelist(filelist)
# dataset.load_into_memory()
# dataset.local_shuffle()

# paddle.enable_static()

# dataset = paddle.distributed.InMemoryDataset()
# slots = ["slot1", "slot2", "slot3", "slot4"]
# slots_vars = []
# for slot in slots:
#     var = paddle.static.data(
#         name=slot, shape=[None, 1], dtype="int64", lod_level=1)
#     slots_vars.append(var)
# dataset.init(
#     batch_size=1,
#     thread_num=2,
#     input_type=1,
#     pipe_command="cat",
#     use_var=slots_vars)
# filelist = ["a.txt", "b.txt"]
# dataset.set_filelist(filelist)
# dataset.load_into_memory()
# dataset.global_shuffle()

# paddle.enable_static()

# dataset = paddle.distributed.InMemoryDataset()
# slots = ["slot1", "slot2", "slot3", "slot4"]
# slots_vars = []
# for slot in slots:
#     var = paddle.static.data(
#         name=slot, shape=[None, 1], dtype="int64", lod_level=1)
#     slots_vars.append(var)
# dataset.init(
#     batch_size=1,
#     thread_num=2,
#     input_type=1,
#     pipe_command="cat",
#     use_var=slots_vars)
# filelist = ["a.txt", "b.txt"]
# dataset.set_filelist(filelist)
# dataset.load_into_memory()
# dataset.global_shuffle()
# exe = paddle.static.Executor(paddle.CPUPlace())
# startup_program = paddle.static.Program()
# main_program = paddle.static.Program()
# exe.run(startup_program)
# exe.train_from_dataset(main_program, dataset)
# dataset.release_memory()

# paddle.enable_static()

# dataset = paddle.distributed.InMemoryDataset()
# slots = ["slot1", "slot2", "slot3", "slot4"]
# slots_vars = []
# for slot in slots:
#     var = paddle.static.data(
#         name=slot, shape=[None, 1], dtype="int64", lod_level=1)
#     slots_vars.append(var)
# dataset.init(
#     batch_size=1,
#     thread_num=2,
#     input_type=1,
#     pipe_command="cat",
#     use_var=slots_vars)
# filelist = ["a.txt", "b.txt"]
# dataset.set_filelist(filelist)
# dataset.load_into_memory()

# paddle.enable_static()

# dataset = paddle.distributed.InMemoryDataset()
# dataset = paddle.distributed.InMemoryDataset()
# slots = ["slot1", "slot2", "slot3", "slot4"]
# slots_vars = []
# for slot in slots:
#     var = paddle.static.data(
#         name=slot, shape=[None, 1], dtype="int64", lod_level=1)
#     slots_vars.append(var)
# dataset.init(
#     batch_size=1,
#     thread_num=2,
#     input_type=1,
#     pipe_command="cat",
#     use_var=slots_vars)
# filelist = ["a.txt", "b.txt"]
# dataset.set_filelist(filelist)
# dataset.load_into_memory()
# dataset.global_shuffle()

# paddle.enable_static()

# dataset = paddle.distributed.InMemoryDataset()
# dataset._init_distributed_settings(fea_eval=True)
# slots = ["slot1", "slot2", "slot3", "slot4"]
# slots_vars = []
# for slot in slots:
#     var = paddle.static.data(
#         name=slot, shape=[None, 1], dtype="int64", lod_level=1)
#     slots_vars.append(var)
# dataset.init(
#     batch_size=1,
#     thread_num=2,
#     input_type=1,
#     pipe_command="cat",
#     use_var=slots_vars)
# filelist = ["a.txt", "b.txt"]
# dataset.set_filelist(filelist)
# dataset.load_into_memory()
# dataset.slots_shuffle(['slot1'])

# dataset = paddle.distributed.fleet.DatasetBase()
# dataset.set_filelist(['a.txt', 'b.txt'])

# paddle.distributed.init_parallel_env()
# tindata = paddle.randn(shape=[2, 3])
# gp = paddle.distributed.new_group([2,4,6])
# paddle.distributed.all_reduce(tindata, group=gp, use_calc_stream=False)

# def train():
    
#     dist.init_parallel_env()

    
#     parallel_env = dist.ParallelEnv()
#     print("rank: ", parallel_env.rank)
#     print("world_size: ", parallel_env.world_size)

    
    
    
    
    
    

# if __name__ == '__main__':
    
#     dist.spawn(train, nprocs=2)
    
    

# env = dist.ParallelEnv()
# print("The rank is %d" % env.rank)

# env = dist.ParallelEnv()
# print("The world_size is %d" % env.world_size)

# env = dist.ParallelEnv()
# print("The device id are %d" % env.device_id)

# env = dist.ParallelEnv()
# print("The current endpoint are %s" % env.current_endpoint)

# env = dist.ParallelEnv()
# print("The trainer endpoints are %s" % env.trainer_endpoints)

# env = dist.ParallelEnv()
# print("The nrings is %d" % env.nrings)

# env = dist.ParallelEnv()
# print("The rank is %d" % env.rank)

# env = dist.ParallelEnv()
# print("The world_size is %d" % env.world_size)

# env = dist.ParallelEnv()
# print("The device id are %d" % env.device_id)

# sparse_feature_dim = 1024
# embedding_size = 64

# entry = paddle.distributed.ProbabilityEntry(0.1)

# input = paddle.static.data(name='ins', shape=[1], dtype='int64')

# emb = paddle.static.nn.sparse_embedding(
#     input=input,
#     size=[sparse_feature_dim, embedding_size],
#     is_test=False,
#     entry=entry,
#     param_attr=paddle.ParamAttr(name="SparseFeatFactors",
#                                initializer=paddle.nn.initializer.Uniform()))

# dataset = paddle.distributed.QueueDataset()

# dataset = paddle.distributed.fleet.DatasetBase()
# dataset.set_filelist(['a.txt', 'b.txt'])

# init_parallel_env()
# if paddle.distributed.ParallelEnv().rank == 0:
#     data = paddle.to_tensor([7, 8, 9])
#     paddle.distributed.send(data, dst=1)
# else:
#     data = paddle.to_tensor([1,2,3])
#     paddle.distributed.recv(data, src=0)
# out = data.numpy()

# paddle.set_device('gpu:%d'%paddle.distributed.ParallelEnv().dev_id)
# init_parallel_env()
# if paddle.distributed.ParallelEnv().local_rank == 0:
#     np_data = np.array([[4, 5, 6], [4, 5, 6]])
# else:
#     np_data = np.array([[1, 2, 3], [1, 2, 3]])
# data = paddle.to_tensor(np_data)
# paddle.distributed.reduce(data, 0)
# out = data.numpy()

# paddle.set_device('gpu:%d'%paddle.distributed.ParallelEnv().dev_id)
# init_parallel_env()
# if paddle.distributed.ParallelEnv().local_rank == 0:
#     np_data = np.array([[4, 5, 6], [4, 5, 6]])
# else:
#     np_data = np.array([[1, 2, 3], [1, 2, 3]])
# data = paddle.to_tensor(np_data)
# paddle.distributed.all_reduce(data, op=ReduceOp.SUM)
# out = data.numpy()

# paddle.set_device('gpu:%d'%paddle.distributed.ParallelEnv().dev_id)
# init_parallel_env()
# if paddle.distributed.ParallelEnv().local_rank == 0:
#     np_data1 = np.array([7, 8, 9])
#     np_data2 = np.array([10, 11, 12])
# else:
#     np_data1 = np.array([1, 2, 3])
#     np_data2 = np.array([4, 5, 6])
# data1 = paddle.to_tensor(np_data1)
# data2 = paddle.to_tensor(np_data2)
# if paddle.distributed.ParallelEnv().local_rank == 0:
#     paddle.distributed.scatter(data1, src=1)
# else:
#     paddle.distributed.scatter(data1, tensor_list=[data1, data2], src=1)
# out = data1.numpy()

# init_parallel_env()
# if paddle.distributed.ParallelEnv().rank == 0:
#     data = paddle.to_tensor([7, 8, 9])
#     paddle.distributed.send(data, dst=1)
# else:
#     data = paddle.to_tensor([1,2,3])
#     paddle.distributed.recv(data, src=0)
# out = data.numpy()

# class LinearNet(nn.Layer):
#     def __init__(self):
#         super(LinearNet, self).__init__()
#         self._linear1 = nn.Linear(10, 10)
#         self._linear2 = nn.Linear(10, 1)

#     def forward(self, x):
#         return self._linear2(self._linear1(x))

# def train(print_result=False):
    
#     dist.init_parallel_env()

    
#     layer = LinearNet()
#     dp_layer = paddle.DataParallel(layer)

#     loss_fn = nn.MSELoss()
#     adam = opt.Adam(
#         learning_rate=0.001, parameters=dp_layer.parameters())

    
#     inputs = paddle.randn([10, 10], 'float32')
#     outputs = dp_layer(inputs)
#     labels = paddle.randn([10, 1], 'float32')
#     loss = loss_fn(outputs, labels)

#     if print_result is True:
#         print("loss:", loss.numpy())

#     loss.backward()

#     adam.step()
#     adam.clear_grad()

# if __name__ == '__main__':
#     dist.spawn(train)

# if __name__ == '__main__':
#     dist.spawn(train, args=(True,))

# if __name__ == '__main__':
#     dist.spawn(train, args=(True,), nprocs=2)

# if __name__ == '__main__':
#     dist.spawn(train, args=(True,), nprocs=2, gpus='4,5')

# paddle.enable_static()
# paddle.set_device('gpu:%d'%paddle.distributed.ParallelEnv().dev_id)
# fleet.init(is_collective=True)
# data = paddle.randint(0, 8, shape=[10,4])
# emb_out = paddle.distributed.split(
#     data,
#     (8, 8),
#     operation="embedding",
#     num_partitions=2)

# paddle.distributed.init_parallel_env()
# tindata = paddle.randn(shape=[2, 3])
# paddle.distributed.all_reduce(tindata, use_calc_stream=True)
# paddle.distributed.wait(tindata)

# paddle.seed(100) 
# x = paddle.rand([6])
# print(x)

# paddle.seed(200) 
# y = paddle.rand([6])
# print(y)

# cat = Categorical(x)
# cat2 = Categorical(y)

# paddle.seed(1000) 
# cat.sample([2,3])

# cat.entropy()

# cat.kl_divergence(cat2)

# value = paddle.to_tensor([2,1,3])
# cat.probs(value)

# cat.log_prob(value)

# paddle.seed(100) 
# x = paddle.rand([6])
# print(x)

# cat = Categorical(x)

# paddle.seed(1000) 
# cat.sample([2,3])

# paddle.seed(100) 
# x = paddle.rand([6])
# print(x)

# paddle.seed(200) 
# y = paddle.rand([6])
# print(y)

# cat = Categorical(x)
# cat2 = Categorical(y)

# cat.kl_divergence(cat2)

# paddle.seed(100) 
# x = paddle.rand([6])
# print(x)

# cat = Categorical(x)

# cat.entropy()

# paddle.seed(100) 
# x = paddle.rand([6])
# print(x)

# cat = Categorical(x)

# value = paddle.to_tensor([2,1,3])
# cat.probs(value)

# paddle.seed(100) 
# x = paddle.rand([6])
# print(x)

# cat = Categorical(x)

# value = paddle.to_tensor([2,1,3])
# cat.log_prob(value)

# dist = Normal(loc=0., scale=3.)

# dist = Normal(loc=[1., 2.], scale=[11., 22.])

# dist.sample([3])

# dist = Normal(loc=1., scale=[11., 22.])

# value_tensor = paddle.to_tensor([0.8], dtype="float32")

# normal_a = Normal([0.], [1.])
# normal_b = Normal([0.5], [2.])
# sample = normal_a.sample([2])

# entropy = normal_a.entropy()

# lp = normal_a.log_prob(value_tensor)

# p = normal_a.probs(value_tensor)

# kl = normal_a.kl_divergence(normal_b)

# u1 = Uniform(low=3.0, high=4.0)

# u2 = Uniform(low=[1.0, 2.0], high=[3.0, 4.0])

# u3 = Uniform(low=[[1.0, 2.0], [3.0, 4.0]],
#           high=[[1.5, 2.5], [3.5, 4.5]])

# u4 = Uniform(low=3.0, high=[5.0, 6.0, 7.0])

# value_tensor = paddle.to_tensor([0.8], dtype="float32")

# uniform = Uniform([0.], [2.])

# sample = uniform.sample([2])

# entropy = uniform.entropy()

# lp = uniform.log_prob(value_tensor)

# p = uniform.probs(value_tensor)

# x = np.exp(3j * np.pi * np.arange(7) / 7)
# xp = paddle.to_tensor(x)
# fft_xp = paddle.fft.fft(xp).numpy()
# print(fft_xp)

# x = np.mgrid[:2, :2][1]
# xp = paddle.to_tensor(x)
# fft2_xp = paddle.fft.fft2(xp).numpy()
# print(fft2_xp)

# x = np.array([3, 1, 2, 2, 3], dtype=float)
# scalar_temp = 0.5
# n = x.size
# fftfreq_xp = paddle.fft.fftfreq(n, d=scalar_temp)
# print(fftfreq_xp)

# x = np.mgrid[:4, :4, :4][1]
# xp = paddle.to_tensor(x)
# fftn_xp = paddle.fft.fftn(xp, axes=(1, 2)).numpy()
# print(fftn_xp)

# x = np.array([3, 1, 2, 2, 3], dtype=float)
# n = x.size
# fftfreq_xp = paddle.fft.fftfreq(n, d=0.3)
# res = paddle.fft.fftshift(fftfreq_xp).numpy()
# print(res)

# x = np.array([1, -1j, -1])
# xp = paddle.to_tensor(x)
# hfft_xp = paddle.fft.hfft(xp).numpy()
# print(hfft_xp)

# x = (np.array([[3,2,3],[2, 2, 3]]) + 1j * np.array([[3,2,3],[2, 2, 3]])).astype(np.complex128)
# xp = paddle.to_tensor(x)
# hfft2_xp = paddle.fft.hfft2(xp).numpy()
# print(hfft2_xp)

# x = (np.array([2, 2, 3]) + 1j * np.array([2, 2, 3])).astype(np.complex128)
# xp = paddle.to_tensor(x)
# hfftn_xp = paddle.fft.hfftn(xp).numpy()
# print(hfftn_xp)

# x = np.exp(3j * np.pi * np.arange(7) / 7)
# xp = paddle.to_tensor(x)
# ifft_xp = paddle.fft.ifft(xp).numpy()
# print(ifft_xp)

# x = np.mgrid[:2, :2][1]
# xp = paddle.to_tensor(x)
# ifft2_xp = paddle.fft.ifft2(xp).numpy()
# print(ifft2_xp)

# x = np.eye(3)
# xp = paddle.to_tensor(x)
# ifftn_xp = paddle.fft.ifftn(xp, axes=(1,)).numpy()
# print(ifftn_xp)

# x = np.array([3, 1, 2, 2, 3], dtype=float)
# n = x.size
# fftfreq_xp = paddle.fft.fftfreq(n, d=0.3)
# res = paddle.fft.ifftshift(fftfreq_xp).numpy()
# print(res)

# x = np.mgrid[:5, :5][0].astype(np.float64)
# xp = paddle.to_tensor(x)
# ihfft2_xp = paddle.fft.ihfft2(xp).numpy()
# print(ihfft2_xp)

# x = np.array([1, -1j, -1])
# xp = paddle.to_tensor(x)
# irfft_xp = paddle.fft.irfft(xp).numpy()
# print(irfft_xp)

# x = (np.array([[3,2,3],[2, 2, 3]]) + 1j * np.array([[3,2,3],[2, 2, 3]])).astype(np.complex128)
# xp = paddle.to_tensor(x)
# irfft2_xp = paddle.fft.irfft2(xp).numpy()
# print(irfft2_xp)

# x = (np.array([2, 2, 3]) + 1j * np.array([2, 2, 3])).astype(np.complex128)
# xp = paddle.to_tensor(x)
# irfftn_xp = paddle.fft.irfftn(xp).numpy()
# print(irfftn_xp)


# x = paddle.to_tensor(np.mgrid[:5, :5][0].astype(np.float32))
# print(paddle.fft.rfft2(x))

    
    
    


# x = np.array([3, 1, 2, 2, 3], dtype=float)
# scalar_temp = 0.3
# n = x.size
# rfftfreq_xp = paddle.fft.rfftfreq(n, d=scalar_temp)
# print(rfftfreq_xp)

# paddle.enable_static()

# x = fluid.data(name='x', shape=[3, 2, 1], dtype='float32')

# y = fluid.data(name='y', shape=[-1, 2, 1], dtype='float32')

# z = x + y

# feed_data = np.ones(shape=[3, 2, 1], dtype=np.float32)

# exe = fluid.Executor(fluid.CPUPlace())
# out = exe.run(fluid.default_main_program(),
#               feed={
#                   'x': feed_data,
#                   'y': feed_data
#               },
#               fetch_list=[z.name])

# print(out)

# paddle.hub.help('lyuwenyu/paddlehub_demo:main', model='MM', source='github')

# paddle.hub.list('lyuwenyu/paddlehub_demo:main', source='github', force_reload=False)

# paddle.hub.load('lyuwenyu/paddlehub_demo:main', model='MM', source='github')

# x = paddle.to_tensor([[0, 2, 3], [1, 4, 5], [2, 6, 7]], dtype="float32")
# indexes = paddle.to_tensor([[0, 1], [1, 2], [2, 1], [0, 0]], dtype="int32")
# src_index = indexes[:, 0]
# dst_index = indexes[:, 1]
# out = paddle.incubate.graph_send_recv(x, src_index, dst_index, pool_type="sum")

# BATCH_SIZE = 16
# BATCH_NUM = 4
# EPOCH_NUM = 4

# IMAGE_SIZE = 784
# CLASS_NUM = 10

# class RandomDataset(paddle.io.Dataset):
#     def __init__(self, num_samples):
#         self.num_samples = num_samples

#     def __getitem__(self, idx):
#         image = np.random.random([IMAGE_SIZE]).astype('float32')
#         label = np.random.randint(0, CLASS_NUM - 1,
#                                 (1, )).astype('int64')
#         return image, label

#     def __len__(self):
#         return self.num_samples

# class LinearNet(nn.Layer):
#     def __init__(self):
#         super(LinearNet, self).__init__()
#         self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)
#         self.bias = self._linear.bias

#     @paddle.jit.to_static
#     def forward(self, x):
#         return self._linear(x)

# def train(layer, loader, loss_fn, opt):
#     for epoch_id in range(EPOCH_NUM):
#         for batch_id, (image, label) in enumerate(loader()):
#             out = layer(image)
#             loss = loss_fn(out, label)
#             loss.backward()
#             opt.step()
#             opt.clear_grad()
#             print("Train Epoch {} batch {}: loss = {}".format(
#                 epoch_id, batch_id, np.mean(loss.numpy())))

# layer = LinearNet()
# loss_fn = nn.CrossEntropyLoss()
# optimizer = paddle.optimizer.SGD(learning_rate=0.1, parameters=layer.parameters())
# lookahead = paddle.incubate.LookAhead(optimizer, alpha=0.2, k=5)

# dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
# loader = paddle.io.DataLoader(
#     dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     drop_last=True,
#     num_workers=2)

# train(layer, loader, loss_fn, lookahead)

# inp = paddle.to_tensor(np.random.random([1, 10]).astype('float32'))
# linear = paddle.nn.Linear(10, 1)
# out = linear(inp)
# loss = paddle.mean(out)
# sgd = paddle.optimizer.SGD(learning_rate=0.1,parameters=linear.parameters())
# lookahead = paddle.incubate.LookAhead(sgd, alpha=0.2, k=5)
# loss.backward()
# lookahead.step()
# lookahead.clear_grad()

# inp = paddle.to_tensor(np.random.random([1, 10]).astype('float32'))
# linear = paddle.nn.Linear(10, 1)
# out = linear(inp)
# loss = paddle.mean(out)
# sgd = paddle.optimizer.SGD(learning_rate=0.1,parameters=linear.parameters())
# lookahead = paddle.incubate.LookAhead(sgd, alpha=0.2, k=5)
# loss.backward()
# lookahead.minimize(loss)
# lookahead.clear_grad()

# inp = np.random.uniform(-0.1, 0.1, [10, 10]).astype("float32")
# linear = paddle.nn.Linear(10, 10)
# inp = paddle.to_tensor(inp)
# out = linear(inp)
# loss = paddle.mean(out)
# optimizer = paddle.optimizer.Adam(learning_rate=0.1,
#         parameters=linear.parameters())
# params_grads = optimizer.backward(loss)
# optimizer.apply_gradients(params_grads)

# value = np.arange(26).reshape(2, 13).astype("float32")
# a = paddle.to_tensor(value)
# linear = paddle.nn.Linear(13, 5)

# adam = paddle.optimizer.Adam(learning_rate = 0.01,
#                             parameters = linear.parameters())
# out = linear(a)
# out.backward()
# adam.step()
# adam.clear_grad()

# value = np.arange(26).reshape(2, 13).astype("float32")
# a = paddle.to_tensor(value)
# linear = paddle.nn.Linear(13, 5)

# adam = paddle.optimizer.Adam(learning_rate = 0.01,
#                             parameters = linear.parameters())
# out = linear(a)
# out.backward()
# adam.step()
# adam.clear_grad()

# emb = paddle.nn.Embedding(10, 3)

# adam = paddle.optimizer.Adam(0.01, parameters = emb.parameters())
# for batch in range(10):
#     input = paddle.randint(low=0, high=5, shape=[5])
#     out = emb(input)
#     out.backward()
#     print("Learning rate of step{}: {}".format(batch, adam.get_lr())) 
#     adam.step()

# scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.5, step_size=2, gamma=0.1)
# adam = paddle.optimizer.Adam(scheduler, parameters = emb.parameters())
# for batch in range(10):
#     input = paddle.randint(low=0, high=5, shape=[5])
#     out = emb(input)
#     out.backward()
#     print("Learning rate of step{}: {}".format(batch, adam.get_lr())) 
#     adam.step()
#     scheduler.step()

# paddle.enable_static()
# main_prog = paddle.static.Program()
# start_prog = paddle.static.Program()
# with paddle.static.program_guard(main_prog, start_prog):
#     x = paddle.static.data(name='x', shape=[None, 10])
#     z = paddle.static.nn.fc(x, 100)
#     loss = paddle.mean(z)
#     scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.5, step_size=2, gamma=0.1)
#     adam = paddle.optimizer.Adam(learning_rate=scheduler)
#     adam.minimize(loss)

# exe = paddle.static.Executor()
# exe.run(start_prog)
# for batch in range(10):
#     print("Learning rate of step{}: {}", adam.get_lr())     
#     out = exe.run(main_prog, feed={'x': np.random.randn(3, 10).astype('float32')})
#     scheduler.step()

# linear = paddle.nn.Linear(10, 10)

# adam = paddle.optimizer.Adam(0.1, parameters=linear.parameters())

# lr_list = [0.2, 0.3, 0.4, 0.5, 0.6]
# for i in range(5):
#     adam.set_lr(lr_list[i])
#     lr = adam.get_lr()
#     print("current lr is {}".format(lr))

# emb = paddle.nn.Embedding(10, 10)

# layer_state_dict = emb.state_dict()
# paddle.save(layer_state_dict, "emb.pdparams")

# scheduler = paddle.optimizer.lr.NoamDecay(
#     d_model=0.01, warmup_steps=100, verbose=True)
# adam = paddle.optimizer.Adam(
#     learning_rate=scheduler,
#     parameters=emb.parameters())
# opt_state_dict = adam.state_dict()
# paddle.save(opt_state_dict, "adam.pdopt")

# opti_state_dict = paddle.load("adam.pdopt")
# adam.set_state_dict(opti_state_dict)

# emb = paddle.nn.Embedding(10, 10)

# adam = paddle.optimizer.Adam(0.001, parameters=emb.parameters())
# state_dict = adam.state_dict()
# if num_accumulates >= min_average_window and num_accumulates >= min(max_average_window, num_updates * average_window_rate):
#     num_accumulates = 0

# BATCH_SIZE = 16
# BATCH_NUM = 4
# EPOCH_NUM = 4

# IMAGE_SIZE = 784
# CLASS_NUM = 10

# class RandomDataset(paddle.io.Dataset):
#     def __init__(self, num_samples):
#         self.num_samples = num_samples

#     def __getitem__(self, idx):
#         image = np.random.random([IMAGE_SIZE]).astype('float32')
#         label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
#         return image, label

#     def __len__(self):
#         return self.num_samples

# class LinearNet(nn.Layer):
#     def __init__(self):
#         super(LinearNet, self).__init__()
#         self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)
#         self.bias = self._linear.bias

#     @paddle.jit.to_static
#     def forward(self, x):
#         return self._linear(x)

# def train(layer, loader, loss_fn, opt, model_average):
#     for epoch_id in range(EPOCH_NUM):
#         for batch_id, (image, label) in enumerate(loader()):
#             out = layer(image)
#             loss = loss_fn(out, label)
#             loss.backward()
#             opt.step()
#             model_average.step()
#             opt.clear_grad()
#             model_average.clear_grad()
#             print("Train Epoch {} batch {}: loss = {}, bias = {}".format(
#                 epoch_id, batch_id, np.mean(loss.numpy()), layer.bias.numpy()))
# def evaluate(layer, loader, loss_fn):
#     for batch_id, (image, label) in enumerate(loader()):
#         out = layer(image)
#         loss = loss_fn(out, label)
#         loss.backward()
#         print("Evaluate batch {}: loss = {}, bias = {}".format(
#             batch_id, np.mean(loss.numpy()), layer.bias.numpy()))

# layer = LinearNet()
# loss_fn = nn.CrossEntropyLoss()
# optimizer = opt.Momentum(learning_rate=0.2, momentum=0.1, parameters=layer.parameters())
# model_average = paddle.incubate.ModelAverage(0.15,
#                                             parameters=layer.parameters(),
#                                             min_average_window=2,
#                                             max_average_window=10)

# dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
# loader = paddle.io.DataLoader(dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     drop_last=True,
#     num_workers=2)

# eval_loader = paddle.io.DataLoader(dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     drop_last=True,
#     num_workers=1)

# train(layer, loader, loss_fn, optimizer, model_average)

# print("\nEvaluate With ModelAverage")
# with model_average.apply(need_restore=False):
#     evaluate(layer, eval_loader, loss_fn)

# print("\nEvaluate With Restored Paramters")
# model_average.restore()
# evaluate(layer, eval_loader, loss_fn)

# inp = paddle.to_tensor(np.random.random([1, 10]).astype('float32'))
# linear = paddle.nn.Linear(10, 1)
# out = linear(inp)
# loss = paddle.mean(out)
# loss.backward()

# sgd = paddle.optimizer.SGD(learning_rate=0.1,parameters=linear.parameters())
# sgd.minimize(loss)

# modelaverage = paddle.incubate.ModelAverage(0.15,
#                                             parameters=linear.parameters(),
#                                             min_average_window=2,
#                                             max_average_window=4)
# modelaverage.minimize(loss)
# sgd.clear_grad()
# modelaverage.clear_grad()

# inp = paddle.to_tensor(np.random.random([1, 10]).astype('float32'))
# linear = paddle.nn.Linear(10, 1)
# out = linear(inp)
# loss = paddle.mean(out)
# sgd = paddle.optimizer.SGD(learning_rate=0.1,parameters=linear.parameters())
# modelaverage = paddle.incubate.ModelAverage(0.15,
#                                             parameters=linear.parameters(),
#                                             min_average_window=2,
#                                             max_average_window=4)
# loss.backward()
# sgd.step()
# modelaverage.step()
# sgd.clear_grad()
# modelaverage.clear_grad()

# inp = paddle.to_tensor(np.random.random([1, 10]).astype('float32'))
# linear = paddle.nn.Linear(10, 1)
# out = linear(inp)
# loss = paddle.mean(out)
# loss.backward()

# sgd = paddle.optimizer.SGD(learning_rate=0.1,parameters=linear.parameters())

# modelaverage = paddle.incubate.ModelAverage(0.15,
#                                             parameters=linear.parameters(),
#                                             min_average_window=2,
#                                             max_average_window=4)
# sgd.step()
# modelaverage.step()

# with modelaverage.apply():
#     for param in linear.parameters():
#         print(param)

# for param in linear.parameters():
#     print(param)

# inp = paddle.to_tensor(np.random.random([1, 10]).astype('float32'))
# linear = paddle.nn.Linear(10, 1)
# out = linear(inp)
# loss = paddle.mean(out)
# loss.backward()

# sgd = paddle.optimizer.SGD(learning_rate=0.1,parameters=linear.parameters())

# modelaverage = paddle.incubate.ModelAverage(0.15,
#                                             parameters=linear.parameters(),
#                                             min_average_window=2,
#                                             max_average_window=4)
# sgd.step()
# modelaverage.step()

# with modelaverage.apply(need_restore=False):
#     for param in linear.parameters():
#         print(param)

# for param in linear.parameters():
#     print(param)

# modelaverage.restore()

# for param in linear.parameters():
#     print(param)

# inp = np.random.uniform(-0.1, 0.1, [10, 10]).astype("float32")
# linear = paddle.nn.Linear(10, 10)
# inp = paddle.to_tensor(inp)
# out = linear(inp)
# loss = paddle.mean(out)
# optimizer = paddle.optimizer.Adam(learning_rate=0.1,
#         parameters=linear.parameters())
# params_grads = optimizer.backward(loss)
# optimizer.apply_gradients(params_grads)

# value = np.arange(26).reshape(2, 13).astype("float32")
# a = paddle.to_tensor(value)
# linear = paddle.nn.Linear(13, 5)

# adam = paddle.optimizer.Adam(learning_rate = 0.01,
#                             parameters = linear.parameters())
# out = linear(a)
# out.backward()
# adam.step()
# adam.clear_grad()

# value = np.arange(26).reshape(2, 13).astype("float32")
# a = paddle.to_tensor(value)
# linear = paddle.nn.Linear(13, 5)

# adam = paddle.optimizer.Adam(learning_rate = 0.01,
#                             parameters = linear.parameters())
# out = linear(a)
# out.backward()
# adam.step()
# adam.clear_grad()

# emb = paddle.nn.Embedding(10, 3)

# adam = paddle.optimizer.Adam(0.01, parameters = emb.parameters())
# for batch in range(10):
#     input = paddle.randint(low=0, high=5, shape=[5])
#     out = emb(input)
#     out.backward()
#     print("Learning rate of step{}: {}".format(batch, adam.get_lr())) 
#     adam.step()

# scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.5, step_size=2, gamma=0.1)
# adam = paddle.optimizer.Adam(scheduler, parameters = emb.parameters())
# for batch in range(10):
#     input = paddle.randint(low=0, high=5, shape=[5])
#     out = emb(input)
#     out.backward()
#     print("Learning rate of step{}: {}".format(batch, adam.get_lr())) 
#     adam.step()
#     scheduler.step()

# paddle.enable_static()
# main_prog = paddle.static.Program()
# start_prog = paddle.static.Program()
# with paddle.static.program_guard(main_prog, start_prog):
#     x = paddle.static.data(name='x', shape=[None, 10])
#     z = paddle.static.nn.fc(x, 100)
#     loss = paddle.mean(z)
#     scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.5, step_size=2, gamma=0.1)
#     adam = paddle.optimizer.Adam(learning_rate=scheduler)
#     adam.minimize(loss)

# exe = paddle.static.Executor()
# exe.run(start_prog)
# for batch in range(10):
#     print("Learning rate of step{}: {}", adam.get_lr())     
#     out = exe.run(main_prog, feed={'x': np.random.randn(3, 10).astype('float32')})
#     scheduler.step()

# linear = paddle.nn.Linear(10, 10)

# adam = paddle.optimizer.Adam(0.1, parameters=linear.parameters())

# lr_list = [0.2, 0.3, 0.4, 0.5, 0.6]
# for i in range(5):
#     adam.set_lr(lr_list[i])
#     lr = adam.get_lr()
#     print("current lr is {}".format(lr))

# emb = paddle.nn.Embedding(10, 10)

# layer_state_dict = emb.state_dict()
# paddle.save(layer_state_dict, "emb.pdparams")

# scheduler = paddle.optimizer.lr.NoamDecay(
#     d_model=0.01, warmup_steps=100, verbose=True)
# adam = paddle.optimizer.Adam(
#     learning_rate=scheduler,
#     parameters=emb.parameters())
# opt_state_dict = adam.state_dict()
# paddle.save(opt_state_dict, "adam.pdopt")

# opti_state_dict = paddle.load("adam.pdopt")
# adam.set_state_dict(opti_state_dict)

# emb = paddle.nn.Embedding(10, 10)

# adam = paddle.optimizer.Adam(0.001, parameters=emb.parameters())
# state_dict = adam.state_dict()

# data = paddle.to_tensor([[1, 2, 3], [3, 2, 1], [4, 5, 6]], dtype='float32')
# segment_ids = paddle.to_tensor([0, 0, 1], dtype='int32')
# out = paddle.incubate.segment_max(data, segment_ids)

# data = paddle.to_tensor([[1, 2, 3], [3, 2, 1], [4, 5, 6]], dtype='float32')
# segment_ids = paddle.to_tensor([0, 0, 1], dtype='int32')
# out = paddle.incubate.segment_mean(data, segment_ids)

# data = paddle.to_tensor([[1, 2, 3], [3, 2, 1], [4, 5, 6]], dtype='float32')
# segment_ids = paddle.to_tensor([0, 0, 1], dtype='int32')
# out = paddle.incubate.segment_min(data, segment_ids)

# data = paddle.to_tensor([[1, 2, 3], [3, 2, 1], [4, 5, 6]], dtype='float32')
# segment_ids = paddle.to_tensor([0, 0, 1], dtype='int32')
# out = paddle.incubate.segment_sum(data, segment_ids)

# class RandomDataset(Dataset):
#     def __init__(self, num_samples):
#         self.num_samples = num_samples

#     def __getitem__(self, idx):
#         image = np.random.random([784]).astype('float32')
#         label = np.random.randint(0, 9, (1, )).astype('int64')
#         return image, label

#     def __len__(self):
#         return self.num_samples

# bs = BatchSampler(dataset=RandomDataset(100),
#                   shuffle=False,
#                   batch_size=16,
#                   drop_last=False)

# for batch_indices in bs:
#     print(batch_indices)

# sampler = RandomSampler(RandomDataset(100))
# bs = BatchSampler(sampler=sampler,
#                   batch_size=8,
#                   drop_last=True)

# for batch_indices in bs:
#     print(batch_indices)

# class RandomDataset(IterableDataset):
#     def __init__(self, num_samples):
#         self.num_samples = num_samples

#     def __iter__(self):
#         for i in range(10):
#             image = np.random.random([32]).astype('float32')
#             label = np.random.randint(0, 9, (1, )).astype('int64')
#             yield image, label

# dataset = ChainDataset([RandomDataset(10), RandomDataset(10)])
# for image, label in iter(dataset):
#     print(image, label)

# class RandomDataset(Dataset):
#     def __init__(self, num_samples):
#         self.num_samples = num_samples

#     def __getitem__(self, idx):
#         image = np.random.random([32]).astype('float32')
#         label = np.random.randint(0, 9, (1, )).astype('int64')
#         return image, label

#     def __len__(self):
#         return self.num_samples

# dataset = ComposeDataset([RandomDataset(10), RandomDataset(10)])
# for i in range(len(dataset)):
#     image1, label1, image2, label2 = dataset[i]
#     print(image1)
#     print(label1)
#     print(image2)
#     print(label2)

# BATCH_NUM = 20
# BATCH_SIZE = 16
# EPOCH_NUM = 4

# IMAGE_SIZE = 784
# CLASS_NUM = 10

# class RandomDataset(Dataset):
#     def __init__(self, num_samples):
#         self.num_samples = num_samples

#     def __getitem__(self, idx):
#         image = np.random.random([IMAGE_SIZE]).astype('float32')
#         label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
#         return image, label

#     def __len__(self):
#         return self.num_samples

# dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)

# class SimpleNet(nn.Layer):
#     def __init__(self):
#         super(SimpleNet, self).__init__()
#         self.fc = nn.Linear(IMAGE_SIZE, CLASS_NUM)

#     def forward(self, image, label=None):
#         return self.fc(image)

# simple_net = SimpleNet()
# opt = paddle.optimizer.SGD(learning_rate=1e-3,
#                           parameters=simple_net.parameters())

# loader = DataLoader(dataset,
#                     batch_size=BATCH_SIZE,
#                     shuffle=True,
#                     drop_last=True,
#                     num_workers=2)

# for e in range(EPOCH_NUM):
#     for i, (image, label) in enumerate(loader()):
#         out = simple_net(image)
#         loss = F.cross_entropy(out, label)
#         avg_loss = paddle.mean(loss)
#         avg_loss.backward()
#         opt.minimize(avg_loss)
#         simple_net.clear_gradients()
#         print("Epoch {} batch {}: loss = {}".format(e, i, np.mean(loss.numpy())))
# '''
# Example in static graph mode
# '''

# BATCH_NUM = 10
# BATCH_SIZE = 16
# EPOCH_NUM = 4

# CLASS_NUM = 10

# ITERABLE = True 
# USE_GPU = False 

# DATA_FORMAT = 'batch_generator' 

# paddle.enable_static()

# def simple_net(image, label):
#     fc_tmp = static.nn.fc(image, size=CLASS_NUM)
#     cross_entropy = F.softmax_with_cross_entropy(image, label)
#     loss = paddle.mean(cross_entropy)
#     sgd = paddle.optimizer.SGD(learning_rate=1e-3)
#     sgd.minimize(loss)
#     return loss

# def get_random_images_and_labels(image_shape, label_shape):
#     image = np.random.random(size=image_shape).astype('float32')
#     label = np.random.random(size=label_shape).astype('int64')
#     return image, label

# def sample_generator_creator():
#     def __reader__():
#         for _ in range(BATCH_NUM * BATCH_SIZE):
#             image, label = get_random_images_and_labels([784], [1])
#             yield image, label

#     return __reader__

# def sample_list_generator_creator():
#     def __reader__():
#         for _ in range(BATCH_NUM):
#             sample_list = []
#             for _ in range(BATCH_SIZE):
#                 image, label = get_random_images_and_labels([784], [1])
#                 sample_list.append([image, label])

#             yield sample_list

#     return __reader__

# def batch_generator_creator():
#     def __reader__():
#         for _ in range(BATCH_NUM):
#             batch_image, batch_label = get_random_images_and_labels([BATCH_SIZE, 784], [BATCH_SIZE, 1])
#             yield batch_image, batch_label

#     return __reader__

# def train_iterable(exe, prog, loss, loader):
#     for _ in range(EPOCH_NUM):
#         for data in loader():
#             exe.run(prog, feed=data, fetch_list=[loss])

# def train_non_iterable(exe, prog, loss, loader):
#     for _ in range(EPOCH_NUM):
#         loader.start() 
#         try:
#             while True:
#                 exe.run(prog, fetch_list=[loss])
#         except paddle.core.EOFException:
#             loader.reset() 

# def set_data_source(loader, places):
#     if DATA_FORMAT == 'sample_generator':
#         loader.set_sample_generator(sample_generator_creator(), batch_size=BATCH_SIZE, drop_last=True, places=places)
#     elif DATA_FORMAT == 'sample_list_generator':
#         loader.set_sample_list_generator(sample_list_generator_creator(), places=places)
#     elif DATA_FORMAT == 'batch_generator':
#         loader.set_batch_generator(batch_generator_creator(), places=places)
#     else:
#         raise ValueError('Unsupported data format')

# image = static.data(name='image', shape=[None, 784], dtype='float32')
# label = static.data(name='label', shape=[None, 1], dtype='int64')

# loader = paddle.io.DataLoader.from_generator(feed_list=[image, label], capacity=16, iterable=ITERABLE)

# loss = simple_net(image, label)

# places = static.cuda_places() if USE_GPU else static.cpu_places()
# set_data_source(loader, places)

# exe = static.Executor(places[0])
# exe.run(static.default_startup_program())

# prog = static.CompiledProgram(static.default_main_program()).with_data_parallel(loss_name=loss.name)

# if loader.iterable:
#     train_iterable(exe, prog, loss, loader)
# else:
#     train_non_iterable(exe, prog, loss, loader)
# '''
# Example in dynamic graph mode.
# '''

# BATCH_SIZE = 16
# BATCH_NUM = 4
# EPOCH_NUM = 4

# IMAGE_SIZE = 784
# CLASS_NUM = 10

# USE_GPU = False 

# def _get_random_images_and_labels(image_shape, label_shape):
#         image = np.random.random(size=image_shape).astype('float32')
#         label = np.random.random(size=label_shape).astype('int64')
#         return image, label

# def __reader__():
#         for _ in range(BATCH_NUM):
#             batch_image, batch_label = _get_random_images_and_labels(
#                 [BATCH_SIZE, IMAGE_SIZE], [BATCH_SIZE, CLASS_NUM])
#             yield batch_image, batch_label

# def random_batch_reader():
#     return __reader__

# class LinearNet(nn.Layer):
#     def __init__(self):
#         super(LinearNet, self).__init__()
#         self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

#     @paddle.jit.to_static
#     def forward(self, x):
#         return self._linear(x)

# paddle.set_device('gpu' if USE_GPU else 'cpu')

# layer = LinearNet()
# dp_layer = paddle.DataParallel(layer)
# loss_fn = nn.CrossEntropyLoss()
# adam = opt.Adam(learning_rate=0.001, parameters=dp_layer.parameters())

# loader = paddle.io.DataLoader.from_generator(capacity=5)
# loader.set_batch_generator(random_batch_reader())

# for epoch_id in range(EPOCH_NUM):
#     for batch_id, (image, label) in enumerate(loader()):
#         out = layer(image)
#         loss = loss_fn(out, label)

#         loss.backward()

#         adam.step()
#         adam.clear_grad()
#         print("Epoch {} batch {}: loss = {}".format(
#             epoch_id, batch_id, np.mean(loss.numpy())))
# '''
# Example of `drop_last` using in static graph multi-cards mode
# '''

# os.environ['CPU_NUM'] = '2'

# paddle.enable_static()

# def batch_generator():
#     for i in range(3):
#         yield np.array([i+1]).astype('float32'),

# x = static.data(name='x', shape=[None], dtype='float32')
# y = x * x

# def run_inference(drop_last):
#     loader = paddle.io.DataLoader.from_generator(feed_list=[x],
#             capacity=8, drop_last=drop_last)
#     loader.set_batch_generator(batch_generator, static.cpu_places())

#     exe = static.Executor(paddle.CPUPlace())
#     prog = static.CompiledProgram(static.default_main_program())
#     prog = prog.with_data_parallel()

#     result = []
#     for data in loader():
#         each_ret, = exe.run(prog, feed=data, fetch_list=[y])
#         result.extend(each_ret)
#     return result

# print(run_inference(drop_last=True)) 

# print(run_inference(drop_last=False)) 

# paddle.enable_static()

# image = static.data(name='image', shape=[None, 784], dtype='float32')
# label = static.data(name='label', shape=[None, 1], dtype='int64')

# dataset = paddle.distributed.QueueDataset()
# dataset.init(
#     batch_size=32,
#     pipe_command='cat',
#     use_var=[image, label])
# dataset.set_filelist(['a.txt', 'b.txt', 'c.txt'])

# loader = paddle.io.DataLoader.from_dataset(dataset, static.cpu_places())

# class RandomDataset(Dataset):
#     def __init__(self, num_samples):
#         self.num_samples = num_samples

#     def __getitem__(self, idx):
#         image = np.random.random([784]).astype('float32')
#         label = np.random.randint(0, 9, (1, )).astype('int64')
#         return image, label

#     def __len__(self):
#         return self.num_samples

# dataset = RandomDataset(10)
# for i in range(len(dataset)):
#     print(dataset[i])

# class RandomDataset(Dataset):
#     def __init__(self, num_samples):
#         self.num_samples = num_samples

#     def __getitem__(self, idx):
#         image = np.random.random([784]).astype('float32')
#         label = np.random.randint(0, 9, (1, )).astype('int64')
#         return image, label

#     def __len__(self):
#         return self.num_samples

# dataset = RandomDataset(100)
# sampler = DistributedBatchSampler(dataset, batch_size=64)

# for data in sampler:
    
#     break

# class RandomDataset(Dataset):
#     def __init__(self, num_samples):
#         self.num_samples = num_samples

#     def __getitem__(self, idx):
#         image = np.random.random([784]).astype('float32')
#         label = np.random.randint(0, 9, (1, )).astype('int64')
#         return image, label

#     def __len__(self):
#         return self.num_samples

# dataset = RandomDataset(100)
# sampler = DistributedBatchSampler(dataset, batch_size=64)

# for epoch in range(10):
#     sampler.set_epoch(epoch)

# class SplitedIterableDataset(IterableDataset):
#     def __init__(self, start, end):
#         self.start = start
#         self.end = end

#     def __iter__(self):
#         worker_info = get_worker_info()
#         if worker_info is None:
#             iter_start = self.start
#             iter_end = self.end
#         else:
#             per_worker = int(
#                 math.ceil((self.end - self.start) / float(
#                     worker_info.num_workers)))
#             worker_id = worker_info.id
#             iter_start = self.start + worker_id * per_worker
#             iter_end = min(iter_start + per_worker, self.end)

#         for i in range(iter_start, iter_end):
#             yield np.array([i])

# place = paddle.CPUPlace()
# dataset = SplitedIterableDataset(start=2, end=9)
# dataloader = DataLoader(
#     dataset,
#     places=place,
#     num_workers=2,
#     batch_size=1,
#     drop_last=True)

# for data in dataloader:
#     print(data)

# class RandomDataset(IterableDataset):
#     def __init__(self, num_samples):
#         self.num_samples = num_samples

#     def __iter__(self):
#         for i in range(self.num_samples):
#             image = np.random.random([784]).astype('float32')
#             label = np.random.randint(0, 9, (1, )).astype('int64')
#             yield image, label

# dataset = RandomDataset(10)
# for img, lbl in dataset:
#     print(img, lbl)

# class SplitedIterableDataset(IterableDataset):
#     def __init__(self, start, end):
#         self.start = start
#         self.end = end

#     def __iter__(self):
#         worker_info = get_worker_info()
#         if worker_info is None:
#             iter_start = self.start
#             iter_end = self.end
#         else:
#             per_worker = int(
#                 math.ceil((self.end - self.start) / float(
#                     worker_info.num_workers)))
#             worker_id = worker_info.id
#             iter_start = self.start + worker_id * per_worker
#             iter_end = min(iter_start + per_worker, self.end)

#         for i in range(iter_start, iter_end):
#             yield np.array([i])

# dataset = SplitedIterableDataset(start=2, end=9)
# dataloader = DataLoader(
#     dataset,
#     num_workers=2,
#     batch_size=1,
#     drop_last=True)

# for data in dataloader:
#     print(data)
    

# class RangeIterableDataset(IterableDataset):
#     def __init__(self, start, end):
#         self.start = start
#         self.end = end

#     def __iter__(self):
#         for i in range(self.start, self.end):
#             yield np.array([i])

# dataset = RangeIterableDataset(start=2, end=9)

# def worker_init_fn(worker_id):
#     worker_info = get_worker_info()

#     dataset = worker_info.dataset
#     start = dataset.start
#     end = dataset.end
#     num_per_worker = int(
#         math.ceil((end - start) / float(worker_info.num_workers)))

#     worker_id = worker_info.id
#     dataset.start = start + worker_id * num_per_worker
#     dataset.end = min(dataset.start + num_per_worker, end)

# dataloader = DataLoader(
#     dataset,
#     num_workers=2,
#     batch_size=1,
#     drop_last=True,
#     worker_init_fn=worker_init_fn)

# for data in dataloader:
#     print(data)

# a_list = paddle.io.random_split(range(10), [3, 7])
# print(len(a_list))

# for idx, v in enumerate(a_list[0]):
#     print(idx, v)

# for idx, v in enumerate(a_list[1]):
#     print(idx, v)

# class RandomDataset(Dataset):
#     def __init__(self, num_samples):
#         self.num_samples = num_samples

#     def __getitem__(self, idx):
#         image = np.random.random([784]).astype('float32')
#         label = np.random.randint(0, 9, (1, )).astype('int64')
#         return image, label

#     def __len__(self):
#         return self.num_samples

# sampler = RandomSampler(data_source=RandomDataset(100))

# for index in sampler:
#     print(index)

# class RandomDataset(Dataset):
#     def __init__(self, num_samples):
#         self.num_samples = num_samples

#     def __getitem__(self, idx):
#         image = np.random.random([784]).astype('float32')
#         label = np.random.randint(0, 9, (1, )).astype('int64')
#         return image, label

#     def __len__(self):
#         return self.num_samples

# class MySampler(Sampler):
#     def __init__(self, data_source):
#         self.data_source = data_source

#     def __iter__(self):
#         return iter(range(len(self.data_source)))

#     def __len__(self):
#         return len(self.data_source)

# sampler = MySampler(data_source=RandomDataset(100))

# for index in sampler:
#     print(index)

# class RandomDataset(Dataset):
#     def __init__(self, num_samples):
#         self.num_samples = num_samples

#     def __getitem__(self, idx):
#         image = np.random.random([784]).astype('float32')
#         label = np.random.randint(0, 9, (1, )).astype('int64')
#         return image, label

#     def __len__(self):
#         return self.num_samples

# sampler = SequenceSampler(data_source=RandomDataset(100))

# for index in sampler:
#     print(index)

# a = paddle.io.Subset(dataset=range(1, 4), indices=[0, 2])
# print(list(a))

# b = paddle.io.Subset(dataset=range(1, 4), indices=[1, 1])
# print(list(b))

# input_np = np.random.random([2, 3, 4]).astype('float32')
# input = paddle.to_tensor(input_np)
# label_np = np.random.random([2, 1]).astype('int32')
# label = paddle.to_tensor(label_np)

# dataset = TensorDataset([input, label])

# for i in range(len(dataset)):
#     input, label = dataset[i]
#     print(input, label)

# sampler = WeightedRandomSampler(weights=[0.1, 0.3, 0.5, 0.7, 0.2],
#                                 num_samples=5,
#                                 replacement=True)

# for index in sampler:
#     print(index)

# BATCH_SIZE = 16
# BATCH_NUM = 4
# EPOCH_NUM = 4

# IMAGE_SIZE = 784
# CLASS_NUM = 10

# class RandomDataset(paddle.io.Dataset):
#     def __init__(self, num_samples):
#         self.num_samples = num_samples

#     def __getitem__(self, idx):
#         image = np.random.random([IMAGE_SIZE]).astype('float32')
#         label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
#         return image, label

#     def __len__(self):
#         return self.num_samples

# class LinearNet(nn.Layer):
#     def __init__(self):
#         super(LinearNet, self).__init__()
#         self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

#     @paddle.jit.to_static
#     def forward(self, x):
#         return self._linear(x)

# def train(layer, loader, loss_fn, opt):
#     for epoch_id in range(EPOCH_NUM):
#         for batch_id, (image, label) in enumerate(loader()):
#             out = layer(image)
#             loss = loss_fn(out, label)
#             loss.backward()
#             opt.step()
#             opt.clear_grad()
#             print("Epoch {} batch {}: loss = {}".format(
#                 epoch_id, batch_id, np.mean(loss.numpy())))

# layer = LinearNet()
# loss_fn = nn.CrossEntropyLoss()
# adam = opt.Adam(learning_rate=0.001, parameters=layer.parameters())

# dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
# loader = paddle.io.DataLoader(dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     drop_last=True,
#     num_workers=2)

# train(layer, loader, loss_fn, adam)

# path = "example_model/linear"
# paddle.jit.save(layer, path)

# loaded_layer = paddle.jit.load(path)

# loaded_layer.eval()
# x = paddle.randn([1, IMAGE_SIZE], 'float32')
# pred = loaded_layer(x)

# loaded_layer.train()
# adam = opt.Adam(learning_rate=0.001, parameters=loaded_layer.parameters())
# train(loaded_layer, loader, loss_fn, adam)

# BATCH_SIZE = 16
# BATCH_NUM = 4
# EPOCH_NUM = 4

# IMAGE_SIZE = 784
# CLASS_NUM = 10

# class RandomDataset(paddle.io.Dataset):
#     def __init__(self, num_samples):
#         self.num_samples = num_samples

#     def __getitem__(self, idx):
#         image = np.random.random([IMAGE_SIZE]).astype('float32')
#         label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
#         return image, label

#     def __len__(self):
#         return self.num_samples

# paddle.enable_static()

# image = static.data(name='image', shape=[None, 784], dtype='float32')
# label = static.data(name='label', shape=[None, 1], dtype='int64')
# pred = static.nn.fc(x=image, size=10, activation='softmax')
# loss = F.cross_entropy(input=pred, label=label)
# avg_loss = paddle.mean(loss)

# optimizer = paddle.optimizer.SGD(learning_rate=0.001)
# optimizer.minimize(avg_loss)

# place = paddle.CPUPlace()
# exe = static.Executor(place)
# exe.run(static.default_startup_program())

# dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
# loader = paddle.io.DataLoader(dataset,
#     feed_list=[image, label],
#     places=place,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     drop_last=True,
#     return_list=False,
#     num_workers=2)

# for data in loader():
#     exe.run(
#         static.default_main_program(),
#         feed=data,
#         fetch_list=[avg_loss])

# model_path = "fc.example.model"
# paddle.fluid.io.save_inference_model(
#     model_path, ["image"], [pred], exe)

# paddle.disable_static(place)

# fc = paddle.jit.load(model_path)

# fc.eval()
# x = paddle.randn([1, IMAGE_SIZE], 'float32')
# pred = fc(x)

# fc.train()
# loss_fn = nn.CrossEntropyLoss()
# adam = opt.Adam(learning_rate=0.001, parameters=fc.parameters())
# loader = paddle.io.DataLoader(dataset,
#     places=place,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     drop_last=True,
#     num_workers=2)
# for epoch_id in range(EPOCH_NUM):
#     for batch_id, (image, label) in enumerate(loader()):
#         out = fc(image)
#         loss = loss_fn(out, label)
#         loss.backward()
#         adam.step()
#         adam.clear_grad()
#         print("Epoch {} batch {}: loss = {}".format(
#             epoch_id, batch_id, np.mean(loss.numpy())))

# @paddle.jit.not_to_static
# def func_not_to_static(x):
#     res = x - 1
#     return res

# @paddle.jit.to_static
# def func(x):
#     if paddle.mean(x) < 0:
#         out = func_not_to_static(x)
#     else:
#         out = x + 1
#     return out

# x = paddle.ones([1, 2], dtype='float32')
# out = func(x)
# print(out) 

# paddle.jit.ProgramTranslator()
# paddle.jit.ProgramTranslator.get_instance()

# @paddle.jit.to_static
# def func(x):
#     if paddle.mean(x) > 0:
#         x_v = x - 1
#     else:
#         x_v = x + 1
#     return x_v

# prog_trans = paddle.jit.ProgramTranslator()
# prog_trans.enable(False)

# x = paddle.ones([1, 2])

# print(func(x))  

# def func(x):
#     if paddle.mean(x) > 0:
#         x_v = x - 1
#     else:
#         x_v = x + 1
#     return x_v

# prog_trans = paddle.jit.ProgramTranslator()

# x = paddle.ones([1, 2])
# x_v = prog_trans.get_output(func, x)
# print(x_v)  

# def func(x):
#     if paddle.mean(x) > 0:
#         x_v = x - 1
#     else:
#         x_v = x + 1
#     return x_v

# prog_trans = paddle.jit.ProgramTranslator()
# static_func = prog_trans.get_func(func)
# print(callable(static_func)) 

# def func(x):
#     if paddle.mean(x) > 0:
#         x_v = x - 1
#     else:
#         x_v = x + 1
#     return x_v

# prog_trans = paddle.jit.ProgramTranslator()
# x = paddle.ones([1, 2])
# main_prog, start_prog, inputs, outputs = prog_trans.get_program(func, x)
# print([i.name for i in inputs])

# print([o.name for o in outputs])

# def func(x):
#     if paddle.mean(x) > 0:
#         x_v = x - 1
#     else:
#         x_v = x + 1
#     return x_v

# prog_trans = paddle.jit.ProgramTranslator()

# code = prog_trans.get_code(func)
# print(type(code)) 

# prog_trans = paddle.jit.ProgramTranslator()
# prog_cache = prog_trans.get_program_cache()

# BATCH_SIZE = 16
# BATCH_NUM = 4
# EPOCH_NUM = 4

# IMAGE_SIZE = 784
# CLASS_NUM = 10

# class RandomDataset(paddle.io.Dataset):
#     def __init__(self, num_samples):
#         self.num_samples = num_samples

#     def __getitem__(self, idx):
#         image = np.random.random([IMAGE_SIZE]).astype('float32')
#         label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
#         return image, label

#     def __len__(self):
#         return self.num_samples

# class LinearNet(nn.Layer):
#     def __init__(self):
#         super(LinearNet, self).__init__()
#         self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

#     @paddle.jit.to_static
#     def forward(self, x):
#         return self._linear(x)

# def train(layer, loader, loss_fn, opt):
#     for epoch_id in range(EPOCH_NUM):
#         for batch_id, (image, label) in enumerate(loader()):
#             out = layer(image)
#             loss = loss_fn(out, label)
#             loss.backward()
#             opt.step()
#             opt.clear_grad()
#             print("Epoch {} batch {}: loss = {}".format(
#                 epoch_id, batch_id, np.mean(loss.numpy())))

# layer = LinearNet()
# loss_fn = nn.CrossEntropyLoss()
# adam = opt.Adam(learning_rate=0.001, parameters=layer.parameters())

# dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
# loader = paddle.io.DataLoader(dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     drop_last=True,
#     num_workers=2)

# train(layer, loader, loss_fn, adam)

# path = "example_model/linear"
# paddle.jit.save(layer, path)

# def save_function():
#     @paddle.jit.to_static
#     def fun(inputs):
#         return paddle.tanh(inputs)

#     path = 'test_jit_save_load_function_1/func'
#     inps = paddle.rand([3, 6])
#     origin = fun(inps)

#     paddle.jit.save(fun, path)
#     load_func = paddle.jit.load(path)

#     load_result = load_func(inps)
#     print((load_result - origin).abs().max() < 1e-10)

# save_function()

# paddle.jit.set_code_level(2)

# os.environ['TRANSLATOR_CODE_LEVEL'] = '3'

# paddle.jit.set_verbosity(1)

# os.environ['TRANSLATOR_VERBOSITY'] = '3'

# @to_static
# def func(x):
#     if paddle.mean(x) < 0:
#         x_v = x - 1
#     else:
#         x_v = x + 1
#     return x_v

# x = paddle.ones([1, 2], dtype='float32')
# x_v = func(x)
# print(x_v) 

# BATCH_SIZE = 16
# BATCH_NUM = 4
# EPOCH_NUM = 4

# IMAGE_SIZE = 784
# CLASS_NUM = 10

# class RandomDataset(paddle.io.Dataset):
#     def __init__(self, num_samples):
#         self.num_samples = num_samples

#     def __getitem__(self, idx):
#         image = np.random.random([IMAGE_SIZE]).astype('float32')
#         label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
#         return image, label

#     def __len__(self):
#         return self.num_samples

# class LinearNet(nn.Layer):
#     def __init__(self):
#         super(LinearNet, self).__init__()
#         self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

#     @paddle.jit.to_static
#     def forward(self, x):
#         return self._linear(x)

# def train(layer, loader, loss_fn, opt):
#     for epoch_id in range(EPOCH_NUM):
#         for batch_id, (image, label) in enumerate(loader()):
#             out = layer(image)
#             loss = loss_fn(out, label)
#             loss.backward()
#             opt.step()
#             opt.clear_grad()
#             print("Epoch {} batch {}: loss = {}".format(
#                 epoch_id, batch_id, np.mean(loss.numpy())))

# layer = LinearNet()
# loss_fn = nn.CrossEntropyLoss()
# adam = opt.Adam(learning_rate=0.001, parameters=layer.parameters())

# dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
# loader = paddle.io.DataLoader(dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     drop_last=True,
#     num_workers=2)

# train(layer, loader, loss_fn, adam)

# model_path = "linear.example.model"
# paddle.jit.save(layer, model_path)

# translated_layer = paddle.jit.load(model_path)

# translated_layer.eval()
# x = paddle.randn([1, IMAGE_SIZE], 'float32')
# pred = translated_layer(x)

# translated_layer.train()
# adam = opt.Adam(learning_rate=0.001, parameters=translated_layer.parameters())
# train(translated_layer, loader, loss_fn, adam)

# class MyLayer(paddle.nn.Layer):
#     def __init__(self):
#         super(MyLayer, self).__init__()
#         self._linear = paddle.nn.Linear(1, 1)
#         self._dropout = paddle.nn.Dropout(p=0.5)

#     def forward(self, input):
#         temp = self._linear(input)
#         temp = self._dropout(temp)
#         return temp

# x = paddle.randn([10, 1], 'float32')
# mylayer = MyLayer()
# mylayer.eval()  
# out = mylayer(x)
# mylayer.train()  
# out = mylayer(x)

# class MyLayer(paddle.nn.Layer):
#     def __init__(self):
#         super(MyLayer, self).__init__()
#         self._linear = paddle.nn.Linear(1, 1)
#         self._dropout = paddle.nn.Dropout(p=0.5)

#     def forward(self, input):
#         temp = self._linear(input)
#         temp = self._dropout(temp)
#         return temp

# x = paddle.randn([10, 1], 'float32')
# mylayer = MyLayer()
# mylayer.eval()  
# out = mylayer(x)
# print(out)

# BATCH_SIZE = 16
# BATCH_NUM = 4
# EPOCH_NUM = 4

# IMAGE_SIZE = 784
# CLASS_NUM = 10

# class RandomDataset(paddle.io.Dataset):
#     def __init__(self, num_samples):
#         self.num_samples = num_samples

#     def __getitem__(self, idx):
#         image = np.random.random([IMAGE_SIZE]).astype('float32')
#         label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
#         return image, label

#     def __len__(self):
#         return self.num_samples

# class LinearNet(nn.Layer):
#     def __init__(self):
#         super(LinearNet, self).__init__()
#         self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

#     @paddle.jit.to_static
#     def forward(self, x):
#         return self._linear(x)

# def train(layer, loader, loss_fn, opt):
#     for epoch_id in range(EPOCH_NUM):
#         for batch_id, (image, label) in enumerate(loader()):
#             out = layer(image)
#             loss = loss_fn(out, label)
#             loss.backward()
#             opt.step()
#             opt.clear_grad()
#             print("Epoch {} batch {}: loss = {}".format(
#                 epoch_id, batch_id, np.mean(loss.numpy())))

# layer = LinearNet()
# loss_fn = nn.CrossEntropyLoss()
# adam = opt.Adam(learning_rate=0.001, parameters=layer.parameters())

# dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
# loader = paddle.io.DataLoader(dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     drop_last=True,
#     num_workers=2)

# train(layer, loader, loss_fn, adam)

# model_path = "linear.example.model"
# paddle.jit.save(layer, model_path)

# translated_layer = paddle.jit.load(model_path)

# program = translated_layer.program()

# class MyLayer(paddle.nn.Layer):
#     def __init__(self):
#         super(MyLayer, self).__init__()
#         self._linear = paddle.nn.Linear(1, 1)
#         w_tmp = self.create_parameter([1,1])
#         self.add_parameter("w_tmp", w_tmp)

#     def forward(self, input):
#         return self._linear(input)

# mylayer = MyLayer()
# for name, param in mylayer.named_parameters():
#     print(name, param)      

# class MySequential(paddle.nn.Layer):
#     def __init__(self, *layers):
#         super(MySequential, self).__init__()
#         if len(layers) > 0 and isinstance(layers[0], tuple):
#             for name, layer in layers:
#                 self.add_sublayer(name, layer)
#         else:
#             for idx, layer in enumerate(layers):
#                 self.add_sublayer(str(idx), layer)

#     def forward(self, input):
#         for layer in self._sub_layers.values():
#             input = layer(input)
#         return input

# fc1 = paddle.nn.Linear(10, 3)
# fc2 = paddle.nn.Linear(3, 10, bias_attr=False)
# model = MySequential(fc1, fc2)
# for prefix, layer in model.named_sublayers():
#     print(prefix, layer)

# net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))

# def init_weights(layer):
#     if type(layer) == nn.Linear:
#         print('before init weight:', layer.weight.numpy())
#         new_weight = paddle.full(shape=layer.weight.shape, dtype=layer.weight.dtype, fill_value=0.9)
#         layer.weight.set_value(new_weight)
#         print('after init weight:', layer.weight.numpy())

# net.apply(init_weights)

# print(net.state_dict())

# linear = paddle.nn.Linear(10, 3)
# value = np.array([0]).astype("float32")
# buffer = paddle.to_tensor(value)
# linear.register_buffer("buf_name", buffer, persistable=True)

# print(linear.buffers())     

# linear1 = paddle.nn.Linear(10, 3)
# linear2 = paddle.nn.Linear(3, 10, bias_attr=False)
# model = paddle.nn.Sequential(linear1, linear2)

# layer_list = list(model.children())

# print(layer_list)   

# value = np.arange(26).reshape(2, 13).astype("float32")
# a = paddle.to_tensor(value)
# linear = paddle.nn.Linear(13, 5)
# adam = paddle.optimizer.Adam(learning_rate=0.01,
#                             parameters=linear.parameters())
# out = linear(a)
# out.backward()
# adam.step()
# linear.clear_gradients()

# class MyLayer(paddle.nn.Layer):
#     def __init__(self):
#         super(MyLayer, self).__init__()
#         self._linear = paddle.nn.Linear(1, 1)
#         w_tmp = self.create_parameter([1,1])
#         self.add_parameter("w_tmp", w_tmp)

#     def forward(self, input):
#         return self._linear(input)

# mylayer = MyLayer()
# for name, param in mylayer.named_parameters():
#     print(name, param)      

# class MyLinear(paddle.nn.Layer):
#     def __init__(self,
#                 in_features,
#                 out_features):
#         super(MyLinear, self).__init__()
#         self.linear = paddle.nn.Linear( 10, 10)

#         self.back_var = self.create_tensor(name = "linear_tmp_0", dtype=self._dtype)

#     def forward(self, input):
#         out = self.linear(input)
#         paddle.assign( out, self.back_var)

#         return out

# class MyLinear(paddle.nn.Layer):
#     def __init__(self,
#                 in_features,
#                 out_features):
#         super(MyLinear, self).__init__()
#         self.linear = paddle.nn.Linear( 10, 10)

#         self.back_var = self.create_variable(name = "linear_tmp_0", dtype=self._dtype)

#     def forward(self, input):
#         out = self.linear(input)
#         paddle.assign( out, self.back_var)

#         return out

# class LinearNet(paddle.nn.Layer):
#     def __init__(self):
#         super(LinearNet, self).__init__(name_scope = "demo_linear_net")
#         self._linear = paddle.nn.Linear(1, 1)

#     def forward(self, x):
#         return self._linear(x)

# linear_net = LinearNet()
# print(linear_net.full_name())   

# emb = paddle.nn.Embedding(10, 10)

# state_dict = emb.state_dict()
# paddle.save(state_dict, "paddle_dy.pdparams")
# para_state_dict = paddle.load("paddle_dy.pdparams")
# emb.set_state_dict(para_state_dict)

# fc1 = paddle.nn.Linear(10, 3)
# buffer1 = paddle.to_tensor(np.array([0]).astype("float32"))

# fc1.register_buffer("buf_name_1", buffer1, persistable=True)

# fc2 = paddle.nn.Linear(3, 10)
# buffer2 = paddle.to_tensor(np.array([1]).astype("float32"))

# fc2.buf_name_2 = buffer2

# model = paddle.nn.Sequential(fc1, fc2)

# for name, buffer in model.named_buffers():
#     print(name, buffer)

# linear1 = paddle.nn.Linear(10, 3)
# linear2 = paddle.nn.Linear(3, 10, bias_attr=False)
# model = paddle.nn.Sequential(linear1, linear2)
# for prefix, layer in model.named_children():
#     print(prefix, layer)
    
    

# fc1 = paddle.nn.Linear(10, 3)
# fc2 = paddle.nn.Linear(3, 10, bias_attr=False)
# model = paddle.nn.Sequential(fc1, fc2)
# for name, param in model.named_parameters():
#     print(name, param)

# fc1 = paddle.nn.Linear(10, 3)
# fc2 = paddle.nn.Linear(3, 10, bias_attr=False)
# model = paddle.nn.Sequential(fc1, fc2)
# for prefix, layer in model.named_sublayers():
#     print(prefix, layer)

# linear = paddle.nn.Linear(10, 3)
# value = np.array([0]).astype("float32")
# buffer = paddle.to_tensor(value)
# linear.register_buffer("buf_name", buffer, persistable=True)

# print(linear.buf_name)

# def forward_post_hook(layer, input, output):
    

    
#     return output * 2

# linear = paddle.nn.Linear(13, 5)

# forward_post_hook_handle = linear.register_forward_post_hook(forward_post_hook)

# value1 = np.arange(26).reshape(2, 13).astype("float32")
# in1 = paddle.to_tensor(value1)

# out0 = linear(in1)

# forward_post_hook_handle.remove()

# out1 = linear(in1)

# assert (out0.numpy() == (out1.numpy()) * 2).any()

# def forward_pre_hook(layer, input):
    

    
#     input_return = (input[0] * 2)
#     return input_return

# linear = paddle.nn.Linear(13, 5)

# forward_pre_hook_handle = linear.register_forward_pre_hook(forward_pre_hook)

# value0 = np.arange(26).reshape(2, 13).astype("float32")
# in0 = paddle.to_tensor(value0)
# out0 = linear(in0)

# forward_pre_hook_handle.remove()

# value1 = value0 * 2
# in1 = paddle.to_tensor(value1)
# out1 = linear(in1)

# assert (out0.numpy() == out1.numpy()).any()

# emb = paddle.nn.Embedding(10, 10)

# state_dict = emb.state_dict()
# paddle.save(state_dict, "paddle_dy.pdparams")
# para_state_dict = paddle.load("paddle_dy.pdparams")
# emb.set_state_dict(para_state_dict)

# emb = paddle.nn.Embedding(10, 10)

# state_dict = emb.state_dict()
# paddle.save(state_dict, "paddle_dy.pdparams")
# para_state_dict = paddle.load("paddle_dy.pdparams")
# emb.set_state_dict(para_state_dict)

# emb = paddle.nn.Embedding(10, 10)

# state_dict = emb.state_dict()
# paddle.save( state_dict, "paddle_dy.pdparams")

# class MyLayer(paddle.nn.Layer):
#     def __init__(self):
#         super(MyLayer, self).__init__()
#         self._linear = paddle.nn.Linear(1, 1)
#         self._dropout = paddle.nn.Dropout(p=0.5)

#     def forward(self, input):
#         temp = self._linear(input)
#         temp = self._dropout(temp)
#         return temp

# mylayer = MyLayer()
# print(mylayer.sublayers())  

# linear=paddle.nn.Linear(2, 2)
# linear.weight

# linear.to(dtype='float64')
# linear.weight

# linear.to(device='cpu')
# linear.weight

# linear.to(device=paddle.CUDAPinnedPlace(), blocking=False)
# linear.weight

# emb = paddle.nn.Embedding(10, 10)

# state_dict = emb.to_static_state_dict()
# paddle.save( state_dict, "paddle_dy.pdparams")

# x = paddle.to_tensor([[1., 0, -1], [0, 1, 0], [1, 0, 1]])

# out = paddle.linalg.cond(x)

# out_fro = paddle.linalg.cond(x, p='fro')

# out_nuc = paddle.linalg.cond(x, p='nuc')

# out_1 = paddle.linalg.cond(x, p=1)

# out_minus_1 = paddle.linalg.cond(x, p=-1)

# out_2 = paddle.linalg.cond(x, p=2)

# out_minus_2 = paddle.linalg.cond(x, p=-2)

# out_inf = paddle.linalg.cond(x, p=np.inf)

# out_minus_inf = paddle.linalg.cond(x, p=-np.inf)

# a = paddle.to_tensor(np.random.randn(2, 4, 4).astype('float32'))

# a_cond_fro = paddle.linalg.cond(a, p='fro')

# b = paddle.to_tensor(np.random.randn(2, 3, 4).astype('float64'))

# b_cond_2 = paddle.linalg.cond(b, p=2)

# paddle.device.set_device("cpu")

# x_data = np.array([[1.6707249, 7.2249975, 6.5045543],
#                    [9.956216,  8.749598,  6.066444 ],
#                    [4.4251957, 1.7983172, 0.370647 ]]).astype("float32")
# x = paddle.to_tensor(x_data)
# w, v = paddle.linalg.eig(x)
# print(w)

# print(v)

# x_data = np.array([[1, -2j], [2j, 5]])
# x = paddle.to_tensor(x_data)
# out_value, out_vector = paddle.linalg.eigh(x, UPLO='L')
# print(out_value)

# print(out_vector)

# paddle.set_device("cpu")
# paddle.seed(1234)

# x = paddle.rand(shape=[3, 3], dtype='float64')

# print(paddle.linalg.eigvals(x))

# x_data = np.array([[1, -2j], [2j, 5]])
# x = paddle.to_tensor(x_data)
# out_value = paddle.eigvalsh(x, UPLO='L')
# print(out_value)

# x = paddle.to_tensor([[1, 2, 3],
#                       [1, 4, 9],
#                       [1, 8, 27]], dtype='float64')
# print(paddle.linalg.matrix_power(x, 2))

# print(paddle.linalg.matrix_power(x, 0))

# print(paddle.linalg.matrix_power(x, -2))

# a = paddle.eye(10)
# b = paddle.linalg.matrix_rank(a)
# print(b)

# c = paddle.ones(shape=[3, 4, 5, 5])
# d = paddle.linalg.matrix_rank(c, tol=0.01, hermitian=True)
# print(d)

# A_data = np.random.random([3, 4]).astype(np.float32)
# B_data = np.random.random([4, 5]).astype(np.float32)
# A = paddle.to_tensor(A_data)
# B = paddle.to_tensor(B_data)
# out = paddle.linalg.multi_dot([A, B])
# print(out.numpy().shape)

# A_data = np.random.random([10, 5]).astype(np.float32)
# B_data = np.random.random([5, 8]).astype(np.float32)
# C_data = np.random.random([8, 7]).astype(np.float32)
# A = paddle.to_tensor(A_data)
# B = paddle.to_tensor(B_data)
# C = paddle.to_tensor(C_data)
# out = paddle.linalg.multi_dot([A, B, C])
# print(out.numpy().shape)

# x = paddle.arange(15).reshape((3, 5)).astype('float64')
# input = paddle.to_tensor(x)
# out = paddle.linalg.pinv(input)
# print(input)
# print(out)

# x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).astype('float64')
# q, r = paddle.linalg.qr(x)
# print (q)
# print (r)

# x = paddle.to_tensor([[1.0, 2.0], [1.0, 3.0], [4.0, 6.0]]).astype('float64')
# x = x.reshape([3, 2])
# u, s, vh = paddle.linalg.svd(x)
# print (u)

# print (s)

# print (vh)

# x = paddle.to_tensor(np.array([
#     [0.1, 0.2, 0.3, 0.4],
#     [0.1, 0.4, 0.3, 0.2],
#     [0.1, 0.2, 0.4, 0.3],
#     [0.1, 0.2, 0.3, 0.4]]))
# y = paddle.to_tensor(np.array([[0], [1], [2], [3]]))

# m = paddle.metric.Accuracy()
# correct = m.compute(x, y)
# m.update(correct)
# res = m.accumulate()
# print(res) 

# input = InputSpec([None, 1, 28, 28], 'float32', 'image')
# label = InputSpec([None, 1], 'int64', 'label')
# transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
# train_dataset = MNIST(mode='train', transform=transform)

# model = paddle.Model(paddle.vision.models.LeNet(), input, label)
# optim = paddle.optimizer.Adam(
#     learning_rate=0.001, parameters=model.parameters())
# model.prepare(
#     optim,
#     loss=paddle.nn.CrossEntropyLoss(),
#     metrics=paddle.metric.Accuracy())

# model.fit(train_dataset, batch_size=64)

# predictions = paddle.to_tensor([[0.2, 0.1, 0.4, 0.1, 0.1], [0.2, 0.3, 0.1, 0.15, 0.25]], dtype='float32')
# label = paddle.to_tensor([[2], [0]], dtype="int64")
# result = paddle.metric.accuracy(input=predictions, label=label, k=1)

# m = paddle.metric.Auc()

# n = 8
# class0_preds = np.random.random(size = (n, 1))
# class1_preds = 1 - class0_preds

# preds = np.concatenate((class0_preds, class1_preds), axis=1)
# labels = np.random.randint(2, size = (n, 1))

# m.update(preds=preds, labels=labels)
# res = m.accumulate()

# class Data(paddle.io.Dataset):
#     def __init__(self):
#         super(Data, self).__init__()
#         self.n = 1024
#         self.x = np.random.randn(self.n, 10).astype('float32')
#         self.y = np.random.randint(2, size=(self.n, 1)).astype('int64')

#     def __getitem__(self, idx):
#         return self.x[idx], self.y[idx]

#     def __len__(self):
#         return self.n

# model = paddle.Model(nn.Sequential(
#     nn.Linear(10, 2), nn.Softmax())
# )
# optim = paddle.optimizer.Adam(
#     learning_rate=0.001, parameters=model.parameters())

# def loss(x, y):
#     return nn.functional.nll_loss(paddle.log(x), y)

# model.prepare(
#     optim,
#     loss=loss,
#     metrics=paddle.metric.Auc())
# data = Data()
# model.fit(data, batch_size=16)

# x = np.array([0.1, 0.5, 0.6, 0.7])
# y = np.array([0, 1, 1, 1])

# m = paddle.metric.Precision()
# m.update(x, y)
# res = m.accumulate()
# print(res) 

# class Data(paddle.io.Dataset):
#     def __init__(self):
#         super(Data, self).__init__()
#         self.n = 1024
#         self.x = np.random.randn(self.n, 10).astype('float32')
#         self.y = np.random.randint(2, size=(self.n, 1)).astype('float32')

#     def __getitem__(self, idx):
#         return self.x[idx], self.y[idx]

#     def __len__(self):
#         return self.n

# model = paddle.Model(nn.Sequential(
#     nn.Linear(10, 1),
#     nn.Sigmoid()
# ))
# optim = paddle.optimizer.Adam(
#     learning_rate=0.001, parameters=model.parameters())
# model.prepare(
#     optim,
#     loss=nn.BCELoss(),
#     metrics=paddle.metric.Precision())

# data = Data()
# model.fit(data, batch_size=16)

# x = np.array([0.1, 0.5, 0.6, 0.7])
# y = np.array([1, 0, 1, 1])

# m = paddle.metric.Recall()
# m.update(x, y)
# res = m.accumulate()
# print(res) 

# class Data(paddle.io.Dataset):
#     def __init__(self):
#         super(Data, self).__init__()
#         self.n = 1024
#         self.x = np.random.randn(self.n, 10).astype('float32')
#         self.y = np.random.randint(2, size=(self.n, 1)).astype('float32')

#     def __getitem__(self, idx):
#         return self.x[idx], self.y[idx]

#     def __len__(self):
#         return self.n

# model = paddle.Model(nn.Sequential(
#     nn.Linear(10, 1),
#     nn.Sigmoid()
# ))
# optim = paddle.optimizer.Adam(
#     learning_rate=0.001, parameters=model.parameters())
# model.prepare(
#     optim,
#     loss=nn.BCELoss(),
#     metrics=[paddle.metric.Precision(), paddle.metric.Recall()])

# data = Data()
# model.fit(data, batch_size=16)

# data = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32]).astype(np.float32))
# AdaptiveAvgPool1D = nn.AdaptiveAvgPool1D(output_size=16)
# pool_out = AdaptiveAvgPool1D(data)

# input_data = np.random.rand(2, 3, 32, 32)
# x = paddle.to_tensor(input_data)

# adaptive_avg_pool = paddle.nn.AdaptiveAvgPool2D(output_size=3)
# pool_out = adaptive_avg_pool(x = x)

# input_data = np.random.rand(2, 3, 8, 32, 32)
# x = paddle.to_tensor(input_data)

# adaptive_avg_pool = paddle.nn.AdaptiveAvgPool3D(output_size=3)
# pool_out = adaptive_avg_pool(x = x)

# data = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32]).astype(np.float32))
# AdaptiveMaxPool1D = nn.AdaptiveMaxPool1D(output_size=16)
# pool_out = AdaptiveMaxPool1D(data)

# AdaptiveMaxPool1D = nn.AdaptiveMaxPool1D(output_size=16, return_mask=True)
# pool_out, indices = AdaptiveMaxPool1D(data)

# input_data = np.random.rand(2, 3, 32, 32)
# x = paddle.to_tensor(input_data)
# adaptive_max_pool = paddle.nn.AdaptiveMaxPool2D(output_size=3, return_mask=True)
# pool_out, indices = adaptive_max_pool(x = x)

# input_data = np.random.rand(2, 3, 8, 32, 32)
# x = paddle.to_tensor(input_data)
# pool = paddle.nn.AdaptiveMaxPool3D(output_size=4)
# out = pool(x)

# pool = paddle.nn.AdaptiveMaxPool3D(output_size=3, return_mask=True)
# out, indices = pool(x)

# x = np.array([[-1, 1], [-1, 1]]).astype('float32')
# x = paddle.to_tensor(x)
# m = paddle.nn.AlphaDropout(p=0.5)
# y_train = m(x)
# m.eval()  
# y_test = m(x)
# print(x)
# print(y_train)

# print(y_test)

# data = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32]).astype(np.float32))
# AvgPool1D = nn.AvgPool1D(kernel_size=2, stride=2, padding=0)
# pool_out = AvgPool1D(data)

# input = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32, 32]).astype(np.float32))
# AvgPool2D = nn.AvgPool2D(kernel_size=2,
#                     stride=2, padding=0)
# output = AvgPool2D(input)

# input = paddle.to_tensor(np.random.uniform(-1, 1, [1, 2, 3, 32, 32]).astype(np.float32))
# AvgPool3D = nn.AvgPool3D(kernel_size=2,
#                        stride=2, padding=0)
# output = AvgPool3D(input)

# x = np.random.random(size=(3, 10, 3, 7)).astype('float32')
# with fluid.dygraph.guard():
#     x = to_variable(x)
#     batch_norm = fluid.BatchNorm(10)
#     hidden1 = batch_norm(x)

# np.random.seed(123)
# x_data = np.random.random(size=(2, 1, 3)).astype('float32')
# x = paddle.to_tensor(x_data)
# batch_norm = paddle.nn.BatchNorm1D(1)
# batch_norm_out = batch_norm(x)

# print(batch_norm_out)

# np.random.seed(123)
# x_data = np.random.random(size=(2, 1, 2, 3)).astype('float32')
# x = paddle.to_tensor(x_data)
# batch_norm = paddle.nn.BatchNorm2D(1)
# batch_norm_out = batch_norm(x)

# print(batch_norm_out)

# np.random.seed(123)
# x_data = np.random.random(size=(2, 1, 2, 2, 3)).astype('float32')
# x = paddle.to_tensor(x_data)
# batch_norm = paddle.nn.BatchNorm3D(1)
# batch_norm_out = batch_norm(x)

# print(batch_norm_out)

# input_data = np.array([0.5, 0.6, 0.7]).astype("float32")
# label_data = np.array([1.0, 0.0, 1.0]).astype("float32")

# input = paddle.to_tensor(input_data)
# label = paddle.to_tensor(label_data)
# bce_loss = paddle.nn.BCELoss()
# output = bce_loss(input, label)
# print(output)  

# logit = paddle.to_tensor([5.0, 1.0, 3.0], dtype="float32")
# label = paddle.to_tensor([1.0, 0.0, 1.0], dtype="float32")
# bce_logit_loss = paddle.nn.BCEWithLogitsLoss()
# output = bce_logit_loss(logit, label)
# print(output.numpy())  

# trg_embeder = Embedding(100, 32)
# output_layer = Linear(32, 32)
# decoder_cell = GRUCell(input_size=32, hidden_size=32)
# decoder = BeamSearchDecoder(decoder_cell,
#                             start_token=0,
#                             end_token=1,
#                             beam_size=4,
#                             embedding_fn=trg_embeder,
#                             output_fn=output_layer)

# layer1 = numpy.random.random((5, 5)).astype('float32')
# layer2 = numpy.random.random((5, 4)).astype('float32')
# bilinear = paddle.nn.Bilinear(
#     in1_features=5, in2_features=4, out_features=1000)
# result = bilinear(paddle.to_tensor(layer1),
#                 paddle.to_tensor(layer2))     

# cell_fw = paddle.nn.LSTMCell(16, 32)
# cell_bw = paddle.nn.LSTMCell(16, 32)
# rnn = paddle.nn.BiRNN(cell_fw, cell_bw)

# inputs = paddle.rand((2, 23, 16))
# outputs, final_states = rnn(inputs)

# print(outputs.shape)
# print(final_states[0][0].shape,len(final_states),len(final_states[0]))

# x = paddle.uniform([10, 10], min=-1.0, max=1.0, dtype='float32')
# linear = paddle.nn.Linear(in_features=10, out_features=10,
#                           weight_attr=paddle.ParamAttr(need_clip=True),
#                           bias_attr=paddle.ParamAttr(need_clip=False))
# out = linear(x)
# loss = paddle.mean(out)
# loss.backward()

# clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
# sdg = paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters(), grad_clip=clip)
# sdg.step()

# x = paddle.uniform([10, 10], min=-1.0, max=1.0, dtype='float32')
# linear = paddle.nn.Linear(in_features=10, out_features=10,
#                           weight_attr=paddle.ParamAttr(need_clip=True),
#                           bias_attr=paddle.ParamAttr(need_clip=False))
# out = linear(x)
# loss = paddle.mean(out)
# loss.backward()

# clip = paddle.nn.ClipGradByNorm(clip_norm=1.0)
# sdg = paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters(), grad_clip=clip)
# sdg.step()

# x = paddle.uniform([10, 10], min=-1.0, max=1.0, dtype='float32')
# linear = paddle.nn.Linear(in_features=10, out_features=10,
#                           weight_attr=paddle.ParamAttr(need_clip=True),
#                           bias_attr=paddle.ParamAttr(need_clip=False))
# out = linear(x)
# loss = paddle.mean(out)
# loss.backward()

# clip = paddle.nn.ClipGradByValue(min=-1, max=1)
# sdg = paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters(), grad_clip=clip)
# sdg.step()

# x = np.array([[[4, 8, 1, 9],
#   [7, 2, 0, 9],
#   [6, 9, 2, 6]]]).astype(np.float32)
# w=np.array(
# [[[9, 3, 4],
#   [0, 0, 7],
#   [2, 5, 6]],
#  [[0, 3, 4],
#   [2, 9, 7],
#   [5, 6, 8]]]).astype(np.float32)
# x_t = paddle.to_tensor(x)
# conv = Conv1D(3, 2, 3)
# conv.weight.set_value(w)
# y_t = conv(x_t)
# print(y_t)

# x=np.array([[[4, 0, 9, 7],
#              [8, 0, 9, 2]]]).astype(np.float32)

# y=np.array([[[7, 0]],
#             [[4, 2]]]).astype(np.float32)
# x_t = paddle.to_tensor(x)
# conv = Conv1DTranspose(2, 1, 2)
# conv.weight.set_value(y)
# y_t = conv(x_t)
# print(y_t)

# paddle.disable_static()

# x_var = paddle.uniform((2, 4, 8, 8), dtype='float32', min=-1., max=1.)

# conv = nn.Conv2D(4, 6, (3, 3))
# y_var = conv(x_var)
# y_np = y_var.numpy()
# print(y_np.shape)

# paddle.disable_static()

# x_var = paddle.uniform((2, 4, 8, 8), dtype='float32', min=-1., max=1.)

# conv = nn.Conv2DTranspose(4, 6, (3, 3))
# y_var = conv(x_var)
# y_np = y_var.numpy()
# print(y_np.shape)

# paddle.disable_static()

# x_var = paddle.uniform((2, 4, 8, 8, 8), dtype='float32', min=-1., max=1.)

# conv = nn.Conv3D(4, 6, (3, 3, 3))
# y_var = conv(x_var)
# y_np = y_var.numpy()
# print(y_np.shape)

# paddle.disable_static()

# x_var = paddle.uniform((2, 4, 8, 8, 8), dtype='float32', min=-1., max=1.)

# conv = nn.Conv3DTranspose(4, 6, (3, 3, 3))
# y_var = conv(x_var)
# y_np = y_var.numpy()
# print(y_np.shape)

# np.random.seed(0)
# x1 = np.random.rand(2,3)
# x2 = np.random.rand(2,3)
# x1 = paddle.to_tensor(x1)
# x2 = paddle.to_tensor(x2)

# cos_sim_func = nn.CosineSimilarity(axis=0)
# result = cos_sim_func(x1, x2)
# print(result)

# paddle.seed(99999)
# N=100
# C=200
# reduction='mean'
# input =  paddle.rand([N, C], dtype='float64')
# label =  paddle.randint(0, C, shape=[N], dtype='int64')
# weight = paddle.rand([C], dtype='float64')

# cross_entropy_loss = paddle.nn.loss.CrossEntropyLoss(
#     weight=weight, reduction=reduction)
# dy_ret = cross_entropy_loss(
#                            input,
#                            label)
# print(dy_ret.numpy()) 

# paddle.seed(99999)
# axis = -1
# ignore_index = -100
# N = 4
# C = 3
# shape = [N, C]
# reduction='mean'
# weight = None
# logits = paddle.uniform(shape, dtype='float64', min=0.1, max=1.0)
# labels = paddle.uniform(shape, dtype='float64', min=0.1, max=1.0)
# labels /= paddle.sum(labels, axis=axis, keepdim=True)
# paddle_loss_mean = paddle.nn.functional.cross_entropy(
#                                                       logits,
#                                                       labels,
#                                                       soft_label=True,
#                                                       axis=axis,
#                                                       weight=weight,
#                                                       reduction=reduction)
# print(paddle_loss_mean.numpy()) 

# max_seq_length = 4

# max_label_length = 3

# batch_size = 2

# class_num = 3

# np.random.seed(1)
# log_probs = np.array([[[4.17021990e-01, 7.20324516e-01, 1.14374816e-04],
#                         [3.02332580e-01, 1.46755889e-01, 9.23385918e-02]],

#                         [[1.86260208e-01, 3.45560730e-01, 3.96767467e-01],
#                         [5.38816750e-01, 4.19194520e-01, 6.85219526e-01]],

#                         [[2.04452246e-01, 8.78117442e-01, 2.73875929e-02],
#                         [6.70467496e-01, 4.17304814e-01, 5.58689833e-01]],

#                         [[1.40386939e-01, 1.98101491e-01, 8.00744593e-01],
#                         [9.68261600e-01, 3.13424170e-01, 6.92322612e-01]],

#                         [[8.76389146e-01, 8.94606650e-01, 8.50442126e-02],
#                         [3.90547849e-02, 1.69830427e-01, 8.78142476e-01]]]).astype("float32")
# labels = np.array([[1, 2, 2],
#                 [1, 2, 2]]).astype("int32")
# input_lengths = np.array([5, 5]).astype("int64")
# label_lengths = np.array([3, 3]).astype("int64")

# log_probs = paddle.to_tensor(log_probs)
# labels = paddle.to_tensor(labels)
# input_lengths = paddle.to_tensor(input_lengths)
# label_lengths = paddle.to_tensor(label_lengths)

# loss = paddle.nn.CTCLoss(blank=0, reduction='none')(log_probs, labels,
#     input_lengths,
#     label_lengths)
# print(loss)  

# loss = paddle.nn.CTCLoss(blank=0, reduction='mean')(log_probs, labels,
#     input_lengths,
#     label_lengths)
# print(loss)  

# x = np.array([[1,2,3], [4,5,6]]).astype('float32')
# x = paddle.to_tensor(x)
# m = paddle.nn.Dropout(p=0.5)
# y_train = m(x)
# m.eval()  
# y_test = m(x)
# print(x)
# print(y_train)
# print(y_test)

# x = np.random.random(size=(2, 3, 4, 5)).astype('float32')
# x = paddle.to_tensor(x)
# m = paddle.nn.Dropout2D(p=0.5)
# y_train = m(x)
# m.eval()  
# y_test = m(x)
# print(x)
# print(y_train)
# print(y_test)

# x = np.random.random(size=(2, 3, 4, 5, 6)).astype('float32')
# x = paddle.to_tensor(x)
# m = paddle.nn.Dropout3D(p=0.5)
# y_train = m(x)
# m.eval()  
# y_test = m(x)
# print(x)
# print(y_train)
# print(y_test)

# trg_embeder = Embedding(100, 32)
# output_layer = Linear(32, 32)
# decoder_cell = GRUCell(input_size=32, hidden_size=32)
# decoder = BeamSearchDecoder(decoder_cell,
#                             start_token=0,
#                             end_token=1,
#                             beam_size=4,
#                             embedding_fn=trg_embeder,
#                             output_fn=output_layer)
# encoder_output = paddle.ones((4, 8, 32), dtype=paddle.get_default_dtype())
# outputs = dynamic_decode(decoder=decoder,
#                         inits=decoder_cell.get_initial_states(encoder_output),
#                         max_step_num=10)

# x = paddle.to_tensor([[-1. ,6.], [1., 15.6]])
# m = paddle.nn.ELU(0.2)
# out = m(x)

# x_data = np.arange(3, 6).reshape((3, 1)).astype(np.int64)
# y_data = np.arange(6, 12).reshape((3, 2)).astype(np.float32)

# x = paddle.to_tensor(x_data, stop_gradient=False)
# y = paddle.to_tensor(y_data, stop_gradient=False)

# embedding = paddle.nn.Embedding(10, 3, sparse=True)

# w0=np.full(shape=(10, 3), fill_value=2).astype(np.float32)
# embedding.weight.set_value(w0)

# adam = paddle.optimizer.Adam(parameters=[embedding.weight], learning_rate=0.01)
# adam.clear_grad()

# out=embedding(x)
# out.backward()
# adam.step()

# inp_np = np.ones([5, 2, 3, 4]).astype('float32')
# inp_np = paddle.to_tensor(inp_np)
# flatten = paddle.nn.Flatten(start_axis=1, stop_axis=2)
# flatten_res = flatten(inp_np)

# x = paddle.to_tensor(np.array([[-1, 0.5],[1, 1.5]]))

# m = paddle.nn.GELU()
# out = m(x) 

# m = paddle.nn.GELU(True)
# out = m(x) 

# paddle.disable_static()
# np.random.seed(123)
# x_data = np.random.random(size=(2, 6, 2, 2)).astype('float32')
# x = paddle.to_tensor(x_data)
# group_norm = paddle.nn.GroupNorm(num_channels=6, num_groups=6)
# group_norm_out = group_norm(x)

# print(group_norm_out.numpy())

# rnn = paddle.nn.GRU(16, 32, 2)

# x = paddle.randn((4, 23, 16))
# prev_h = paddle.randn((2, 4, 32))
# y, h = rnn(x, prev_h)

# print(y.shape)
# print(h.shape)

# x = paddle.randn((4, 16))
# prev_h = paddle.randn((4, 32))

# cell = paddle.nn.GRUCell(16, 32)
# y, h = cell(x, prev_h)

# print(y.shape)
# print(h.shape)

# x = paddle.to_tensor([-1, 0.3, 2.5])
# m = paddle.nn.Hardshrink()
# out = m(x) 

# m = paddle.nn.Hardsigmoid()
# x = paddle.to_tensor([-4., 5., 1.])
# out = m(x) 

# x = paddle.to_tensor([-4., 5., 1.])
# m = paddle.nn.Hardswish()
# out = m(x) 

# x = paddle.to_tensor([-1.5, 0.3, 2.5])
# m = paddle.nn.Hardtanh()
# out = m(x) 

# paddle.set_device('cpu')

# input = paddle.uniform([2, 3])

# label = paddle.to_tensor([0, 1, 4, 5])
# m = paddle.nn.HSigmoidLoss(3, 5)
# out = m(input, label)

# np.random.seed(123)
# x_data = np.random.random(size=(2, 2, 3)).astype('float32')
# x = paddle.to_tensor(x_data)
# instance_norm = paddle.nn.InstanceNorm1D(2)
# instance_norm_out = instance_norm(x)

# print(instance_norm_out)

# np.random.seed(123)
# x_data = np.random.random(size=(2, 2, 2, 3)).astype('float32')
# x = paddle.to_tensor(x_data)
# instance_norm = paddle.nn.InstanceNorm2D(2)
# instance_norm_out = instance_norm(x)

# print(instance_norm_out)

# np.random.seed(123)
# x_data = np.random.random(size=(2, 2, 2, 2, 3)).astype('float32')
# x = paddle.to_tensor(x_data)
# instance_norm = paddle.nn.InstanceNorm3D(2)
# instance_norm_out = instance_norm(x)

# print(instance_norm_out.numpy)

# shape = (5, 20)
# x = np.random.uniform(-10, 10, shape).astype('float32')
# target = np.random.uniform(-10, 10, shape).astype('float32')

# kldiv_criterion = nn.KLDivLoss(reduction='batchmean')
# pred_loss = kldiv_criterion(paddle.to_tensor(x),
#                             paddle.to_tensor(target))

# kldiv_criterion = nn.KLDivLoss(reduction='mean')
# pred_loss = kldiv_criterion(paddle.to_tensor(x),
#                             paddle.to_tensor(target))

# kldiv_criterion = nn.KLDivLoss(reduction='sum')
# pred_loss = kldiv_criterion(paddle.to_tensor(x),
#                             paddle.to_tensor(target))

# kldiv_criterion = nn.KLDivLoss(reduction='none')
# pred_loss = kldiv_criterion(paddle.to_tensor(x),
#                             paddle.to_tensor(target))

# input_data = np.array([[1.5, 0.8], [0.2, 1.3]]).astype("float32")
# label_data = np.array([[1.7, 1], [0.4, 0.5]]).astype("float32")
# input = paddle.to_tensor(input_data)
# label = paddle.to_tensor(label_data)

# l1_loss = paddle.nn.L1Loss()
# output = l1_loss(input, label)
# print(output.numpy())

# l1_loss = paddle.nn.L1Loss(reduction='sum')
# output = l1_loss(input, label)
# print(output.numpy())

# l1_loss = paddle.nn.L1Loss(reduction='none')
# output = l1_loss(input, label)
# print(output)

# class MyLayer(paddle.nn.Layer):
#     def __init__(self):
#         super(MyLayer, self).__init__()
#         self._linear = paddle.nn.Linear(1, 1)
#         self._dropout = paddle.nn.Dropout(p=0.5)

#     def forward(self, input):
#         temp = self._linear(input)
#         temp = self._dropout(temp)
#         return temp

# x = paddle.randn([10, 1], 'float32')
# mylayer = MyLayer()
# mylayer.eval()  
# out = mylayer(x)
# mylayer.train()  
# out = mylayer(x)

# class MyLayer(paddle.nn.Layer):
#     def __init__(self):
#         super(MyLayer, self).__init__()
#         self._linear = paddle.nn.Linear(1, 1)
#         self._dropout = paddle.nn.Dropout(p=0.5)

#     def forward(self, input):
#         temp = self._linear(input)
#         temp = self._dropout(temp)
#         return temp

# x = paddle.randn([10, 1], 'float32')
# mylayer = MyLayer()
# mylayer.eval()  
# out = mylayer(x)
# print(out)

# net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))

# def init_weights(layer):
#     if type(layer) == nn.Linear:
#         print('before init weight:', layer.weight.numpy())
#         new_weight = paddle.full(shape=layer.weight.shape, dtype=layer.weight.dtype, fill_value=0.9)
#         layer.weight.set_value(new_weight)
#         print('after init weight:', layer.weight.numpy())

# net.apply(init_weights)

# print(net.state_dict())

# class LinearNet(paddle.nn.Layer):
#     def __init__(self):
#         super(LinearNet, self).__init__(name_scope = "demo_linear_net")
#         self._linear = paddle.nn.Linear(1, 1)

#     def forward(self, x):
#         return self._linear(x)

# linear_net = LinearNet()
# print(linear_net.full_name())   

# def forward_post_hook(layer, input, output):
    

    
#     return output * 2

# linear = paddle.nn.Linear(13, 5)

# forward_post_hook_handle = linear.register_forward_post_hook(forward_post_hook)

# value1 = np.arange(26).reshape(2, 13).astype("float32")
# in1 = paddle.to_tensor(value1)

# out0 = linear(in1)

# forward_post_hook_handle.remove()

# out1 = linear(in1)

# assert (out0.numpy() == (out1.numpy()) * 2).any()

# def forward_pre_hook(layer, input):
    

    
#     input_return = (input[0] * 2)
#     return input_return

# linear = paddle.nn.Linear(13, 5)

# forward_pre_hook_handle = linear.register_forward_pre_hook(forward_pre_hook)

# value0 = np.arange(26).reshape(2, 13).astype("float32")
# in0 = paddle.to_tensor(value0)
# out0 = linear(in0)

# forward_pre_hook_handle.remove()

# value1 = value0 * 2
# in1 = paddle.to_tensor(value1)
# out1 = linear(in1)

# assert (out0.numpy() == out1.numpy()).any()

# class MyLayer(paddle.nn.Layer):
#     def __init__(self):
#         super(MyLayer, self).__init__()
#         self._linear = paddle.nn.Linear(1, 1)
#         w_tmp = self.create_parameter([1,1])
#         self.add_parameter("w_tmp", w_tmp)

#     def forward(self, input):
#         return self._linear(input)

# mylayer = MyLayer()
# for name, param in mylayer.named_parameters():
#     print(name, param)      

# class MyLinear(paddle.nn.Layer):
#     def __init__(self,
#                 in_features,
#                 out_features):
#         super(MyLinear, self).__init__()
#         self.linear = paddle.nn.Linear( 10, 10)

#         self.back_var = self.create_variable(name = "linear_tmp_0", dtype=self._dtype)

#     def forward(self, input):
#         out = self.linear(input)
#         paddle.assign( out, self.back_var)

#         return out

# class MyLinear(paddle.nn.Layer):
#     def __init__(self,
#                 in_features,
#                 out_features):
#         super(MyLinear, self).__init__()
#         self.linear = paddle.nn.Linear( 10, 10)

#         self.back_var = self.create_tensor(name = "linear_tmp_0", dtype=self._dtype)

#     def forward(self, input):
#         out = self.linear(input)
#         paddle.assign( out, self.back_var)

#         return out

# linear1 = paddle.nn.Linear(10, 3)
# linear2 = paddle.nn.Linear(3, 10, bias_attr=False)
# model = paddle.nn.Sequential(linear1, linear2)

# layer_list = list(model.children())

# print(layer_list)   

# linear1 = paddle.nn.Linear(10, 3)
# linear2 = paddle.nn.Linear(3, 10, bias_attr=False)
# model = paddle.nn.Sequential(linear1, linear2)
# for prefix, layer in model.named_children():
#     print(prefix, layer)
    
    

# class MyLayer(paddle.nn.Layer):
#     def __init__(self):
#         super(MyLayer, self).__init__()
#         self._linear = paddle.nn.Linear(1, 1)
#         self._dropout = paddle.nn.Dropout(p=0.5)

#     def forward(self, input):
#         temp = self._linear(input)
#         temp = self._dropout(temp)
#         return temp

# mylayer = MyLayer()
# print(mylayer.sublayers())  

# fc1 = paddle.nn.Linear(10, 3)
# fc2 = paddle.nn.Linear(3, 10, bias_attr=False)
# model = paddle.nn.Sequential(fc1, fc2)
# for name, param in model.named_parameters():
#     print(name, param)

# fc1 = paddle.nn.Linear(10, 3)
# fc2 = paddle.nn.Linear(3, 10, bias_attr=False)
# model = paddle.nn.Sequential(fc1, fc2)
# for prefix, layer in model.named_sublayers():
#     print(prefix, layer)

# linear = paddle.nn.Linear(10, 3)
# value = np.array([0]).astype("float32")
# buffer = paddle.to_tensor(value)
# linear.register_buffer("buf_name", buffer, persistable=True)

# print(linear.buf_name)

# linear = paddle.nn.Linear(10, 3)
# value = np.array([0]).astype("float32")
# buffer = paddle.to_tensor(value)
# linear.register_buffer("buf_name", buffer, persistable=True)

# print(linear.buffers())     

# fc1 = paddle.nn.Linear(10, 3)
# buffer1 = paddle.to_tensor(np.array([0]).astype("float32"))

# fc1.register_buffer("buf_name_1", buffer1, persistable=True)

# fc2 = paddle.nn.Linear(3, 10)
# buffer2 = paddle.to_tensor(np.array([1]).astype("float32"))

# fc2.buf_name_2 = buffer2

# model = paddle.nn.Sequential(fc1, fc2)

# for name, buffer in model.named_buffers():
#     print(name, buffer)

# value = np.arange(26).reshape(2, 13).astype("float32")
# a = paddle.to_tensor(value)
# linear = paddle.nn.Linear(13, 5)
# adam = paddle.optimizer.Adam(learning_rate=0.01,
#                             parameters=linear.parameters())
# out = linear(a)
# out.backward()
# adam.step()
# linear.clear_gradients()

# class MySequential(paddle.nn.Layer):
#     def __init__(self, *layers):
#         super(MySequential, self).__init__()
#         if len(layers) > 0 and isinstance(layers[0], tuple):
#             for name, layer in layers:
#                 self.add_sublayer(name, layer)
#         else:
#             for idx, layer in enumerate(layers):
#                 self.add_sublayer(str(idx), layer)

#     def forward(self, input):
#         for layer in self._sub_layers.values():
#             input = layer(input)
#         return input

# fc1 = paddle.nn.Linear(10, 3)
# fc2 = paddle.nn.Linear(3, 10, bias_attr=False)
# model = MySequential(fc1, fc2)
# for prefix, layer in model.named_sublayers():
#     print(prefix, layer)

# class MyLayer(paddle.nn.Layer):
#     def __init__(self):
#         super(MyLayer, self).__init__()
#         self._linear = paddle.nn.Linear(1, 1)
#         w_tmp = self.create_parameter([1,1])
#         self.add_parameter("w_tmp", w_tmp)

#     def forward(self, input):
#         return self._linear(input)

# mylayer = MyLayer()
# for name, param in mylayer.named_parameters():
#     print(name, param)      

# emb = paddle.nn.Embedding(10, 10)

# state_dict = emb.to_static_state_dict()
# paddle.save( state_dict, "paddle_dy.pdparams")

# emb = paddle.nn.Embedding(10, 10)

# state_dict = emb.state_dict()
# paddle.save( state_dict, "paddle_dy.pdparams")

# emb = paddle.nn.Embedding(10, 10)

# state_dict = emb.state_dict()
# paddle.save(state_dict, "paddle_dy.pdparams")
# para_state_dict = paddle.load("paddle_dy.pdparams")
# emb.set_state_dict(para_state_dict)

# linear=paddle.nn.Linear(2, 2)
# linear.weight

# linear.to(dtype='float64')
# linear.weight

# linear.to(device='cpu')
# linear.weight

# linear.to(device=paddle.CUDAPinnedPlace(), blocking=False)
# linear.weight

# emb = paddle.nn.Embedding(10, 10)

# state_dict = emb.state_dict()
# paddle.save(state_dict, "paddle_dy.pdparams")
# para_state_dict = paddle.load("paddle_dy.pdparams")
# emb.set_state_dict(para_state_dict)

# emb = paddle.nn.Embedding(10, 10)

# state_dict = emb.state_dict()
# paddle.save(state_dict, "paddle_dy.pdparams")
# para_state_dict = paddle.load("paddle_dy.pdparams")
# emb.set_state_dict(para_state_dict)

# sublayers = OrderedDict([
#     ('conv1d', paddle.nn.Conv1D(3, 2, 3)),
#     ('conv2d', paddle.nn.Conv2D(3, 2, 3)),
#     ('conv3d', paddle.nn.Conv3D(4, 6, (3, 3, 3))),
# ])

# layers_dict = paddle.nn.LayerDict(sublayers=sublayers)

# l = layers_dict['conv1d']

# for k in layers_dict:
#     l = layers_dict[k]

# len(layers_dict)

# del layers_dict['conv2d']
# len(layers_dict)

# conv1d = layers_dict.pop('conv1d')
# len(layers_dict)

# layers_dict.clear()
# len(layers_dict)

# sublayers = OrderedDict([
#     ('conv1d', paddle.nn.Conv1D(3, 2, 3)),
#     ('conv2d', paddle.nn.Conv2D(3, 2, 3)),
#     ('conv3d', paddle.nn.Conv3D(4, 6, (3, 3, 3))),
# ])

# layer_dict = paddle.nn.LayerDict(sublayers=sublayers)
# len(layer_dict)

# layer_dict.clear()
# len(layer_dict)

# sublayers = OrderedDict([
#     ('conv1d', paddle.nn.Conv1D(3, 2, 3)),
#     ('conv2d', paddle.nn.Conv2D(3, 2, 3)),
#     ('conv3d', paddle.nn.Conv3D(4, 6, (3, 3, 3))),
# ])

# layer_dict = paddle.nn.LayerDict(sublayers=sublayers)
# len(layer_dict)

# layer_dict.pop('conv2d')
# len(layer_dict)

# sublayers = OrderedDict([
#     ('conv1d', paddle.nn.Conv1D(3, 2, 3)),
#     ('conv2d', paddle.nn.Conv2D(3, 2, 3)),
#     ('conv3d', paddle.nn.Conv3D(4, 6, (3, 3, 3))),
# ])

# layer_dict = paddle.nn.LayerDict(sublayers=sublayers)
# for k in layer_dict.keys():
#     print(k)

# sublayers = OrderedDict([
#     ('conv1d', paddle.nn.Conv1D(3, 2, 3)),
#     ('conv2d', paddle.nn.Conv2D(3, 2, 3)),
#     ('conv3d', paddle.nn.Conv3D(4, 6, (3, 3, 3))),
# ])

# layer_dict = paddle.nn.LayerDict(sublayers=sublayers)
# for k, v in layer_dict.items():
#     print(k, ":", v)

# sublayers = OrderedDict([
#     ('conv1d', paddle.nn.Conv1D(3, 2, 3)),
#     ('conv2d', paddle.nn.Conv2D(3, 2, 3)),
#     ('conv3d', paddle.nn.Conv3D(4, 6, (3, 3, 3))),
# ])

# layer_dict = paddle.nn.LayerDict(sublayers=sublayers)
# for v in layer_dict.values():
#     print(v)

# sublayers = OrderedDict([
#     ('conv1d', paddle.nn.Conv1D(3, 2, 3)),
#     ('conv2d', paddle.nn.Conv2D(3, 2, 3)),
#     ('conv3d', paddle.nn.Conv3D(4, 6, (3, 3, 3))),
# ])

# new_sublayers = OrderedDict([
#     ('relu', paddle.nn.ReLU()),
#     ('conv2d', paddle.nn.Conv2D(4, 2, 4)),
# ])
# layer_dict = paddle.nn.LayerDict(sublayers=sublayers)

# layer_dict.update(new_sublayers)

# for k, v in layer_dict.items():
#     print(k, ":", v)

# class MyLayer(paddle.nn.Layer):
#     def __init__(self):
#         super(MyLayer, self).__init__()
#         self.linears = paddle.nn.LayerList(
#             [paddle.nn.Linear(10, 10) for i in range(10)])

#     def forward(self, x):
        
#         for i, l in enumerate(self.linears):
#             x = self.linears[i // 2](x) + l(x)
#         return x

# linears = paddle.nn.LayerList([paddle.nn.Linear(10, 10) for i in range(10)])
# another = paddle.nn.Linear(10, 10)
# linears.append(another)
# print(len(linears))  

# linears = paddle.nn.LayerList([paddle.nn.Linear(10, 10) for i in range(10)])
# another = paddle.nn.Linear(10, 10)
# linears.insert(3, another)
# print(linears[3] is another)  
# another = paddle.nn.Linear(10, 10)
# linears.insert(-1, another)
# print(linears[-2] is another) 

# linears = paddle.nn.LayerList([paddle.nn.Linear(10, 10) for i in range(10)])
# another_list = paddle.nn.LayerList([paddle.nn.Linear(10, 10) for i in range(5)])
# linears.extend(another_list)
# print(len(linears))  
# print(another_list[0] is linears[10])  

# np.random.seed(123)
# x_data = np.random.random(size=(2, 2, 2, 3)).astype('float32')
# x = paddle.to_tensor(x_data)
# layer_norm = paddle.nn.LayerNorm(x_data.shape[1:])
# layer_norm_out = layer_norm(x)

# print(layer_norm_out)

# m = paddle.nn.LeakyReLU()
# x = paddle.to_tensor(np.array([-2, 0, 1], 'float32'))
# out = m(x)  

# weight_attr = paddle.ParamAttr(
#     name="weight",
#     initializer=paddle.nn.initializer.Constant(value=0.5))
# bias_attr = paddle.ParamAttr(
#     name="bias",
#     initializer=paddle.nn.initializer.Constant(value=1.0))
# linear = paddle.nn.Linear(2, 4, weight_attr=weight_attr, bias_attr=bias_attr)

# x = paddle.randn((3, 2), dtype="float32")

# y = linear(x)

# x = paddle.rand(shape=(3, 3, 112, 112), dtype="float32")
# m = paddle.nn.LocalResponseNorm(size=5)
# y = m(x)
# print(y.shape)  

# x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
# m = paddle.nn.LogSigmoid()
# out = m(x) 

# x = [[[-2.0, 3.0, -4.0, 5.0],
#       [3.0, -4.0, 5.0, -6.0],
#       [-7.0, -8.0, 8.0, 9.0]],
#      [[1.0, -2.0, -3.0, 4.0],
#       [-5.0, 6.0, 7.0, -8.0],
#       [6.0, 7.0, 8.0, 9.0]]]
# m = paddle.nn.LogSoftmax()
# x = paddle.to_tensor(x)
# out = m(x)

# rnn = paddle.nn.LSTM(16, 32, 2)

# x = paddle.randn((4, 23, 16))
# prev_h = paddle.randn((2, 4, 32))
# prev_c = paddle.randn((2, 4, 32))
# y, (h, c) = rnn(x, (prev_h, prev_c))

# print(y.shape)
# print(h.shape)
# print(c.shape)

# x = paddle.randn((4, 16))
# prev_h = paddle.randn((4, 32))
# prev_c = paddle.randn((4, 32))

# cell = paddle.nn.LSTMCell(16, 32)
# y, (h, c) = cell(x, (prev_h, prev_c))

# print(y.shape)
# print(h.shape)
# print(c.shape)

# input = paddle.to_tensor([[1, 2], [3, 4]], dtype="float32")
# other = paddle.to_tensor([[2, 1], [2, 4]], dtype="float32")
# label = paddle.to_tensor([[1, -1], [-1, -1]], dtype="float32")
# margin_rank_loss = paddle.nn.MarginRankingLoss()
# loss = margin_rank_loss(input, other, label)

# print(loss)

# x = paddle.rand([1, 2, 3, 4])

# m = paddle.nn.Maxout(groups=2)
# out = m(x)

# data = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32]).astype(np.float32))
# MaxPool1D = nn.MaxPool1D(kernel_size=2, stride=2, padding=0)
# pool_out = MaxPool1D(data)

# MaxPool1D = nn.MaxPool1D(kernel_size=2, stride=2, padding=0, return_mask=True)
# pool_out, indices = MaxPool1D(data)

# input = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32, 32]).astype(np.float32))
# MaxPool2D = nn.MaxPool2D(kernel_size=2,
#                        stride=2, padding=0)
# output = MaxPool2D(input)

# MaxPool2D = nn.MaxPool2D(kernel_size=2, stride=2, padding=0, return_mask=True)
# output, max_indices = MaxPool2D(input)

# input = paddle.to_tensor(np.random.uniform(-1, 1, [1, 2, 3, 32, 32]).astype(np.float32))
# MaxPool3D = nn.MaxPool3D(kernel_size=2,
#                        stride=2, padding=0)
# output = MaxPool3D(input)

# MaxPool3D = nn.MaxPool3D(kernel_size=2, stride=2, padding=0, return_mask=True)
# output, max_indices = MaxPool3D(input)

# x = paddle.to_tensor([-5., 0., 5.])
# m = paddle.nn.Mish()
# out = m(x) 

# input_data = np.array([1.5]).astype("float32")
# label_data = np.array([1.7]).astype("float32")

# mse_loss = paddle.nn.loss.MSELoss()
# input = paddle.to_tensor(input_data)
# label = paddle.to_tensor(label_data)
# output = mse_loss(input, label)
# print(output)

# query = paddle.rand((2, 4, 128))

# attn_mask = paddle.rand((2, 2, 4, 4))
# multi_head_attn = paddle.nn.MultiHeadAttention(128, 2)
# output = multi_head_attn(query, None, None, attn_mask=attn_mask)  

# nll_loss = paddle.nn.loss.NLLLoss()
# log_softmax = paddle.nn.LogSoftmax(axis=1)

# input = paddle.to_tensor([[0.88103855, 0.9908683 , 0.6226845 ],
#                           [0.53331435, 0.07999352, 0.8549948 ],
#                           [0.25879037, 0.39530203, 0.698465  ],
#                           [0.73427284, 0.63575995, 0.18827209],
#                           [0.05689114, 0.0862954 , 0.6325046 ]], "float32")
# log_out = log_softmax(input)
# label = paddle.to_tensor([0, 2, 1, 1, 0], "int64")
# result = nll_loss(log_out, label)
# print(result) 

# input_shape = (1, 2, 3)
# pad = [1, 2]
# mode = "constant"
# data = paddle.arange(np.prod(input_shape), dtype="float32").reshape(input_shape) + 1
# my_pad = nn.Pad1D(padding=pad, mode=mode)
# result = my_pad(data)
# print(result)

# input_shape = (1, 1, 2, 3)
# pad = [1, 0, 1, 2]
# mode = "constant"
# data = paddle.arange(np.prod(input_shape), dtype="float32").reshape(input_shape) + 1
# my_pad = nn.Pad2D(padding=pad, mode=mode)
# result = my_pad(data)
# print(result)

# input_shape = (1, 1, 1, 2, 3)
# pad = [1, 0, 1, 2, 0, 0]
# mode = "constant"
# data = paddle.arange(np.prod(input_shape), dtype="float32").reshape(input_shape) + 1
# my_pad = nn.Pad3D(padding=pad, mode=mode)
# result = my_pad(data)
# print(result)

# paddle.disable_static()
# x_np = np.array([[1., 3.], [3., 5.]]).astype(np.float64)
# y_np = np.array([[5., 6.], [7., 8.]]).astype(np.float64)
# x = paddle.to_tensor(x_np)
# y = paddle.to_tensor(y_np)
# dist = paddle.nn.PairwiseDistance()
# distance = dist(x, y)
# print(distance.numpy()) 

# class MyLayer(paddle.nn.Layer):
#     def __init__(self, num_stacked_param):
#         super(MyLayer, self).__init__()
        
#         self.params = paddle.nn.ParameterList(
#             [paddle.create_parameter(
#                 shape=[2, 2], dtype='float32')] * num_stacked_param)

#     def forward(self, x):
#         for i, p in enumerate(self.params):
#             tmp = self._helper.create_variable_for_type_inference('float32')
#             self._helper.append_op(
#                 type="mul",
#                 inputs={"X": x,
#                         "Y": p},
#                 outputs={"Out": tmp},
#                 attrs={"x_num_col_dims": 1,
#                         "y_num_col_dims": 1})
#             x = tmp
#         return x

# data_np = np.random.uniform(-1, 1, [5, 2]).astype('float32')
# x = paddle.to_tensor(data_np)
# num_stacked_param = 4
# model = MyLayer(num_stacked_param)
# print(len(model.params))  
# res = model(x)
# print(res.shape)  

# replaced_param = paddle.create_parameter(shape=[2, 3], dtype='float32')
# model.params[num_stacked_param - 1] = replaced_param  
# res = model(x)
# print(res.shape)  
# model.params.append(paddle.create_parameter(shape=[3, 4], dtype='float32'))  
# print(len(model.params))  
# res = model(x)
# print(res.shape)  

# x = np.random.randn(2, 9, 4, 4).astype(np.float32)
# x_var = paddle.to_tensor(x)
# pixel_shuffle = nn.PixelShuffle(3)
# out_var = pixel_shuffle(x_var)
# out = out_var.numpy()
# print(out.shape)

# paddle.set_default_dtype("float64")

# data = np.array([[[[-2.0,  3.0, -4.0,  5.0],
#                 [ 3.0, -4.0,  5.0, -6.0],
#                 [-7.0, -8.0,  8.0,  9.0]],
#                 [[ 1.0, -2.0, -3.0,  4.0],
#                 [-5.0,  6.0,  7.0, -8.0],
#                 [ 6.0,  7.0,  8.0,  9.0]]]], 'float64')
# x = paddle.to_tensor(data)
# m = paddle.nn.PReLU(1, 0.25)
# out = m(x)

# x = paddle.to_tensor([-2., 0., 1.])
# m = paddle.nn.ReLU()
# out = m(x) 

# x = paddle.to_tensor(np.array([-1, 0.3, 6.5]))
# m = paddle.nn.ReLU6()
# out = m(x) 

# inputs = paddle.rand((4, 23, 16))
# prev_h = paddle.randn((4, 32))

# cell = paddle.nn.SimpleRNNCell(16, 32)
# rnn = paddle.nn.RNN(cell)
# outputs, final_states = rnn(inputs, prev_h)

# print(outputs.shape)
# print(final_states.shape)

# x = paddle.to_tensor(np.array([[0.0, 1.0],[2.0, 3.0]]))
# m = paddle.nn.SELU()
# out = m(x) 

# data = np.random.uniform(-1, 1, [30, 10]).astype('float32')
# data = paddle.to_tensor(data)

# model1 = paddle.nn.Sequential(
#     paddle.nn.Linear(10, 1), paddle.nn.Linear(1, 2)
# )
# model1[0]  
# res1 = model1(data)  

# model2 = paddle.nn.Sequential(
#     ('l1', paddle.nn.Linear(10, 2)),
#     ('l2', paddle.nn.Linear(2, 3))
# )
# model2['l1']  
# model2.add_sublayer('l3', paddle.nn.Linear(3, 3))  
# res2 = model2(data)  

# m = paddle.nn.Sigmoid()
# x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
# out = m(x) 

# x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
# m = paddle.nn.Silu()
# out = m(x) 

# rnn = paddle.nn.SimpleRNN(16, 32, 2)

# x = paddle.randn((4, 23, 16))
# prev_h = paddle.randn((2, 4, 32))
# y, h = rnn(x, prev_h)

# print(y.shape)
# print(h.shape)

# x = paddle.randn((4, 16))
# prev_h = paddle.randn((4, 32))

# cell = paddle.nn.SimpleRNNCell(16, 32)
# y, h = cell(x, prev_h)
# print(y.shape)

# input_data = np.random.rand(3,3).astype("float32")
# label_data = np.random.rand(3,3).astype("float32")
# input = paddle.to_tensor(input_data)
# label = paddle.to_tensor(label_data)
# loss = paddle.nn.SmoothL1Loss()
# output = loss(input, label)
# print(output)

# x = np.array([[[2.0, 3.0, 4.0, 5.0],
#             [3.0, 4.0, 5.0, 6.0],
#             [7.0, 8.0, 8.0, 9.0]],
#             [[1.0, 2.0, 3.0, 4.0],
#             [5.0, 6.0, 7.0, 8.0],
#             [6.0, 7.0, 8.0, 9.0]]], 'float32')
# x = paddle.to_tensor(x)
# m = paddle.nn.Softmax()
# out = m(x)

# x = paddle.to_tensor(np.array([-0.4, -0.2, 0.1, 0.3]))
# m = paddle.nn.Softplus()
# out = m(x) 

# x = paddle.to_tensor(np.array([-0.9, -0.2, 0.1, 0.8]))
# m = paddle.nn.Softshrink()
# out = m(x) 

# x = paddle.to_tensor(np.array([-0.4, -0.2, 0.1, 0.3]))
# m = paddle.nn.Softsign()
# out = m(x) 

# x = paddle.rand((2,8,32,32))

# spectral_norm = paddle.nn.SpectralNorm(x.shape, dim=1, power_iters=2)
# spectral_norm_out = spectral_norm(x)

# print(spectral_norm_out.shape) 

# x = paddle.to_tensor(np.array([-2., 0., 1.]))
# m = paddle.nn.Swish()
# out = m(x) 

# x = np.array([[[[0.3, 0.4], [0.3, 0.07]], [[0.83, 0.37], [0.18, 0.93]]]]).astype('float32')
# x = paddle.to_tensor(x)

# if paddle.is_compiled_with_cuda():
#     sync_batch_norm = nn.SyncBatchNorm(2)
#     hidden1 = sync_batch_norm(x)
#     print(hidden1)
    


# model = nn.Sequential(nn.Conv2D(3, 5, 3), nn.BatchNorm2D(5))
# sync_model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

# x = paddle.to_tensor(np.array([-0.4, -0.2, 0.1, 0.3]))
# m = paddle.nn.Tanh()
# out = m(x)
# print(out)

# x = paddle.to_tensor(np.array([-0.4, -0.2, 0.1, 0.3]))
# m = paddle.nn.Tanhshrink()
# out = m(x) 

# x = paddle.to_tensor(np.array([2., 0., 1.]))
# m = paddle.nn.ThresholdedReLU()
# out = m(x) 

# enc_input = paddle.rand((2, 4, 128))

# dec_input = paddle.rand((2, 6, 128))

# enc_self_attn_mask = paddle.rand((2, 2, 4, 4))

# dec_self_attn_mask = paddle.rand((2, 2, 6, 6))

# cross_attn_mask = paddle.rand((2, 2, 6, 4))
# transformer = Transformer(128, 2, 4, 4, 512)
# output = transformer(enc_input,
#                      dec_input,
#                      enc_self_attn_mask,
#                      dec_self_attn_mask,
#                      cross_attn_mask)  

# length = 5
# d_model, n_head, dim_feedforward = 8, 4, 64
# transformer_paddle = Transformer(
#     d_model, n_head, dim_feedforward=dim_feedforward)
# mask = transformer_paddle.generate_square_subsequent_mask(length)
# print(mask)

# dec_input = paddle.rand((2, 4, 128))

# enc_output = paddle.rand((2, 6, 128))

# self_attn_mask = paddle.rand((2, 2, 4, 4))

# cross_attn_mask = paddle.rand((2, 2, 4, 6))
# decoder_layer = TransformerDecoderLayer(128, 2, 512)
# decoder = TransformerDecoder(decoder_layer, 2)
# output = decoder(dec_input,
#                  enc_output,
#                  self_attn_mask,
#                  cross_attn_mask)  

# dec_input = paddle.rand((2, 4, 128))

# enc_output = paddle.rand((2, 6, 128))

# self_attn_mask = paddle.rand((2, 2, 4, 4))

# cross_attn_mask = paddle.rand((2, 2, 4, 6))
# decoder_layer = TransformerDecoderLayer(128, 2, 512)
# output = decoder_layer(dec_input,
#                        enc_output,
#                        self_attn_mask,
#                        cross_attn_mask)  

# enc_input = paddle.rand((2, 4, 128))

# attn_mask = paddle.rand((2, 2, 4, 4))
# encoder_layer = TransformerEncoderLayer(128, 2, 512)
# encoder = TransformerEncoder(encoder_layer, 2)
# enc_output = encoder(enc_input, attn_mask)  

# enc_input = paddle.rand((2, 4, 128))

# attn_mask = paddle.rand((2, 2, 4, 4))
# encoder_layer = TransformerEncoderLayer(128, 2, 512)
# enc_output = encoder_layer(enc_input, attn_mask)  

# x = paddle.randn((100,3,224,224))
# unfold = nn.Unfold(kernel_sizes=[3, 3])
# result = unfold(x)
# print(result)

# input_data = np.random.rand(2,3,6,10).astype("float32")
# upsample_out  = paddle.nn.Upsample(size=[12,12])

# input = paddle.to_tensor(input_data)
# output = upsample_out(x=input)
# print(output.shape)

# input_data = paddle.rand(shape=(2,3,6,10)).astype("float32")
# upsample_out  = paddle.nn.UpsamplingBilinear2D(size=[12,12])
# input = paddle.to_tensor(input_data)
# output = upsample_out(x=input)
# print(output.shape)

# input_data = paddle.rand(shape=(2,3,6,10)).astype("float32")
# upsample_out  = paddle.nn.UpsamplingNearest2D(size=[12,12])
# input = paddle.to_tensor(input_data)
# output = upsample_out(x=input)
# print(output.shape)

# class LinearNet(paddle.nn.Layer):
#     def __init__(self):
#         super(LinearNet, self).__init__()
#         self._linear = paddle.nn.Linear(128, 10)

#     def forward(self, x):
#         return self._linear(x)

# def export_linear_net():
#     model = LinearNet()
#     x_spec = paddle.static.InputSpec(shape=[None, 128], dtype='float32')
#     paddle.onnx.export(model, 'linear_net', input_spec=[x_spec])

# export_linear_net()

# class Logic(paddle.nn.Layer):
#     def __init__(self):
#         super(Logic, self).__init__()

#     def forward(self, x, y, z):
#         if z:
#             return x
#         else:
#             return y

# def export_logic():
#     model = Logic()
#     x = paddle.to_tensor(np.array([1]))
#     y = paddle.to_tensor(np.array([2]))
    
#     paddle.jit.to_static(model)
#     out = model(x, y, z=True)
#     paddle.onnx.export(model, 'pruned', input_spec=[x], output_spec=[out])

# export_logic()

# inp = np.random.uniform(-0.1, 0.1, [10, 10]).astype("float32")
# linear = paddle.nn.Linear(10, 10)
# inp = paddle.to_tensor(inp)
# out = linear(inp)
# loss = paddle.mean(out)
# beta1 = paddle.to_tensor([0.9], dtype="float32")
# beta2 = paddle.to_tensor([0.99], dtype="float32")
# adadelta = paddle.optimizer.Adadelta(learning_rate=0.1, parameters=linear.parameters(), weight_decay=0.01)
# back = out.backward()
# adadelta.step()
# adadelta.clear_grad()

# linear_1 = paddle.nn.Linear(10, 10)
# linear_2 = paddle.nn.Linear(10, 10)
# inp = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
# out = linear_1(inp)
# out = linear_2(out)
# loss = paddle.mean(out)
# adadelta = paddle.optimizer.Adadelta(
#     learning_rate=0.1,
#     parameters=[{
#         'params': linear_1.parameters()
#     }, {
#         'params': linear_2.parameters(),
#         'weight_decay': 0.001,
#         'learning_rate': 0.1,
#     }],
#     weight_decay=0.01)
# out.backward()
# adadelta.step()
# adadelta.clear_grad()

# value = np.arange(26).reshape(2, 13).astype("float32")
# a = paddle.to_tensor(value)
# linear = paddle.nn.Linear(13, 5)

# adam = paddle.optimizer.Adam(learning_rate = 0.01,
#                             parameters = linear.parameters())
# out = linear(a)
# out.backward()
# adam.step()
# adam.clear_grad()

# emb = paddle.nn.Embedding(10, 3)

# adam = paddle.optimizer.Adam(0.01, parameters = emb.parameters())
# for batch in range(10):
#     input = paddle.randint(low=0, high=5, shape=[5])
#     out = emb(input)
#     out.backward()
#     print("Learning rate of step{}: {}".format(batch, adam.get_lr())) 
#     adam.step()

# scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.5, step_size=2, gamma=0.1)
# adam = paddle.optimizer.Adam(scheduler, parameters = emb.parameters())
# for batch in range(10):
#     input = paddle.randint(low=0, high=5, shape=[5])
#     out = emb(input)
#     out.backward()
#     print("Learning rate of step{}: {}".format(batch, adam.get_lr())) 
#     adam.step()
#     scheduler.step()

# paddle.enable_static()
# main_prog = paddle.static.Program()
# start_prog = paddle.static.Program()
# with paddle.static.program_guard(main_prog, start_prog):
#     x = paddle.static.data(name='x', shape=[None, 10])
#     z = paddle.static.nn.fc(x, 100)
#     loss = paddle.mean(z)
#     scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.5, step_size=2, gamma=0.1)
#     adam = paddle.optimizer.Adam(learning_rate=scheduler)
#     adam.minimize(loss)

# exe = paddle.static.Executor()
# exe.run(start_prog)
# for batch in range(10):
#     print("Learning rate of step{}: {}", adam.get_lr())     
#     out = exe.run(main_prog, feed={'x': np.random.randn(3, 10).astype('float32')})
#     scheduler.step()

# linear = paddle.nn.Linear(10, 10)
# input = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
# out = linear(input)
# loss = paddle.mean(out)

# beta1 = paddle.to_tensor([0.9], dtype="float32")
# beta2 = paddle.to_tensor([0.99], dtype="float32")

# adam = paddle.optimizer.Adam(learning_rate=0.1,
#         parameters=linear.parameters(),
#         weight_decay=0.01)
# out.backward()
# adam.minimize(loss)
# adam.clear_grad()

# linear = paddle.nn.Linear(10, 10)

# adam = paddle.optimizer.Adam(0.1, parameters=linear.parameters())

# lr_list = [0.2, 0.3, 0.4, 0.5, 0.6]
# for i in range(5):
#     adam.set_lr(lr_list[i])
#     lr = adam.get_lr()
#     print("current lr is {}".format(lr))

# emb = paddle.nn.Embedding(10, 10)

# layer_state_dict = emb.state_dict()
# paddle.save(layer_state_dict, "emb.pdparams")

# scheduler = paddle.optimizer.lr.NoamDecay(
#     d_model=0.01, warmup_steps=100, verbose=True)
# adam = paddle.optimizer.Adam(
#     learning_rate=scheduler,
#     parameters=emb.parameters())
# opt_state_dict = adam.state_dict()
# paddle.save(opt_state_dict, "adam.pdopt")

# opti_state_dict = paddle.load("adam.pdopt")
# adam.set_state_dict(opti_state_dict)

# emb = paddle.nn.Embedding(10, 10)

# adam = paddle.optimizer.Adam(0.001, parameters=emb.parameters())
# state_dict = adam.state_dict()

# value = np.arange(26).reshape(2, 13).astype("float32")
# a = paddle.to_tensor(value)
# linear = paddle.nn.Linear(13, 5)

# adam = paddle.optimizer.Adam(learning_rate = 0.01,
#                             parameters = linear.parameters())
# out = linear(a)
# out.backward()
# adam.step()
# adam.clear_grad()

# inp = paddle.rand(shape=[10, 10])
# linear = paddle.nn.Linear(10, 10)
# out = linear(inp)
# loss = paddle.mean(out)
# adagrad = paddle.optimizer.Adagrad(learning_rate=0.1,
#         parameters=linear.parameters())
# out.backward()
# adagrad.step()
# adagrad.clear_grad()

# linear_1 = paddle.nn.Linear(10, 10)
# linear_2 = paddle.nn.Linear(10, 10)
# inp = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
# out = linear_1(inp)
# out = linear_2(out)
# loss = paddle.mean(out)
# adagrad = paddle.optimizer.Adagrad(
#     learning_rate=0.1,
#     parameters=[{
#         'params': linear_1.parameters()
#     }, {
#         'params': linear_2.parameters(),
#         'weight_decay': 0.001,
#         'learning_rate': 0.1,
#     }],
#     weight_decay=0.01)
# out.backward()
# adagrad.step()
# adagrad.clear_grad()

# value = np.arange(26).reshape(2, 13).astype("float32")
# a = paddle.to_tensor(value)
# linear = paddle.nn.Linear(13, 5)

# adam = paddle.optimizer.Adam(learning_rate = 0.01,
#                             parameters = linear.parameters())
# out = linear(a)
# out.backward()
# adam.step()
# adam.clear_grad()

# emb = paddle.nn.Embedding(10, 3)

# adam = paddle.optimizer.Adam(0.01, parameters = emb.parameters())
# for batch in range(10):
#     input = paddle.randint(low=0, high=5, shape=[5])
#     out = emb(input)
#     out.backward()
#     print("Learning rate of step{}: {}".format(batch, adam.get_lr())) 
#     adam.step()

# scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.5, step_size=2, gamma=0.1)
# adam = paddle.optimizer.Adam(scheduler, parameters = emb.parameters())
# for batch in range(10):
#     input = paddle.randint(low=0, high=5, shape=[5])
#     out = emb(input)
#     out.backward()
#     print("Learning rate of step{}: {}".format(batch, adam.get_lr())) 
#     adam.step()
#     scheduler.step()

# paddle.enable_static()
# main_prog = paddle.static.Program()
# start_prog = paddle.static.Program()
# with paddle.static.program_guard(main_prog, start_prog):
#     x = paddle.static.data(name='x', shape=[None, 10])
#     z = paddle.static.nn.fc(x, 100)
#     loss = paddle.mean(z)
#     scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.5, step_size=2, gamma=0.1)
#     adam = paddle.optimizer.Adam(learning_rate=scheduler)
#     adam.minimize(loss)

# exe = paddle.static.Executor()
# exe.run(start_prog)
# for batch in range(10):
#     print("Learning rate of step{}: {}", adam.get_lr())     
#     out = exe.run(main_prog, feed={'x': np.random.randn(3, 10).astype('float32')})
#     scheduler.step()

# linear = paddle.nn.Linear(10, 10)
# input = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
# out = linear(input)
# loss = paddle.mean(out)

# beta1 = paddle.to_tensor([0.9], dtype="float32")
# beta2 = paddle.to_tensor([0.99], dtype="float32")

# adam = paddle.optimizer.Adam(learning_rate=0.1,
#         parameters=linear.parameters(),
#         weight_decay=0.01)
# out.backward()
# adam.minimize(loss)
# adam.clear_grad()

# linear = paddle.nn.Linear(10, 10)

# adam = paddle.optimizer.Adam(0.1, parameters=linear.parameters())

# lr_list = [0.2, 0.3, 0.4, 0.5, 0.6]
# for i in range(5):
#     adam.set_lr(lr_list[i])
#     lr = adam.get_lr()
#     print("current lr is {}".format(lr))

# emb = paddle.nn.Embedding(10, 10)

# layer_state_dict = emb.state_dict()
# paddle.save(layer_state_dict, "emb.pdparams")

# scheduler = paddle.optimizer.lr.NoamDecay(
#     d_model=0.01, warmup_steps=100, verbose=True)
# adam = paddle.optimizer.Adam(
#     learning_rate=scheduler,
#     parameters=emb.parameters())
# opt_state_dict = adam.state_dict()
# paddle.save(opt_state_dict, "adam.pdopt")

# opti_state_dict = paddle.load("adam.pdopt")
# adam.set_state_dict(opti_state_dict)

# emb = paddle.nn.Embedding(10, 10)

# adam = paddle.optimizer.Adam(0.001, parameters=emb.parameters())
# state_dict = adam.state_dict()

# value = np.arange(26).reshape(2, 13).astype("float32")
# a = paddle.to_tensor(value)
# linear = paddle.nn.Linear(13, 5)

# adam = paddle.optimizer.Adam(learning_rate = 0.01,
#                             parameters = linear.parameters())
# out = linear(a)
# out.backward()
# adam.step()
# adam.clear_grad()

# linear = paddle.nn.Linear(10, 10)
# inp = paddle.rand([10,10], dtype="float32")
# out = linear(inp)
# loss = paddle.mean(out)
# adam = paddle.optimizer.Adam(learning_rate=0.1,
#         parameters=linear.parameters())
# out.backward()
# adam.step()
# adam.clear_grad()

# linear = paddle.nn.Linear(10, 10)
# inp = paddle.rand([10,10], dtype="float32")
# out = linear(inp)
# loss = paddle.mean(out)

# beta1 = paddle.to_tensor([0.9], dtype="float32")
# beta2 = paddle.to_tensor([0.99], dtype="float32")

# adam = paddle.optimizer.Adam(learning_rate=0.1,
#         parameters=linear.parameters(),
#         beta1=beta1,
#         beta2=beta2,
#         weight_decay=0.01)
# out.backward()
# adam.step()
# adam.clear_grad()

# linear_1 = paddle.nn.Linear(10, 10)
# linear_2 = paddle.nn.Linear(10, 10)
# inp = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
# out = linear_1(inp)
# out = linear_2(out)
# loss = paddle.mean(out)
# adam = paddle.optimizer.Adam(
#     learning_rate=0.1,
#     parameters=[{
#         'params': linear_1.parameters()
#     }, {
#         'params': linear_2.parameters(),
#         'weight_decay': 0.001,
#         'learning_rate': 0.1,
#         'beta1': 0.8
#     }],
#     weight_decay=0.01,
#     beta1=0.9)
# out.backward()
# adam.step()
# adam.clear_grad()

# a = paddle.rand([2,13], dtype="float32")
# linear = paddle.nn.Linear(13, 5)

# adam = paddle.optimizer.Adam(learning_rate = 0.01,
#                             parameters = linear.parameters())
# out = linear(a)
# out.backward()
# adam.step()
# adam.clear_grad()

# value = np.arange(26).reshape(2, 13).astype("float32")
# a = paddle.to_tensor(value)
# linear = paddle.nn.Linear(13, 5)

# adam = paddle.optimizer.Adam(learning_rate = 0.01,
#                             parameters = linear.parameters())
# out = linear(a)
# out.backward()
# adam.step()
# adam.clear_grad()

# emb = paddle.nn.Embedding(10, 3)

# adam = paddle.optimizer.Adam(0.01, parameters = emb.parameters())
# for batch in range(10):
#     input = paddle.randint(low=0, high=5, shape=[5])
#     out = emb(input)
#     out.backward()
#     print("Learning rate of step{}: {}".format(batch, adam.get_lr())) 
#     adam.step()

# scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.5, step_size=2, gamma=0.1)
# adam = paddle.optimizer.Adam(scheduler, parameters = emb.parameters())
# for batch in range(10):
#     input = paddle.randint(low=0, high=5, shape=[5])
#     out = emb(input)
#     out.backward()
#     print("Learning rate of step{}: {}".format(batch, adam.get_lr())) 
#     adam.step()
#     scheduler.step()

# paddle.enable_static()
# main_prog = paddle.static.Program()
# start_prog = paddle.static.Program()
# with paddle.static.program_guard(main_prog, start_prog):
#     x = paddle.static.data(name='x', shape=[None, 10])
#     z = paddle.static.nn.fc(x, 100)
#     loss = paddle.mean(z)
#     scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.5, step_size=2, gamma=0.1)
#     adam = paddle.optimizer.Adam(learning_rate=scheduler)
#     adam.minimize(loss)

# exe = paddle.static.Executor()
# exe.run(start_prog)
# for batch in range(10):
#     print("Learning rate of step{}: {}", adam.get_lr())     
#     out = exe.run(main_prog, feed={'x': np.random.randn(3, 10).astype('float32')})
#     scheduler.step()

# linear = paddle.nn.Linear(10, 10)
# input = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
# out = linear(input)
# loss = paddle.mean(out)

# beta1 = paddle.to_tensor([0.9], dtype="float32")
# beta2 = paddle.to_tensor([0.99], dtype="float32")

# adam = paddle.optimizer.Adam(learning_rate=0.1,
#         parameters=linear.parameters(),
#         weight_decay=0.01)
# out.backward()
# adam.minimize(loss)
# adam.clear_grad()

# linear = paddle.nn.Linear(10, 10)

# adam = paddle.optimizer.Adam(0.1, parameters=linear.parameters())

# lr_list = [0.2, 0.3, 0.4, 0.5, 0.6]
# for i in range(5):
#     adam.set_lr(lr_list[i])
#     lr = adam.get_lr()
#     print("current lr is {}".format(lr))

# emb = paddle.nn.Embedding(10, 10)

# layer_state_dict = emb.state_dict()
# paddle.save(layer_state_dict, "emb.pdparams")

# scheduler = paddle.optimizer.lr.NoamDecay(
#     d_model=0.01, warmup_steps=100, verbose=True)
# adam = paddle.optimizer.Adam(
#     learning_rate=scheduler,
#     parameters=emb.parameters())
# opt_state_dict = adam.state_dict()
# paddle.save(opt_state_dict, "adam.pdopt")

# opti_state_dict = paddle.load("adam.pdopt")
# adam.set_state_dict(opti_state_dict)

# emb = paddle.nn.Embedding(10, 10)

# adam = paddle.optimizer.Adam(0.001, parameters=emb.parameters())
# state_dict = adam.state_dict()

# inp = np.random.uniform(-0.1, 0.1, [10, 10]).astype("float32")
# linear = paddle.nn.Linear(10, 10)
# inp = paddle.to_tensor(inp)
# out = linear(inp)
# loss = paddle.mean(out)

# beta1 = paddle.to_tensor([0.9], dtype="float32")
# beta2 = paddle.to_tensor([0.99], dtype="float32")

# adam = paddle.optimizer.Adamax(learning_rate=0.1,
#         parameters=linear.parameters(),
#         beta1=beta1,
#         beta2=beta2,
#         weight_decay=0.01)
# out.backward()
# adam.step()
# adam.clear_grad()

# linear_1 = paddle.nn.Linear(10, 10)
# linear_2 = paddle.nn.Linear(10, 10)
# inp = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
# out = linear_1(inp)
# out = linear_2(out)
# loss = paddle.mean(out)
# adam = paddle.optimizer.Adamax(
#     learning_rate=0.1,
#     parameters=[{
#         'params': linear_1.parameters()
#     }, {
#         'params': linear_2.parameters(),
#         'weight_decay': 0.001,
#         'learning_rate': 0.1,
#         'beta1': 0.8
#     }],
#     weight_decay=0.01,
#     beta1=0.9)
# out.backward()
# adam.step()
# adam.clear_grad()

# value = np.arange(26).reshape(2, 13).astype("float32")
# a = paddle.to_tensor(value)
# linear = paddle.nn.Linear(13, 5)

# adam = paddle.optimizer.Adam(learning_rate = 0.01,
#                             parameters = linear.parameters())
# out = linear(a)
# out.backward()
# adam.step()
# adam.clear_grad()

# emb = paddle.nn.Embedding(10, 3)

# adam = paddle.optimizer.Adam(0.01, parameters = emb.parameters())
# for batch in range(10):
#     input = paddle.randint(low=0, high=5, shape=[5])
#     out = emb(input)
#     out.backward()
#     print("Learning rate of step{}: {}".format(batch, adam.get_lr())) 
#     adam.step()

# scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.5, step_size=2, gamma=0.1)
# adam = paddle.optimizer.Adam(scheduler, parameters = emb.parameters())
# for batch in range(10):
#     input = paddle.randint(low=0, high=5, shape=[5])
#     out = emb(input)
#     out.backward()
#     print("Learning rate of step{}: {}".format(batch, adam.get_lr())) 
#     adam.step()
#     scheduler.step()

# paddle.enable_static()
# main_prog = paddle.static.Program()
# start_prog = paddle.static.Program()
# with paddle.static.program_guard(main_prog, start_prog):
#     x = paddle.static.data(name='x', shape=[None, 10])
#     z = paddle.static.nn.fc(x, 100)
#     loss = paddle.mean(z)
#     scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.5, step_size=2, gamma=0.1)
#     adam = paddle.optimizer.Adam(learning_rate=scheduler)
#     adam.minimize(loss)

# exe = paddle.static.Executor()
# exe.run(start_prog)
# for batch in range(10):
#     print("Learning rate of step{}: {}", adam.get_lr())     
#     out = exe.run(main_prog, feed={'x': np.random.randn(3, 10).astype('float32')})
#     scheduler.step()

# linear = paddle.nn.Linear(10, 10)
# input = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
# out = linear(input)
# loss = paddle.mean(out)

# beta1 = paddle.to_tensor([0.9], dtype="float32")
# beta2 = paddle.to_tensor([0.99], dtype="float32")

# adam = paddle.optimizer.Adam(learning_rate=0.1,
#         parameters=linear.parameters(),
#         weight_decay=0.01)
# out.backward()
# adam.minimize(loss)
# adam.clear_grad()

# linear = paddle.nn.Linear(10, 10)

# adam = paddle.optimizer.Adam(0.1, parameters=linear.parameters())

# lr_list = [0.2, 0.3, 0.4, 0.5, 0.6]
# for i in range(5):
#     adam.set_lr(lr_list[i])
#     lr = adam.get_lr()
#     print("current lr is {}".format(lr))

# emb = paddle.nn.Embedding(10, 10)

# layer_state_dict = emb.state_dict()
# paddle.save(layer_state_dict, "emb.pdparams")

# scheduler = paddle.optimizer.lr.NoamDecay(
#     d_model=0.01, warmup_steps=100, verbose=True)
# adam = paddle.optimizer.Adam(
#     learning_rate=scheduler,
#     parameters=emb.parameters())
# opt_state_dict = adam.state_dict()
# paddle.save(opt_state_dict, "adam.pdopt")

# opti_state_dict = paddle.load("adam.pdopt")
# adam.set_state_dict(opti_state_dict)

# emb = paddle.nn.Embedding(10, 10)

# adam = paddle.optimizer.Adam(0.001, parameters=emb.parameters())
# state_dict = adam.state_dict()

# value = np.arange(26).reshape(2, 13).astype("float32")
# a = paddle.to_tensor(value)
# linear = paddle.nn.Linear(13, 5)

# adam = paddle.optimizer.Adam(learning_rate = 0.01,
#                             parameters = linear.parameters())
# out = linear(a)
# out.backward()
# adam.step()
# adam.clear_grad()

# linear = paddle.nn.Linear(10, 10)
# inp = paddle.rand([10,10], dtype="float32")
# out = linear(inp)
# loss = paddle.mean(out)

# beta1 = paddle.to_tensor([0.9], dtype="float32")
# beta2 = paddle.to_tensor([0.99], dtype="float32")

# adam = paddle.optimizer.AdamW(learning_rate=0.1,
#         parameters=linear.parameters(),
#         beta1=beta1,
#         beta2=beta2,
#         weight_decay=0.01)
# out.backward()
# adam.step()
# adam.clear_grad()

# linear_1 = paddle.nn.Linear(10, 10)
# linear_2 = paddle.nn.Linear(10, 10)
# inp = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
# out = linear_1(inp)
# out = linear_2(out)
# loss = paddle.mean(out)
# adam = paddle.optimizer.AdamW(
#     learning_rate=0.1,
#     parameters=[{
#         'params': linear_1.parameters()
#     }, {
#         'params': linear_2.parameters(),
#         'weight_decay': 0.001,
#         'learning_rate': 0.1,
#         'beta1': 0.8
#     }],
#     weight_decay=0.01,
#     beta1=0.9)
# out.backward()
# adam.step()
# adam.clear_grad()

# value = np.arange(26).reshape(2, 13).astype("float32")
# a = paddle.to_tensor(value)
# linear = paddle.nn.Linear(13, 5)

# adam = paddle.optimizer.Adam(learning_rate = 0.01,
#                             parameters = linear.parameters())
# out = linear(a)
# out.backward()
# adam.step()
# adam.clear_grad()

# emb = paddle.nn.Embedding(10, 3)

# adam = paddle.optimizer.Adam(0.01, parameters = emb.parameters())
# for batch in range(10):
#     input = paddle.randint(low=0, high=5, shape=[5])
#     out = emb(input)
#     out.backward()
#     print("Learning rate of step{}: {}".format(batch, adam.get_lr())) 
#     adam.step()

# scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.5, step_size=2, gamma=0.1)
# adam = paddle.optimizer.Adam(scheduler, parameters = emb.parameters())
# for batch in range(10):
#     input = paddle.randint(low=0, high=5, shape=[5])
#     out = emb(input)
#     out.backward()
#     print("Learning rate of step{}: {}".format(batch, adam.get_lr())) 
#     adam.step()
#     scheduler.step()

# paddle.enable_static()
# main_prog = paddle.static.Program()
# start_prog = paddle.static.Program()
# with paddle.static.program_guard(main_prog, start_prog):
#     x = paddle.static.data(name='x', shape=[None, 10])
#     z = paddle.static.nn.fc(x, 100)
#     loss = paddle.mean(z)
#     scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.5, step_size=2, gamma=0.1)
#     adam = paddle.optimizer.Adam(learning_rate=scheduler)
#     adam.minimize(loss)

# exe = paddle.static.Executor()
# exe.run(start_prog)
# for batch in range(10):
#     print("Learning rate of step{}: {}", adam.get_lr())     
#     out = exe.run(main_prog, feed={'x': np.random.randn(3, 10).astype('float32')})
#     scheduler.step()

# linear = paddle.nn.Linear(10, 10)
# input = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
# out = linear(input)
# loss = paddle.mean(out)

# beta1 = paddle.to_tensor([0.9], dtype="float32")
# beta2 = paddle.to_tensor([0.99], dtype="float32")

# adam = paddle.optimizer.Adam(learning_rate=0.1,
#         parameters=linear.parameters(),
#         weight_decay=0.01)
# out.backward()
# adam.minimize(loss)
# adam.clear_grad()

# linear = paddle.nn.Linear(10, 10)

# adam = paddle.optimizer.Adam(0.1, parameters=linear.parameters())

# lr_list = [0.2, 0.3, 0.4, 0.5, 0.6]
# for i in range(5):
#     adam.set_lr(lr_list[i])
#     lr = adam.get_lr()
#     print("current lr is {}".format(lr))

# emb = paddle.nn.Embedding(10, 10)

# layer_state_dict = emb.state_dict()
# paddle.save(layer_state_dict, "emb.pdparams")

# scheduler = paddle.optimizer.lr.NoamDecay(
#     d_model=0.01, warmup_steps=100, verbose=True)
# adam = paddle.optimizer.Adam(
#     learning_rate=scheduler,
#     parameters=emb.parameters())
# opt_state_dict = adam.state_dict()
# paddle.save(opt_state_dict, "adam.pdopt")

# opti_state_dict = paddle.load("adam.pdopt")
# adam.set_state_dict(opti_state_dict)

# emb = paddle.nn.Embedding(10, 10)

# adam = paddle.optimizer.Adam(0.001, parameters=emb.parameters())
# state_dict = adam.state_dict()

# a = paddle.rand([2,13], dtype="float32")
# linear = paddle.nn.Linear(13, 5)

# adam = paddle.optimizer.Adam(learning_rate = 0.01,
#                             parameters = linear.parameters())
# out = linear(a)
# out.backward()
# adam.step()
# adam.clear_grad()

# inp = paddle.uniform(shape=[10, 10], dtype='float32', min=-0.1, max=0.1)
# linear = paddle.nn.Linear(10, 10)
# out = linear(inp)
# loss = paddle.mean(out)
# beta1 = paddle.to_tensor([0.9], dtype="float32")
# beta2 = paddle.to_tensor([0.85], dtype="float32")
# lamb = paddle.optimizer.Lamb(learning_rate=0.002, parameters=linear.parameters(), lamb_weight_decay=0.01)
# back = out.backward()
# lamb.step()
# lamb.clear_grad()

# value = np.arange(26).reshape(2, 13).astype("float32")
# a = paddle.to_tensor(value)
# linear = paddle.nn.Linear(13, 5)

# adam = paddle.optimizer.Adam(learning_rate = 0.01,
#                             parameters = linear.parameters())
# out = linear(a)
# out.backward()
# adam.step()
# adam.clear_grad()

# emb = paddle.nn.Embedding(10, 3)

# adam = paddle.optimizer.Adam(0.01, parameters = emb.parameters())
# for batch in range(10):
#     input = paddle.randint(low=0, high=5, shape=[5])
#     out = emb(input)
#     out.backward()
#     print("Learning rate of step{}: {}".format(batch, adam.get_lr())) 
#     adam.step()

# scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.5, step_size=2, gamma=0.1)
# adam = paddle.optimizer.Adam(scheduler, parameters = emb.parameters())
# for batch in range(10):
#     input = paddle.randint(low=0, high=5, shape=[5])
#     out = emb(input)
#     out.backward()
#     print("Learning rate of step{}: {}".format(batch, adam.get_lr())) 
#     adam.step()
#     scheduler.step()

# paddle.enable_static()
# main_prog = paddle.static.Program()
# start_prog = paddle.static.Program()
# with paddle.static.program_guard(main_prog, start_prog):
#     x = paddle.static.data(name='x', shape=[None, 10])
#     z = paddle.static.nn.fc(x, 100)
#     loss = paddle.mean(z)
#     scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.5, step_size=2, gamma=0.1)
#     adam = paddle.optimizer.Adam(learning_rate=scheduler)
#     adam.minimize(loss)

# exe = paddle.static.Executor()
# exe.run(start_prog)
# for batch in range(10):
#     print("Learning rate of step{}: {}", adam.get_lr())     
#     out = exe.run(main_prog, feed={'x': np.random.randn(3, 10).astype('float32')})
#     scheduler.step()

# linear = paddle.nn.Linear(10, 10)
# input = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
# out = linear(input)
# loss = paddle.mean(out)

# beta1 = paddle.to_tensor([0.9], dtype="float32")
# beta2 = paddle.to_tensor([0.99], dtype="float32")

# adam = paddle.optimizer.Adam(learning_rate=0.1,
#         parameters=linear.parameters(),
#         weight_decay=0.01)
# out.backward()
# adam.minimize(loss)
# adam.clear_grad()

# linear = paddle.nn.Linear(10, 10)

# adam = paddle.optimizer.Adam(0.1, parameters=linear.parameters())

# lr_list = [0.2, 0.3, 0.4, 0.5, 0.6]
# for i in range(5):
#     adam.set_lr(lr_list[i])
#     lr = adam.get_lr()
#     print("current lr is {}".format(lr))

# emb = paddle.nn.Embedding(10, 10)

# layer_state_dict = emb.state_dict()
# paddle.save(layer_state_dict, "emb.pdparams")

# scheduler = paddle.optimizer.lr.NoamDecay(
#     d_model=0.01, warmup_steps=100, verbose=True)
# adam = paddle.optimizer.Adam(
#     learning_rate=scheduler,
#     parameters=emb.parameters())
# opt_state_dict = adam.state_dict()
# paddle.save(opt_state_dict, "adam.pdopt")

# opti_state_dict = paddle.load("adam.pdopt")
# adam.set_state_dict(opti_state_dict)

# emb = paddle.nn.Embedding(10, 10)

# adam = paddle.optimizer.Adam(0.001, parameters=emb.parameters())
# state_dict = adam.state_dict()

# value = np.arange(26).reshape(2, 13).astype("float32")
# a = paddle.to_tensor(value)
# linear = paddle.nn.Linear(13, 5)

# adam = paddle.optimizer.Adam(learning_rate = 0.01,
#                             parameters = linear.parameters())
# out = linear(a)
# out.backward()
# adam.step()
# adam.clear_grad()

# inp = np.random.uniform(-0.1, 0.1, [10, 10]).astype("float32")
# linear = paddle.nn.Linear(10, 10)
# inp = paddle.to_tensor(inp)
# out = linear(inp)
# loss = paddle.mean(out)
# beta1 = paddle.to_tensor([0.9], dtype="float32")
# beta2 = paddle.to_tensor([0.99], dtype="float32")
# momentum = paddle.optimizer.Momentum(learning_rate=0.1, parameters=linear.parameters(), weight_decay=0.01)
# back = out.backward()
# momentum.step()
# momentum.clear_grad()

# linear_1 = paddle.nn.Linear(10, 10)
# linear_2 = paddle.nn.Linear(10, 10)
# inp = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
# out = linear_1(inp)
# out = linear_2(out)
# loss = paddle.mean(out)
# momentum = paddle.optimizer.Momentum(
#     learning_rate=0.1,
#     parameters=[{
#         'params': linear_1.parameters()
#     }, {
#         'params': linear_2.parameters(),
#         'weight_decay': 0.001,
#         'learning_rate': 0.1
#     }],
#     weight_decay=0.01,
#     momentum=0.9)
# out.backward()
# momentum.step()
# momentum.clear_grad()

# value = np.arange(26).reshape(2, 13).astype("float32")
# a = paddle.to_tensor(value)
# linear = paddle.nn.Linear(13, 5)

# adam = paddle.optimizer.Adam(learning_rate = 0.01,
#                             parameters = linear.parameters())
# out = linear(a)
# out.backward()
# adam.step()
# adam.clear_grad()

# emb = paddle.nn.Embedding(10, 3)

# adam = paddle.optimizer.Adam(0.01, parameters = emb.parameters())
# for batch in range(10):
#     input = paddle.randint(low=0, high=5, shape=[5])
#     out = emb(input)
#     out.backward()
#     print("Learning rate of step{}: {}".format(batch, adam.get_lr())) 
#     adam.step()

# scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.5, step_size=2, gamma=0.1)
# adam = paddle.optimizer.Adam(scheduler, parameters = emb.parameters())
# for batch in range(10):
#     input = paddle.randint(low=0, high=5, shape=[5])
#     out = emb(input)
#     out.backward()
#     print("Learning rate of step{}: {}".format(batch, adam.get_lr())) 
#     adam.step()
#     scheduler.step()

# paddle.enable_static()
# main_prog = paddle.static.Program()
# start_prog = paddle.static.Program()
# with paddle.static.program_guard(main_prog, start_prog):
#     x = paddle.static.data(name='x', shape=[None, 10])
#     z = paddle.static.nn.fc(x, 100)
#     loss = paddle.mean(z)
#     scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.5, step_size=2, gamma=0.1)
#     adam = paddle.optimizer.Adam(learning_rate=scheduler)
#     adam.minimize(loss)

# exe = paddle.static.Executor()
# exe.run(start_prog)
# for batch in range(10):
#     print("Learning rate of step{}: {}", adam.get_lr())     
#     out = exe.run(main_prog, feed={'x': np.random.randn(3, 10).astype('float32')})
#     scheduler.step()

# linear = paddle.nn.Linear(10, 10)
# input = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
# out = linear(input)
# loss = paddle.mean(out)

# beta1 = paddle.to_tensor([0.9], dtype="float32")
# beta2 = paddle.to_tensor([0.99], dtype="float32")

# adam = paddle.optimizer.Adam(learning_rate=0.1,
#         parameters=linear.parameters(),
#         weight_decay=0.01)
# out.backward()
# adam.minimize(loss)
# adam.clear_grad()

# linear = paddle.nn.Linear(10, 10)

# adam = paddle.optimizer.Adam(0.1, parameters=linear.parameters())

# lr_list = [0.2, 0.3, 0.4, 0.5, 0.6]
# for i in range(5):
#     adam.set_lr(lr_list[i])
#     lr = adam.get_lr()
#     print("current lr is {}".format(lr))

# emb = paddle.nn.Embedding(10, 10)

# layer_state_dict = emb.state_dict()
# paddle.save(layer_state_dict, "emb.pdparams")

# scheduler = paddle.optimizer.lr.NoamDecay(
#     d_model=0.01, warmup_steps=100, verbose=True)
# adam = paddle.optimizer.Adam(
#     learning_rate=scheduler,
#     parameters=emb.parameters())
# opt_state_dict = adam.state_dict()
# paddle.save(opt_state_dict, "adam.pdopt")

# opti_state_dict = paddle.load("adam.pdopt")
# adam.set_state_dict(opti_state_dict)

# emb = paddle.nn.Embedding(10, 10)

# adam = paddle.optimizer.Adam(0.001, parameters=emb.parameters())
# state_dict = adam.state_dict()

# value = np.arange(26).reshape(2, 13).astype("float32")
# a = paddle.to_tensor(value)
# linear = paddle.nn.Linear(13, 5)

# adam = paddle.optimizer.Adam(learning_rate = 0.01,
#                             parameters = linear.parameters())
# out = linear(a)
# out.backward()
# adam.step()
# adam.clear_grad()

# linear = paddle.nn.Linear(10, 10)
# inp = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
# out = linear(inp)
# loss = paddle.mean(out)
# adam = paddle.optimizer.Adam(learning_rate=0.1,
#         parameters=linear.parameters())
# out.backward()
# adam.step()
# adam.clear_grad()

# linear_1 = paddle.nn.Linear(10, 10)
# linear_2 = paddle.nn.Linear(10, 10)
# inp = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
# out = linear_1(inp)
# out = linear_2(out)
# loss = paddle.mean(out)
# sgd = paddle.optimizer.SGD(
#     learning_rate=0.1,
#     parameters=[{
#         'params': linear_1.parameters()
#     }, {
#         'params': linear_2.parameters(),
#         'weight_decay': 0.001,
#         'learning_rate': 0.1
#     }],
#     weight_decay=0.01)
# out.backward()
# sgd.step()
# sgd.clear_grad()

# emb = paddle.nn.Embedding(10, 10)

# adam = paddle.optimizer.Adam(0.001, parameters=emb.parameters())
# state_dict = adam.state_dict()

# emb = paddle.nn.Embedding(10, 10)

# layer_state_dict = emb.state_dict()
# paddle.save(layer_state_dict, "emb.pdparams")

# scheduler = paddle.optimizer.lr.NoamDecay(
#     d_model=0.01, warmup_steps=100, verbose=True)
# adam = paddle.optimizer.Adam(
#     learning_rate=scheduler,
#     parameters=emb.parameters())
# opt_state_dict = adam.state_dict()
# paddle.save(opt_state_dict, "adam.pdopt")

# opti_state_dict = paddle.load("adam.pdopt")
# adam.set_state_dict(opti_state_dict)

# linear = paddle.nn.Linear(10, 10)

# adam = paddle.optimizer.Adam(0.1, parameters=linear.parameters())

# lr_list = [0.2, 0.3, 0.4, 0.5, 0.6]
# for i in range(5):
#     adam.set_lr(lr_list[i])
#     lr = adam.get_lr()
#     print("current lr is {}".format(lr))

# emb = paddle.nn.Embedding(10, 3)

# adam = paddle.optimizer.Adam(0.01, parameters = emb.parameters())
# for batch in range(10):
#     input = paddle.randint(low=0, high=5, shape=[5])
#     out = emb(input)
#     out.backward()
#     print("Learning rate of step{}: {}".format(batch, adam.get_lr())) 
#     adam.step()

# scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.5, step_size=2, gamma=0.1)
# adam = paddle.optimizer.Adam(scheduler, parameters = emb.parameters())
# for batch in range(10):
#     input = paddle.randint(low=0, high=5, shape=[5])
#     out = emb(input)
#     out.backward()
#     print("Learning rate of step{}: {}".format(batch, adam.get_lr())) 
#     adam.step()
#     scheduler.step()

# paddle.enable_static()
# main_prog = paddle.static.Program()
# start_prog = paddle.static.Program()
# with paddle.static.program_guard(main_prog, start_prog):
#     x = paddle.static.data(name='x', shape=[None, 10])
#     z = paddle.static.nn.fc(x, 100)
#     loss = paddle.mean(z)
#     scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.5, step_size=2, gamma=0.1)
#     adam = paddle.optimizer.Adam(learning_rate=scheduler)
#     adam.minimize(loss)

# exe = paddle.static.Executor()
# exe.run(start_prog)
# for batch in range(10):
#     print("Learning rate of step{}: {}", adam.get_lr())     
#     out = exe.run(main_prog, feed={'x': np.random.randn(3, 10).astype('float32')})
#     scheduler.step()

# value = np.arange(26).reshape(2, 13).astype("float32")
# a = paddle.to_tensor(value)
# linear = paddle.nn.Linear(13, 5)

# adam = paddle.optimizer.Adam(learning_rate = 0.01,
#                             parameters = linear.parameters())
# out = linear(a)
# out.backward()
# adam.step()
# adam.clear_grad()

# linear = paddle.nn.Linear(10, 10)
# input = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
# out = linear(input)
# loss = paddle.mean(out)

# beta1 = paddle.to_tensor([0.9], dtype="float32")
# beta2 = paddle.to_tensor([0.99], dtype="float32")

# adam = paddle.optimizer.Adam(learning_rate=0.1,
#         parameters=linear.parameters(),
#         weight_decay=0.01)
# out.backward()
# adam.minimize(loss)
# adam.clear_grad()

# value = np.arange(26).reshape(2, 13).astype("float32")
# a = paddle.to_tensor(value)
# linear = paddle.nn.Linear(13, 5)

# adam = paddle.optimizer.Adam(learning_rate = 0.01,
#                             parameters = linear.parameters())
# out = linear(a)
# out.backward()
# adam.step()
# adam.clear_grad()

# inp = paddle.rand([10,10], dtype="float32")
# linear = paddle.nn.Linear(10, 10)
# out = linear(inp)
# loss = paddle.mean(out)

# rmsprop = paddle.optimizer.RMSProp(learning_rate=0.1,
#                  parameters=linear.parameters(),
#                            weight_decay=0.01)
# out.backward()
# rmsprop.step()
# rmsprop.clear_grad()

# linear_1 = paddle.nn.Linear(10, 10)
# linear_2 = paddle.nn.Linear(10, 10)
# inp = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
# out = linear_1(inp)
# out = linear_2(out)
# loss = paddle.mean(out)
# rmsprop = paddle.optimizer.RMSProp(
#     learning_rate=0.1,
#     parameters=[{
#         'params': linear_1.parameters()
#     }, {
#         'params': linear_2.parameters(),
#         'weight_decay': 0.001,
#         'learning_rate': 0.1
#     }],
#     weight_decay=0.01)
# out.backward()
# rmsprop.step()
# rmsprop.clear_grad()

# value = np.arange(26).reshape(2, 13).astype("float32")
# a = paddle.to_tensor(value)
# linear = paddle.nn.Linear(13, 5)

# adam = paddle.optimizer.Adam(learning_rate = 0.01,
#                             parameters = linear.parameters())
# out = linear(a)
# out.backward()
# adam.step()
# adam.clear_grad()

# emb = paddle.nn.Embedding(10, 3)

# adam = paddle.optimizer.Adam(0.01, parameters = emb.parameters())
# for batch in range(10):
#     input = paddle.randint(low=0, high=5, shape=[5])
#     out = emb(input)
#     out.backward()
#     print("Learning rate of step{}: {}".format(batch, adam.get_lr())) 
#     adam.step()

# scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.5, step_size=2, gamma=0.1)
# adam = paddle.optimizer.Adam(scheduler, parameters = emb.parameters())
# for batch in range(10):
#     input = paddle.randint(low=0, high=5, shape=[5])
#     out = emb(input)
#     out.backward()
#     print("Learning rate of step{}: {}".format(batch, adam.get_lr())) 
#     adam.step()
#     scheduler.step()

# paddle.enable_static()
# main_prog = paddle.static.Program()
# start_prog = paddle.static.Program()
# with paddle.static.program_guard(main_prog, start_prog):
#     x = paddle.static.data(name='x', shape=[None, 10])
#     z = paddle.static.nn.fc(x, 100)
#     loss = paddle.mean(z)
#     scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.5, step_size=2, gamma=0.1)
#     adam = paddle.optimizer.Adam(learning_rate=scheduler)
#     adam.minimize(loss)

# exe = paddle.static.Executor()
# exe.run(start_prog)
# for batch in range(10):
#     print("Learning rate of step{}: {}", adam.get_lr())     
#     out = exe.run(main_prog, feed={'x': np.random.randn(3, 10).astype('float32')})
#     scheduler.step()

# linear = paddle.nn.Linear(10, 10)
# input = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
# out = linear(input)
# loss = paddle.mean(out)

# beta1 = paddle.to_tensor([0.9], dtype="float32")
# beta2 = paddle.to_tensor([0.99], dtype="float32")

# adam = paddle.optimizer.Adam(learning_rate=0.1,
#         parameters=linear.parameters(),
#         weight_decay=0.01)
# out.backward()
# adam.minimize(loss)
# adam.clear_grad()

# linear = paddle.nn.Linear(10, 10)

# adam = paddle.optimizer.Adam(0.1, parameters=linear.parameters())

# lr_list = [0.2, 0.3, 0.4, 0.5, 0.6]
# for i in range(5):
#     adam.set_lr(lr_list[i])
#     lr = adam.get_lr()
#     print("current lr is {}".format(lr))

# emb = paddle.nn.Embedding(10, 10)

# layer_state_dict = emb.state_dict()
# paddle.save(layer_state_dict, "emb.pdparams")

# scheduler = paddle.optimizer.lr.NoamDecay(
#     d_model=0.01, warmup_steps=100, verbose=True)
# adam = paddle.optimizer.Adam(
#     learning_rate=scheduler,
#     parameters=emb.parameters())
# opt_state_dict = adam.state_dict()
# paddle.save(opt_state_dict, "adam.pdopt")

# opti_state_dict = paddle.load("adam.pdopt")
# adam.set_state_dict(opti_state_dict)

# emb = paddle.nn.Embedding(10, 10)

# adam = paddle.optimizer.Adam(0.001, parameters=emb.parameters())
# state_dict = adam.state_dict()

# value = np.arange(26).reshape(2, 13).astype("float32")
# a = paddle.to_tensor(value)
# linear = paddle.nn.Linear(13, 5)

# adam = paddle.optimizer.Adam(learning_rate = 0.01,
#                             parameters = linear.parameters())
# out = linear(a)
# out.backward()
# adam.step()
# adam.clear_grad()

# inp = np.random.uniform(-0.1, 0.1, [10, 10]).astype("float32")
# linear = paddle.nn.Linear(10, 10)
# inp = paddle.to_tensor(inp)
# out = linear(inp)
# loss = paddle.mean(out)
# beta1 = paddle.to_tensor([0.9], dtype="float32")
# beta2 = paddle.to_tensor([0.99], dtype="float32")
# sgd = paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters(), weight_decay=0.01)
# back = out.backward()
# sgd.step()
# sgd.clear_grad()

# value = np.arange(26).reshape(2, 13).astype("float32")
# a = paddle.to_tensor(value)
# linear = paddle.nn.Linear(13, 5)

# adam = paddle.optimizer.Adam(learning_rate = 0.01,
#                             parameters = linear.parameters())
# out = linear(a)
# out.backward()
# adam.step()
# adam.clear_grad()

# emb = paddle.nn.Embedding(10, 3)

# adam = paddle.optimizer.Adam(0.01, parameters = emb.parameters())
# for batch in range(10):
#     input = paddle.randint(low=0, high=5, shape=[5])
#     out = emb(input)
#     out.backward()
#     print("Learning rate of step{}: {}".format(batch, adam.get_lr())) 
#     adam.step()

# scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.5, step_size=2, gamma=0.1)
# adam = paddle.optimizer.Adam(scheduler, parameters = emb.parameters())
# for batch in range(10):
#     input = paddle.randint(low=0, high=5, shape=[5])
#     out = emb(input)
#     out.backward()
#     print("Learning rate of step{}: {}".format(batch, adam.get_lr())) 
#     adam.step()
#     scheduler.step()

# paddle.enable_static()
# main_prog = paddle.static.Program()
# start_prog = paddle.static.Program()
# with paddle.static.program_guard(main_prog, start_prog):
#     x = paddle.static.data(name='x', shape=[None, 10])
#     z = paddle.static.nn.fc(x, 100)
#     loss = paddle.mean(z)
#     scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.5, step_size=2, gamma=0.1)
#     adam = paddle.optimizer.Adam(learning_rate=scheduler)
#     adam.minimize(loss)

# exe = paddle.static.Executor()
# exe.run(start_prog)
# for batch in range(10):
#     print("Learning rate of step{}: {}", adam.get_lr())     
#     out = exe.run(main_prog, feed={'x': np.random.randn(3, 10).astype('float32')})
#     scheduler.step()

# linear = paddle.nn.Linear(10, 10)
# input = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
# out = linear(input)
# loss = paddle.mean(out)

# beta1 = paddle.to_tensor([0.9], dtype="float32")
# beta2 = paddle.to_tensor([0.99], dtype="float32")

# adam = paddle.optimizer.Adam(learning_rate=0.1,
#         parameters=linear.parameters(),
#         weight_decay=0.01)
# out.backward()
# adam.minimize(loss)
# adam.clear_grad()

# linear = paddle.nn.Linear(10, 10)

# adam = paddle.optimizer.Adam(0.1, parameters=linear.parameters())

# lr_list = [0.2, 0.3, 0.4, 0.5, 0.6]
# for i in range(5):
#     adam.set_lr(lr_list[i])
#     lr = adam.get_lr()
#     print("current lr is {}".format(lr))

# emb = paddle.nn.Embedding(10, 10)

# layer_state_dict = emb.state_dict()
# paddle.save(layer_state_dict, "emb.pdparams")

# scheduler = paddle.optimizer.lr.NoamDecay(
#     d_model=0.01, warmup_steps=100, verbose=True)
# adam = paddle.optimizer.Adam(
#     learning_rate=scheduler,
#     parameters=emb.parameters())
# opt_state_dict = adam.state_dict()
# paddle.save(opt_state_dict, "adam.pdopt")

# opti_state_dict = paddle.load("adam.pdopt")
# adam.set_state_dict(opti_state_dict)

# emb = paddle.nn.Embedding(10, 10)

# adam = paddle.optimizer.Adam(0.001, parameters=emb.parameters())
# state_dict = adam.state_dict()

# value = np.arange(26).reshape(2, 13).astype("float32")
# a = paddle.to_tensor(value)
# linear = paddle.nn.Linear(13, 5)

# adam = paddle.optimizer.Adam(learning_rate = 0.01,
#                             parameters = linear.parameters())
# out = linear(a)
# out.backward()
# adam.step()
# adam.clear_grad()

# linear = paddle.nn.Linear(10, 10)
# inp = paddle.rand(shape=[10, 10], dtype="float32")
# out = linear(inp)
# loss = paddle.mean(out)
# beta1 = paddle.to_tensor([0.9], dtype="float32")
# beta2 = paddle.to_tensor([0.99], dtype="float32")
# momentum = paddle.optimizer.Momentum(
#     learning_rate=0.1,
#     parameters=linear.parameters(),
#     weight_decay=L1Decay(0.0001))
# back = out.backward()
# momentum.step()
# momentum.clear_grad()

# my_conv2d = Conv2D(
#         in_channels=10,
#         out_channels=10,
#         kernel_size=1,
#         stride=1,
#         padding=0,
#         weight_attr=ParamAttr(regularizer=L2Decay(coeff=0.01)),
#         bias_attr=False)

# linear = paddle.nn.Linear(10, 10)
# inp = paddle.rand(shape=[10, 10], dtype="float32")
# out = linear(inp)
# loss = paddle.mean(out)
# beta1 = paddle.to_tensor([0.9], dtype="float32")
# beta2 = paddle.to_tensor([0.99], dtype="float32")
# momentum = paddle.optimizer.Momentum(
#     learning_rate=0.1,
#     parameters=linear.parameters(),
#     weight_decay=L2Decay(0.0001))
# back = out.backward()
# momentum.step()
# momentum.clear_grad()

# my_conv2d = Conv2D(
#         in_channels=10,
#         out_channels=10,
#         kernel_size=1,
#         stride=1,
#         padding=0,
#         weight_attr=ParamAttr(regularizer=L2Decay(coeff=0.01)),
#         bias_attr=False)

# paddle.seed(0)

# x = paddle.randn([8, 48000], dtype=paddle.float64)
# y = stft(x, n_fft=512)  

# x_ = istft(y, n_fft=512)  

# np.allclose(x, x_)  

# x = paddle.randn([8, 48000], dtype=paddle.float64)
# y1 = stft(x, n_fft=512)  
# y2 = stft(x, n_fft=512, onesided=False)  

# x = paddle.randn([8, 48000], dtype=paddle.float64) +                     paddle.randn([8, 48000], dtype=paddle.float64)*1j  
# y1 = stft(x, n_fft=512, center=False, onesided=False)  

# paddle.enable_static()
# data = static.data(name="input", shape=[-1, 32, 32], dtype="float32")
# label = static.data(name="label", shape=[-1,1], dtype="int")
# fc_out = static.nn.fc(x=data, size=10)
# predict = F.softmax(x=fc_out)
# result = static.accuracy(input=predict, label=label, k=5)

# place = paddle.CPUPlace()
# exe = static.Executor(place)

# exe.run(static.default_startup_program())
# x = np.random.rand(3, 32, 32).astype("float32")
# y = np.array([[1],[0],[1]])
# output= exe.run(feed={"input": x,"label": y},
#             fetch_list=[result[0]])
# print(output)

# paddle.enable_static()

# x = paddle.static.data(name='x', shape=[None, 13], dtype='int64')
# y = paddle.static.data(name='y', shape=[None, 1], dtype='float32')
# x_emb = paddle.static.nn.embedding(x, size=[100, 256])
# y_predict = paddle.static.nn.fc(x=x_emb, size=1, activation=None, name='my_fc')
# loss = F.square_error_cost(input=y_predict, label=y)
# avg_loss = paddle.mean(loss)

# all_weights = [param for param in paddle.static.default_main_program().block(0).all_parameters() if 'w_' in param.name]
# all_weights_name = [w.name for w in all_weights]

# p_g_list1 = paddle.static.append_backward(loss=avg_loss)

# p_g_list2 = paddle.static.append_backward(loss=avg_loss, parameter_list=all_weights)

# p_g_list3 = paddle.static.append_backward(loss=avg_loss, parameter_list=all_weights_name)

# p_g_list4 = paddle.static.append_backward(loss=avg_loss, no_grad_set=set([x_emb]))

# p_g_list5 = paddle.static.append_backward(loss=avg_loss, no_grad_set=set(['my_fc.b_0']))

# p_g_list6 = paddle.static.append_backward(loss=avg_loss, parameter_list=all_weights, no_grad_set=set(all_weights))

# paddle.enable_static()
# data = static.data(name="input", shape=[-1, 32,32], dtype="float32")
# label = static.data(name="label", shape=[-1], dtype="int")
# fc_out = static.nn.fc(x=data, size=2)
# predict = F.softmax(x=fc_out)
# result = static.auc(input=predict, label=label)

# place = paddle.CPUPlace()
# exe = static.Executor(place)

# exe.run(static.default_startup_program())
# x = np.random.rand(3,32,32).astype("float32")
# y = np.array([1,0,1])
# output= exe.run(feed={"input": x,"label": y},
#             fetch_list=[result[0]])
# print(output)

# paddle.enable_static()

# os.environ['CPU_NUM'] = str(2)
# places = static.cpu_places()

# data = static.data(name="x", shape=[None, 1], dtype="float32")
# hidden = static.nn.fc(input=data, size=10)
# loss = paddle.mean(hidden)
# paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)

# build_strategy = static.BuildStrategy()
# build_strategy.enable_inplace = True
# build_strategy.memory_optimize = True
# build_strategy.reduce_strategy = static.BuildStrategy.ReduceStrategy.Reduce
# program = static.CompiledProgram(static.default_main_program())
# program = program.with_data_parallel(loss_name=loss.name,
#                                       build_strategy=build_strategy,
#                                       places=places)

# paddle.enable_static()

# build_strategy = static.BuildStrategy()
# build_strategy.debug_graphviz_path = "./graph"

# paddle.enable_static()

# build_strategy = static.BuildStrategy()
# build_strategy.enable_auto_fusion = True

# paddle.enable_static()

# build_strategy = static.BuildStrategy()
# build_strategy.enable_sequential_execution = True

# paddle.enable_static()

# build_strategy = static.BuildStrategy()
# build_strategy.fuse_bn_act_ops = True

# paddle.enable_static()

# build_strategy = static.BuildStrategy()
# build_strategy.fuse_bn_add_act_ops = True

# paddle.enable_static()

# build_strategy = static.BuildStrategy()
# build_strategy.fuse_broadcast_ops = True

# paddle.enable_static()

# build_strategy = static.BuildStrategy()
# build_strategy.fuse_elewise_add_act_ops = True

# paddle.enable_static()

# build_strategy = static.BuildStrategy()
# build_strategy.fuse_relu_depthwise_conv = True

# paddle.enable_static()

# use_cuda = True
# place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
# exe = static.Executor(place)

# if not use_cuda:
#     os.environ['CPU_NUM'] = str(2)
#     places = static.cpu_places()
# else:
#     places = static.cuda_places()

# data = static.data(name='X', shape=[None, 1], dtype='float32')
# hidden = static.nn.fc(input=data, size=10)
# loss = paddle.mean(hidden)
# paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)

# exe.run(static.default_startup_program())

# build_strategy = static.BuildStrategy()
# build_strategy.gradient_scale_strategy = \
#           static.BuildStrategy.GradientScaleStrategy.Customized
# compiled_prog = static.CompiledProgram(
#           static.default_main_program()).with_data_parallel(
#                   loss_name=loss.name, build_strategy=build_strategy,
#                   places=places)

# dev_count =  len(places)
# x = numpy.random.random(size=(10, 1)).astype('float32')
# loss_grad = numpy.ones((dev_count)).astype("float32") * 0.01
# loss_grad_name = loss.name+"@GRAD"
# loss_data = exe.run(compiled_prog,
#                       feed={"X": x, loss_grad_name : loss_grad},
#                       fetch_list=[loss.name, loss_grad_name])

# paddle.enable_static()

# build_strategy = static.BuildStrategy()
# build_strategy.memory_optimize = True

# paddle.enable_static()

# build_strategy = static.BuildStrategy()
# build_strategy.reduce_strategy = static.BuildStrategy.ReduceStrategy.Reduce

# paddle.enable_static()

# build_strategy = static.BuildStrategy()
# build_strategy.remove_unnecessary_lock = True

# paddle.enable_static()

# build_strategy = static.BuildStrategy()
# build_strategy.sync_batch_norm = True

# paddle.enable_static()

# place = paddle.CUDAPlace(0) 
# exe = static.Executor(place)

# data = static.data(name='X', shape=[None, 1], dtype='float32')
# hidden = static.nn.fc(x=data, size=10)
# loss = paddle.mean(hidden)
# paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)

# exe.run(static.default_startup_program())
# compiled_prog = static.CompiledProgram(
#     static.default_main_program())

# x = numpy.random.random(size=(10, 1)).astype('float32')
# loss_data, = exe.run(compiled_prog,
#                     feed={"X": x},
#                     fetch_list=[loss.name])

# paddle.enable_static()

# use_cuda = True
# place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
# parallel_places = [paddle.CUDAPlace(0), paddle.CUDAPlace(1)] if use_cuda else [paddle.CPUPlace()] * 2

# if not use_cuda:
#     os.environ['CPU_NUM'] = str(2)

# exe = static.Executor(place)

# data = static.data(name='X', shape=[None, 1], dtype='float32')
# hidden = static.nn.fc(x=data, size=10)
# loss = paddle.mean(hidden)

# test_program = static.default_main_program().clone(for_test=True)
# paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)

# exe.run(static.default_startup_program())
# compiled_train_prog = static.CompiledProgram(
#     static.default_main_program()).with_data_parallel(
#             loss_name=loss.name, places=parallel_places)

# compiled_test_prog = static.CompiledProgram(
#     test_program).with_data_parallel(
#             share_vars_from=compiled_train_prog,
#             places=parallel_places)

# train_data = numpy.random.random(size=(10, 1)).astype('float32')
# loss_data, = exe.run(compiled_train_prog,
#                 feed={"X": train_data},
#                 fetch_list=[loss.name])
# test_data = numpy.random.random(size=(10, 1)).astype('float32')
# loss_data, = exe.run(compiled_test_prog,
#                 feed={"X": test_data},
#                 fetch_list=[loss.name])

# paddle.enable_static()

# cpu_places = static.cpu_places()

# paddle.enable_static()
# var = paddle.static.create_global_var(shape=[2,3], value=1.0, dtype='float32',
#                                persistable=True, force_cpu=True, name='new_var')

# paddle.enable_static()

# cuda_places = static.cuda_places()

# paddle.enable_static()

# x = paddle.static.data(name='x', shape=[3, 2, 1])

# y = paddle.static.data(name='y', shape=[-1, 2, 1], dtype='float32')

# z = x + y

# feed_data = np.ones(shape=[3, 2, 1], dtype=np.float32)

# exe = paddle.static.Executor(paddle.framework.CPUPlace())
# out = exe.run(paddle.static.default_main_program(),
#               feed={
#                   'x': feed_data,
#                   'y': feed_data
#               },
#               fetch_list=[z.name])

# print(out)

# paddle.enable_static()

# x = paddle.static.data(name='x', shape=[100, 100], dtype='float32')
# y = paddle.static.data(name='x', shape=[100, 100], dtype='float32')
# out = paddle.add(x, y)

# print(paddle.static.default_main_program().num_blocks) 

# print(paddle.static.default_main_program())

# paddle.enable_static()
# x = paddle.static.data(name="x", shape=[-1, 784], dtype='float32')
# out = paddle.static.nn.fc(name="fc", x=x, size=10, activation="relu")
# print("main program is: {}".format(paddle.static.default_main_program()))
# print("start up program is: {}".format(paddle.static.default_startup_program()))

# paddle.enable_static()

# path_prefix = "./infer_model"

# image = paddle.static.data(name='img', shape=[None, 28, 28], dtype='float32')
# label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
# predict = paddle.static.nn.fc(image, 10, activation='softmax')

# loss = paddle.nn.functional.cross_entropy(predict, label)

# exe = paddle.static.Executor(paddle.CPUPlace())
# exe.run(paddle.static.default_startup_program())

# serialized_params = paddle.static.serialize_persistables([image], [predict], exe)

# main_program = paddle.static.default_main_program()
# deserialized_params = paddle.static.deserialize_persistables(main_program, serialized_params, exe)

# paddle.enable_static()

# path_prefix = "./infer_model"

# image = paddle.static.data(name='img', shape=[None, 28, 28], dtype='float32')
# label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
# predict = paddle.static.nn.fc(image, 10, activation='softmax')

# loss = paddle.nn.functional.cross_entropy(predict, label)

# exe = paddle.static.Executor(paddle.CPUPlace())
# exe.run(paddle.static.default_startup_program())

# serialized_program = paddle.static.serialize_program([image], [predict])

# deserialized_program = paddle.static.deserialize_program(serialized_program)

# paddle.enable_static()
# support_gpu = paddle.is_compiled_with_cuda()
# place = paddle.CPUPlace()
# if support_gpu:
#     place = paddle.CUDAPlace(0)

# data1 = paddle.full(shape=[1, 3, 8, 8], fill_value=0.5, dtype='float32')
# data2 = paddle.full(shape=[1, 3, 64], fill_value=0.5, dtype='float32')
# shape = paddle.shape(data2)

# with paddle.static.device_guard("cpu"):
    
#     shape = paddle.slice(shape, axes=[0], starts=[0], ends=[4])
# with paddle.static.device_guard('gpu'):
    
#     out = paddle.reshape(data1, shape=shape)

# exe = paddle.static.Executor(place)
# exe.run(paddle.static.default_startup_program())
# result = exe.run(fetch_list=[out])

# paddle.enable_static()

# x = static.data(name='x', shape=[None, 13], dtype='float32')
# y = static.data(name='y', shape=[None, 1], dtype='float32')
# y_predict = static.nn.fc(input=x, size=1, act=None)

# cost = F.square_error_cost(input=y_predict, label=y)
# avg_loss = paddle.mean(cost)

# sgd_optimizer = paddle.optimizer.SGD(learning_rate=0.001)
# sgd_optimizer.minimize(avg_loss)

# exec_strategy = static.ExecutionStrategy()
# exec_strategy.num_threads = 4

# train_exe = static.ParallelExecutor(use_cuda=False,
#                                     loss_name=avg_loss.name,
#                                     exec_strategy=exec_strategy)

# paddle.enable_static()

# exec_strategy = static.ExecutionStrategy()
# exec_strategy.num_iteration_per_drop_scope = 10

# paddle.enable_static()

# exec_strategy = static.ExecutionStrategy()
# exec_strategy.num_iteration_per_run = 10

# paddle.enable_static()

# exec_strategy = static.ExecutionStrategy()
# exec_strategy.num_threads = 4

# paddle.enable_static()

# exe = paddle.static.Executor()

# train_program = paddle.static.Program()
# startup_program = paddle.static.Program()
# with paddle.static.program_guard(train_program, startup_program):
#     data = paddle.static.data(name='X', shape=[None, 1], dtype='float32')
#     hidden = paddle.static.nn.fc(data, 10)
#     loss = paddle.mean(hidden)
#     paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)

# exe.run(startup_program)

# x = numpy.random.random(size=(10, 1)).astype('float32')
# loss_data, = exe.run(train_program, feed={"X": x}, fetch_list=[loss.name])

# os.environ['CPU_NUM'] = str(2)

# compiled_prog = paddle.static.CompiledProgram(
#     train_program).with_data_parallel(loss_name=loss.name)
# loss_data, = exe.run(compiled_prog, feed={"X": x}, fetch_list=[loss.name])

# cpu = paddle.CPUPlace()
# exe = paddle.static.Executor(cpu)

# exe.close()

# paddle.enable_static()
# place = paddle.CPUPlace()  
# exe = paddle.static.Executor(place)

# data = paddle.static.data(name='X', shape=[None, 1], dtype='float32')
# hidden = paddle.static.nn.fc(data, 10)
# loss = paddle.mean(hidden)
# adam = paddle.optimizer.Adam()
# adam.minimize(loss)
# i = paddle.zeros(shape=[1], dtype='int64')
# array = paddle.fluid.layers.array_write(x=loss, i=i)

# exe.run(paddle.static.default_startup_program())

# x = numpy.random.random(size=(10, 1)).astype('float32')
# loss_val, array_val = exe.run(feed={'X': x},
#                               fetch_list=[loss.name, array.name])
# print(array_val)

# paddle.enable_static()
# place = paddle.CUDAPlace(0)
# exe = paddle.static.Executor(place)

# data = paddle.static.data(name='X', shape=[None, 1], dtype='float32')
# class_dim = 2
# prediction = paddle.static.nn.fc(data, class_dim)
# loss = paddle.mean(prediction)
# adam = paddle.optimizer.Adam()
# adam.minimize(loss)

# exe.run(paddle.static.default_startup_program())
# build_strategy = paddle.static.BuildStrategy()
# binary = paddle.static.CompiledProgram(
#     paddle.static.default_main_program()).with_data_parallel(
#         loss_name=loss.name, build_strategy=build_strategy)
# batch_size = 6
# x = np.random.random(size=(batch_size, 1)).astype('float32')

# unmerged_prediction, = exe.run(binary,
#                                feed={'X': x},
#                                fetch_list=[prediction.name],
#                                return_merged=False)

# print("The unmerged prediction shape: {}".format(
#     np.array(unmerged_prediction).shape))
# print(unmerged_prediction)

# merged_prediction, = exe.run(binary,
#                              feed={'X': x},
#                              fetch_list=[prediction.name],
#                              return_merged=True)

# print("The merged prediction shape: {}".format(
#     np.array(merged_prediction).shape))
# print(merged_prediction)

# paddle.enable_static()
# place = paddle.CPUPlace()  
# exe = paddle.static.Executor(place)
# x = paddle.static.data(name="x", shape=[None, 10, 10], dtype="int64")
# y = paddle.static.data(name="y", shape=[None, 1], dtype="int64", lod_level=1)
# dataset = paddle.fluid.DatasetFactory().create_dataset()
# dataset.set_use_var([x, y])
# dataset.set_thread(1)

# filelist = []
# dataset.set_filelist(filelist)
# exe.run(paddle.static.default_startup_program())
# exe.infer_from_dataset(program=paddle.static.default_main_program(),
#                        dataset=dataset)

# paddle.enable_static()
# place = paddle.CPUPlace() 
# exe = paddle.static.Executor(place)
# x = paddle.static.data(name="x", shape=[None, 10, 10], dtype="int64")
# y = paddle.static.data(name="y", shape=[None, 1], dtype="int64", lod_level=1)
# dataset = paddle.fluid.DatasetFactory().create_dataset()
# dataset.set_use_var([x, y])
# dataset.set_thread(1)

# filelist = []
# dataset.set_filelist(filelist)
# exe.run(paddle.static.default_startup_program())
# exe.train_from_dataset(program=paddle.static.default_main_program(),
#                        dataset=dataset)

# paddle.enable_static()

# data = static.data(name='x', shape=[-1, 5], dtype='float32')
# hidden = static.nn.fc(x=data, size=10)
# cost = paddle.mean(hidden)

# test_program = static.default_main_program().clone(for_test=True)
# optimizer = paddle.optimizer.Adam(learning_rate=0.001)
# optimizer.minimize(cost)

# ema = ExponentialMovingAverage(0.999)
# ema.update()

# place = paddle.CPUPlace()
# exe = static.Executor(place)
# exe.run(static.default_startup_program())

# for pass_id in range(3):
#     for batch_id in range(6):
#         data = numpy.random.random(size=(10, 5)).astype('float32')
#         exe.run(program=static.default_main_program(),
#         feed={'x': data},
#         fetch_list=[cost.name])

    
#     with ema.apply(exe):
#         data = numpy.random.random(size=(10, 5)).astype('float32')
#         exe.run(program=test_program,
#             feed={'x': data},
#             fetch_list=[hidden.name])

    
#     with ema.apply(exe, need_restore=False):
#         data = numpy.random.random(size=(10, 5)).astype('float32')
#         exe.run(program=test_program,
#             feed={'x': data},
#             fetch_list=[hidden.name])
#     ema.restore(exe)

# paddle.static.global_scope().var("data").get_tensor().set(numpy.ones((2, 2)), paddle.CPUPlace())
# numpy.array(paddle.static.global_scope().find_var("data").get_tensor())

# paddle.enable_static()

# x = paddle.static.data(name='x', shape=[None, 2, 8, 8], dtype='float32')
# x.stop_gradient=False
# y = paddle.static.nn.conv2d(x, 4, 1, bias_attr=False)
# y = F.relu(y)
# z = paddle.static.gradients([y], x)
# print(z) 

# input = InputSpec([None, 784], 'float32', 'x')
# label = InputSpec([None, 1], 'int64', 'label')

# print(input)  
# print(label)  

# paddle.disable_static()

# x = paddle.to_tensor(np.ones([2, 2], np.float32))
# x_spec = InputSpec.from_tensor(x, name='x')
# print(x_spec)  

# x = np.ones([2, 2], np.float32)
# x_spec = InputSpec.from_numpy(x, name='x')
# print(x_spec)  

# x_spec = InputSpec(shape=[64], dtype='float32', name='x')
# x_spec.batch(4)
# print(x_spec) 

# x_spec = InputSpec(shape=[4, 64], dtype='float32', name='x')
# x_spec.unbatch()
# print(x_spec) 

# paddle.enable_static()

# x = static.data(name="x", shape=[10, 10], dtype='float32')
# y = static.nn.fc(x, 10)
# z = static.nn.fc(y, 10)

# place = paddle.CPUPlace()
# exe = static.Executor(place)
# exe.run(static.default_startup_program())
# prog = static.default_main_program()

# static.save(prog, "./temp")
# static.load(prog, "./temp")

# paddle.enable_static()

# startup_prog = paddle.static.default_startup_program()
# main_prog = paddle.static.default_main_program()
# with paddle.static.program_guard(main_prog, startup_prog):
#     image = paddle.static.data(name="img", shape=[64, 784])
#     w = paddle.create_parameter(shape=[784, 200], dtype='float32')
#     b = paddle.create_parameter(shape=[200], dtype='float32')
#     hidden_w = paddle.matmul(x=image, y=w)
#     hidden_b = paddle.add(hidden_w, b)
# exe = paddle.static.Executor(paddle.CPUPlace())
# exe.run(startup_prog)

# path_prefix = "./infer_model"
# paddle.static.save_inference_model(path_prefix, [image], [hidden_b], exe)

# [inference_program, feed_target_names, fetch_targets] = (
#     paddle.static.load_inference_model(path_prefix, exe))
# tensor_img = np.array(np.random.random((64, 784)), dtype=np.float32)
# results = exe.run(inference_program,
#               feed={feed_target_names[0]: tensor_img},
#               fetch_list=fetch_targets)

# paddle.enable_static()

# x = static.data(name="x", shape=[10, 10], dtype='float32')
# y = static.nn.fc(x, 10)
# z = static.nn.fc(y, 10)

# place = paddle.CPUPlace()
# exe = static.Executor(place)
# exe.run(static.default_startup_program())
# prog = static.default_main_program()

# static.save(prog, "./temp")
# program_state = static.load_program_state("./temp")

# paddle.enable_static()
# with paddle.static.name_scope("s1"):
#    a = paddle.static.data(name='data', shape=[None, 1], dtype='int32')
#    b = a + 1
#    with paddle.static.name_scope("s2"):
#       c = b * 1
#    with paddle.static.name_scope("s3"):
#       d = c / 1
# with paddle.static.name_scope("s1"):
#       f = paddle.tensor.pow(d, 2.0)
# with paddle.static.name_scope("s4"):
#       g = f - 1

# for op in paddle.static.default_main_program().block(0).ops:
    
#     if op.type == 'elementwise_add':
#         assert op.desc.attr("op_namescope") == '/s1/'
    
#     elif op.type == 'elementwise_mul':
#         assert op.desc.attr("op_namescope") == '/s1/s2/'
    
#     elif op.type == 'elementwise_div':
#         assert op.desc.attr("op_namescope") == '/s1/s3/'
    
#     elif op.type == 'elementwise_sub':
#         assert op.desc.attr("op_namescope") == '/s4/'
    
#     elif op.type == 'pow':
#         assert op.desc.attr("op_namescope") == '/s1_1/'

# paddle.enable_static()

# path_prefix = "./infer_model"

# image = paddle.static.data(name='img', shape=[None, 28, 28], dtype='float32')
# label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
# predict = paddle.static.nn.fc(image, 10, activation='softmax')

# loss = paddle.nn.functional.cross_entropy(predict, label)

# exe = paddle.static.Executor(paddle.CPUPlace())
# exe.run(paddle.static.default_startup_program())

# program = paddle.static.default_main_program()
# normalized_program = paddle.static.normalize_program(program, [image], [predict])

# use_cuda = True
# paddle.enable_static()
# place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()

# if not use_cuda:
#     os.environ['CPU_NUM'] = str(2)

# exe = paddle.static.Executor(place)

# train_program = paddle.static.Program()
# startup_program = paddle.static.Program()
# with paddle.static.program_guard(train_program, startup_program):
#     data = paddle.static.data(name='X', shape=[None, 1], dtype='float32')
#     hidden = paddle.static.nn.fc(data, 10)
#     loss = paddle.mean(hidden)
#     test_program = paddle.static.default_main_program().clone(for_test=True)
#     paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)

# exe.run(startup_program)

# train_exe = paddle.static.ParallelExecutor(use_cuda=use_cuda,
#                                            main_program=train_program,
#                                            loss_name=loss.name)

# test_exe = paddle.static.ParallelExecutor(use_cuda=use_cuda,
#                                           main_program=test_program,
#                                           share_vars_from=train_exe)

# x = numpy.random.random(size=(10, 1)).astype('float32')
# loss_data, = train_exe.run(feed={"X": x},
#                            fetch_list=[loss.name])

# loss_data, = test_exe.run(feed={"X": x},
#                           fetch_list=[loss.name])

# use_cuda = True
# paddle.enable_static()
# place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()

# if not use_cuda:
#     os.environ['CPU_NUM'] = str(2)

# exe = paddle.static.Executor(place)

# train_program = paddle.static.Program()
# startup_program = paddle.static.Program()
# with paddle.static.program_guard(train_program, startup_program):
#     data = paddle.static.data(name='X', shape=[None, 1], dtype='float32')
#     hidden = paddle.static.nn.fc(data, 10)
#     loss = paddle.mean(hidden)
#     paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)

# exe.run(startup_program)

# train_exe = paddle.static.ParallelExecutor(use_cuda=use_cuda,
#                                            main_program=train_program,
#                                            loss_name=loss.name)

# x = numpy.random.random(size=(10, 1)).astype('float32')
# loss_data, = train_exe.run(feed={"X": x},
#                            fetch_list=[loss.name])

# x2 = numpy.random.random(size=(9, 1)).astype('float32')
# loss_data, = train_exe.run(feed=[{"X": x}, {"X": x2}],
#                            fetch_list=[loss.name])

# use_cuda = True

# if not use_cuda:
#     os.environ['CPU_NUM'] = str(2)

# paddle.enable_static()
# train_program = paddle.static.Program()
# startup_program = paddle.static.Program()
# with paddle.static.program_guard(train_program, startup_program):
#     data = paddle.static.data(name='X', shape=[None, 1], dtype='float32')
#     hidden = paddle.static.nn.fc(data, 10)
#     loss = paddle.mean(hidden)

# place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
# exe = paddle.static.Executor(place)
# exe.run(startup_program)

# parallel_exe = paddle.static.ParallelExecutor(use_cuda=use_cuda,
#                                               main_program=train_program,
#                                               loss_name=loss.name)

# x = numpy.random.random(size=(10, 1)).astype('float32')
# loss_data, = parallel_exe.run(feed={"X": x},
#                               fetch_list=[loss.name])

# parallel_exe.drop_local_exe_scopes()

# paddle.enable_static()

# x = paddle.full(shape=[2, 3], fill_value=3, dtype='int64')
# out = paddle.static.Print(x, message="The content of input layer:")

# main_program = paddle.static.default_main_program()
# exe = paddle.static.Executor(place=paddle.CPUPlace())
# res = exe.run(main_program, fetch_list=[out])

# paddle.enable_static()

# main_program = static.Program()
# startup_program = static.Program()
# with static.program_guard(main_program=main_program, startup_program=startup_program):
#     x = static.data(name="x", shape=[-1, 784], dtype='float32')
#     y = static.data(name="y", shape=[-1, 1], dtype='int32')
#     z = static.nn.fc(name="fc", x=x, size=10, activation="relu")

# print("main program is: {}".format(main_program))
# print("start up program is: {}".format(startup_program))

# paddle.enable_static()

# prog = static.default_main_program()
# print(prog.random_seed)

# prog.global_seed(102)
# prog1 = static.default_main_program()
# print(prog1.random_seed)

# paddle.enable_static()

# prog = static.default_main_program()
# x = static.data(name="X", shape=[2,3], dtype="float32")
# pred = static.nn.fc(x, size=3)
# prog_string = prog.to_string(throw_on_error=True, with_details=False)
# prog_string_with_details = prog.to_string(throw_on_error=False, with_details=True)
# print("program string without detail: {}".format(prog_string))
# print("program string with detail: {}".format(prog_string_with_details))

# paddle.enable_static()

# img = static.data(name='image', shape=[None, 784])
# pred = static.nn.fc(x=img, size=10, actvation='relu')
# loss = paddle.mean(pred)

# test_program = static.default_main_program().clone(for_test=True)
# optimizer = paddle.optimizer.Momentum(learning_rate=0.01, momentum=0.9)
# optimizer.minimize(loss)

# def print_prog(prog):
#     for name, value in sorted(six.iteritems(prog.block(0).vars)):
#         print(value)
#     for op in prog.block(0).ops:
#         print("op type is {}".format(op.type))
#         print("op inputs are {}".format(op.input_arg_names))
#         print("op outputs are {}".format(op.output_arg_names))
#         for key, value in sorted(six.iteritems(op.all_attrs())):
#             if key not in ['op_callstack', 'op_role_var']:
#                 print(" [ attrs: {}:   {} ]".format(key, value))

# paddle.enable_static()

# def print_prog(prog):
#     for name, value in sorted(six.iteritems(prog.block(0).vars)):
#         print(value)
#     for op in prog.block(0).ops:
#         print("op type is {}".format(op.type))
#         print("op inputs are {}".format(op.input_arg_names))
#         print("op outputs are {}".format(op.output_arg_names))
#         for key, value in sorted(six.iteritems(op.all_attrs())):
#             if key not in ['op_callstack', 'op_role_var']:
#                 print(" [ attrs: {}:   {} ]".format(key, value))

# train_program = static.Program()
# startup_program = static.Program()

# with static.program_guard(train_program, startup_program):
#     with utils.unique_name.guard():
#         img = static.data(name='image', shape=[None, 784])
#         hidden = static.nn.fc(x=img, size=200, activation='relu')
#         hidden = F.dropout(hidden, p=0.5)
#         loss = F.cross_entropy(
#             input=static.nn.fc(x=hidden, size=10, activation='softmax'),
#             label=static.data(name='label', shape=[1], dtype='int64'))
#         avg_loss = paddle.mean(loss)
#         test_program = train_program.clone(for_test=True)
# print_prog(test_program)

# with static.program_guard(train_program, startup_program):
#     with utils.unique_name.guard():
#         sgd = paddle.optimizer.SGD(learning_rate=1e-3)
#         sgd.minimize(avg_loss)

# paddle.enable_static()

# def print_prog(prog):
#     for name, value in sorted(six.iteritems(prog.block(0).vars)):
#         print(value)
#     for op in prog.block(0).ops:
#         print("op type is {}".format(op.type))
#         print("op inputs are {}".format(op.input_arg_names))
#         print("op outputs are {}".format(op.output_arg_names))
#         for key, value in sorted(six.iteritems(op.all_attrs())):
#             if key not in ['op_callstack', 'op_role_var']:
#                 print(" [ attrs: {}:   {} ]".format(key, value))

# def network():
#     img = static.data(name='image', shape=[None, 784])
#     hidden = static.nn.fc(x=img, size=200, activation='relu')
#     hidden = F.dropout(hidden, p=0.5)
#     loss = F.cross_entropy(
#         input=static.nn.fc(x=hidden, size=10, activation='softmax'),
#         label=static.data(name='label', shape=[1], dtype='int64'))
#     avg_loss = paddle.mean(loss)
#     return avg_loss

# train_program_2 = static.Program()
# startup_program_2 = static.Program()
# test_program_2 = static.Program()
# with static.program_guard(train_program_2, startup_program_2):
#     with utils.unique_name.guard():
#         avg_loss = network()
#         sgd = paddle.optimizer.SGD(learning_rate=1e-3)
#         sgd.minimize(avg_loss)

# with static.program_guard(test_program_2, startup_program_2):
#     with utils.unique_name.guard():
#         avg_loss = network()
# print_prog(test_program_2)

# paddle.enable_static()

# startup_prog = static.Program()
# main_prog = static.Program()
# with static.program_guard(startup_prog, main_prog):
#     x = static.data(name='X', shape=[1000, 784], dtype='float32')

#     y = static.data(name='Y', shape=[784, 100], dtype='float32')

#     z = paddle.matmul(x=x, y=y)

#     binary_str = static.default_main_program().desc.serialize_to_string()
#     prog_restored = static.default_main_program().parse_from_string(binary_str)

#     print(static.default_main_program())
#     print(prog_restored)

# paddle.enable_static()

# prog = static.default_main_program()
# num_blocks = prog.num_blocks
# print(num_blocks)

# paddle.enable_static()

# prog = static.default_main_program()
# random_seed = prog.random_seed
# x_var = static.data(name="X", shape=[3,3], dtype="float32")
# print(random_seed)

# prog.random_seed = 1
# z_var = F.dropout(x_var, 0.7)

# print(prog.random_seed)

# paddle.enable_static()

# prog = static.default_main_program()
# gb_block = prog.global_block()
# print(gb_block)

# paddle.enable_static()

# prog = static.default_main_program()
# block_0 = prog.block(0)
# print(block_0)

# paddle.enable_static()

# prog = static.default_main_program()
# current_blk = prog.current_block()
# print(current_blk)

# paddle.enable_static()

# prog = static.default_main_program()
# img = static.data(name='img', shape=[None, 1,28,28], dtype='float32')
# label = static.data(name='label', shape=[None,1], dtype='int64')
# for var in prog.list_vars():
#     print(var)

# paddle.enable_static()

# program = static.default_main_program()
# data = static.data(name='x', shape=[None, 13], dtype='float32')
# hidden = static.nn.fc(x=data, size=10)
# loss = paddle.mean(hidden)
# paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)

# for param in program.all_parameters():
#     print(param)

# paddle.enable_static()

# x = static.data(name="x", shape=[10, 10], dtype='float32')
# y = static.nn.fc(x, 10)
# z = static.nn.fc(y, 10)

# place = paddle.CPUPlace()
# exe = static.Executor(place)
# exe.run(static.default_startup_program())
# prog = static.default_main_program()

# path = "./temp/model.pdparams"
# paddle.save(prog.state_dict(), path)

# paddle.enable_static()

# x = static.data(name="x", shape=[10, 10], dtype='float32')
# y = static.nn.fc(x, 10)
# z = static.nn.fc(y, 10)

# place = paddle.CPUPlace()
# exe = static.Executor(place)
# exe.run(static.default_startup_program())
# prog = static.default_main_program()

# path = "./temp/model.pdparams"
# paddle.save(prog.state_dict(), path)
# state_dict_load = paddle.load(path)
# prog.set_state_dict(state_dict_load)

# paddle.enable_static()
# main_program = paddle.static.Program()
# startup_program = paddle.static.Program()
# with paddle.static.program_guard(main_program, startup_program):
#     data = paddle.static.data(name='image', shape=[None, 784, 784], dtype='float32')
#     hidden = paddle.static.nn.fc(x=data, size=10, activation='relu')

# paddle.enable_static()
# main_program = paddle.static.Program()

# with paddle.static.program_guard(main_program, paddle.static.Program()):
#     data = paddle.static.data(name='image', shape=[None, 784, 784], dtype='float32')

# paddle.enable_static()

# def tanh(x):
#     return np.tanh(x)

# def tanh_grad(y, dy):
#     return np.array(dy) * (1 - np.square(np.array(y)))

# def debug_func(x):
#     print(x)

# def create_tmp_var(name, dtype, shape):
#     return paddle.static.default_main_program().current_block().create_var(
#         name=name, dtype=dtype, shape=shape)

# def simple_net(img, label):
#     hidden = img
#     for idx in six.moves.range(4):
#         hidden = paddle.static.nn.fc(hidden, size=200)
#         new_hidden = create_tmp_var(name='hidden_{}'.format(idx),
#             dtype=hidden.dtype, shape=hidden.shape)

        
#         hidden = paddle.static.py_func(func=tanh, x=hidden,
#             out=new_hidden, backward_func=tanh_grad,
#             skip_vars_in_backward_input=hidden)

        
#         paddle.static.py_func(func=debug_func, x=hidden, out=None)

#     prediction = paddle.static.nn.fc(hidden, size=10, activation='softmax')
#     ce_loss = paddle.nn.loss.CrossEntropyLoss()
#     return ce_loss(prediction, label)

# x = paddle.static.data(name='x', shape=[1,4], dtype='float32')
# y = paddle.static.data(name='y', shape=[1,10], dtype='int64')
# res = simple_net(x, y)

# exe = paddle.static.Executor(paddle.CPUPlace())
# exe.run(paddle.static.default_startup_program())
# input1 = np.random.random(size=[1,4]).astype('float32')
# input2 = np.random.randint(1, 10, size=[1,10], dtype='int64')
# out = exe.run(paddle.static.default_main_program(),
#               feed={'x':input1, 'y':input2},
#               fetch_list=[res.name])
# print(out)

# paddle.enable_static()

# def element_wise_add(x, y):
    
    
#     x = np.array(x)
#     y = np.array(y)

#     if x.shape != y.shape:
#         raise AssertionError("the shape of inputs must be the same!")

#     result = np.zeros(x.shape, dtype='int32')
#     for i in range(len(x)):
#         for j in range(len(x[0])):
#             result[i][j] = x[i][j] + y[i][j]

#     return result

# def create_tmp_var(name, dtype, shape):
#     return paddle.static.default_main_program().current_block().create_var(
#                 name=name, dtype=dtype, shape=shape)

# def py_func_demo():
#     start_program = paddle.static.default_startup_program()
#     main_program = paddle.static.default_main_program()

    
#     x = paddle.static.data(name='x', shape=[2,3], dtype='int32')
#     y = paddle.static.data(name='y', shape=[2,3], dtype='int32')

    
#     output = create_tmp_var('output','int32', [3,1])

    
#     paddle.static.py_func(func=element_wise_add, x=[x,y], out=output)

#     exe=paddle.static.Executor(paddle.CPUPlace())
#     exe.run(start_program)

    
#     input1 = np.random.randint(1, 10, size=[2,3], dtype='int32')
#     input2 = np.random.randint(1, 10, size=[2,3], dtype='int32')
#     out = exe.run(main_program,
#                 feed={'x':input1, 'y':input2},
#                 fetch_list=[output.name])
#     print("{0} + {1} = {2}".format(input1, input2, out))

# py_func_demo()

# paddle.enable_static()

# x = static.data(name="x", shape=[10, 10], dtype='float32')
# y = static.nn.fc(x, 10)
# z = static.nn.fc(y, 10)

# place = paddle.CPUPlace()
# exe = static.Executor(place)
# exe.run(static.default_startup_program())
# prog = static.default_main_program()

# static.save(prog, "./temp")

# paddle.enable_static()

# path_prefix = "./infer_model"

# image = paddle.static.data(name='img', shape=[None, 28, 28], dtype='float32')
# label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
# predict = paddle.static.nn.fc(image, 10, activation='softmax')

# loss = paddle.nn.functional.cross_entropy(predict, label)

# exe = paddle.static.Executor(paddle.CPUPlace())
# exe.run(paddle.static.default_startup_program())

# paddle.static.save_inference_model(path_prefix, [image], [predict], exe)

# paddle.enable_static()

# new_scope = paddle.static.Scope()
# with paddle.static.scope_guard(new_scope):
#      paddle.static.global_scope().var("data").get_tensor().set(numpy.ones((2, 2)), paddle.CPUPlace())
# numpy.array(new_scope.find_var("data").get_tensor())

# paddle.enable_static()

# path_prefix = "./infer_model"

# image = paddle.static.data(name='img', shape=[None, 28, 28], dtype='float32')
# label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
# predict = paddle.static.nn.fc(image, 10, activation='softmax')

# loss = paddle.nn.functional.cross_entropy(predict, label)

# exe = paddle.static.Executor(paddle.CPUPlace())
# exe.run(paddle.static.default_startup_program())

# serialized_params = paddle.static.serialize_persistables([image], [predict], exe)

# main_program = paddle.static.default_main_program()
# deserialized_params = paddle.static.deserialize_persistables(main_program, serialized_params, exe)

# paddle.enable_static()

# path_prefix = "./infer_model"

# image = paddle.static.data(name='img', shape=[None, 28, 28], dtype='float32')
# label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
# predict = paddle.static.nn.fc(image, 10, activation='softmax')

# loss = paddle.nn.functional.cross_entropy(predict, label)

# exe = paddle.static.Executor(paddle.CPUPlace())
# exe.run(paddle.static.default_startup_program())

# serialized_program = paddle.static.serialize_program([image], [predict])

# deserialized_program = paddle.static.deserialize_program(serialized_program)

# paddle.enable_static()

# x = static.data(name="x", shape=[10, 10], dtype='float32')
# y = static.nn.fc(x, 10)
# z = static.nn.fc(y, 10)

# place = paddle.CPUPlace()
# exe = static.Executor(place)
# exe.run(static.default_startup_program())
# prog = static.default_main_program()

# static.save(prog, "./temp")
# program_state = static.load_program_state("./temp")

# static.set_program_state(prog, program_state)

# cur_program = fluid.Program()
# cur_block = cur_program.current_block()
# new_variable = cur_block.create_var(name="X",
#                                     shape=[-1, 23, 48],
#                                     dtype='float32')

# with fluid.dygraph.guard():
#     new_variable = fluid.dygraph.to_variable(np.arange(10))

# paddle.enable_static()

# x = paddle.static.data(name='x', shape=[3, 2, 1])

# y = x.detach()

# data = np.random.uniform(-1, 1, [30, 10, 32]).astype('float32')
# with fluid.dygraph.guard():
#     linear = Linear(32, 64)
#     data = to_variable(data)
#     x = linear(data)
#     print(x.numpy())

# paddle.disable_static()

# x = np.ones([2, 2], np.float32)
# inputs = []
# for _ in range(10):
#     tmp = paddle.to_tensor(x)
    
    
#     tmp.stop_gradient=False
#     inputs.append(tmp)
# ret = paddle.add_n(inputs)
# loss = paddle.sum(ret)
# loss.backward()

# x = np.ones([2, 2], np.float32)
# with fluid.dygraph.guard():
#     inputs2 = []
#     for _ in range(10):
#         tmp = fluid.dygraph.base.to_variable(x)
#         tmp.stop_gradient=False
#         inputs2.append(tmp)
#     ret2 = fluid.layers.sums(inputs2)
#     loss2 = fluid.layers.reduce_sum(ret2)
#     loss2.backward()
#     print(loss2.gradient())

# with fluid.dygraph.guard():
#     embedding = fluid.dygraph.Embedding(
#         size=[20, 32],
#         param_attr='emb.w',
#         is_sparse=True)
#     x_data = np.arange(12).reshape(4, 3).astype('int64')
#     x_data = x_data.reshape((-1, 3, 1))
#     x = fluid.dygraph.base.to_variable(x_data)
#     out = embedding(x)
#     out.backward()
#     print(embedding.weight.gradient())

# x = np.ones([2, 2], np.float32)
# with fluid.dygraph.guard():
#     inputs2 = []
#     for _ in range(10):
#         tmp = fluid.dygraph.base.to_variable(x)
#         tmp.stop_gradient=False
#         inputs2.append(tmp)
#     ret2 = fluid.layers.sums(inputs2)
#     loss2 = fluid.layers.reduce_sum(ret2)
#     loss2.backward()
#     print(loss2.gradient())
#     loss2.clear_gradient()
#     print("After clear {}".format(loss2.gradient()))

# paddle.enable_static()
# cur_program = fluid.Program()
# cur_block = cur_program.current_block()
# new_variable = cur_block.create_var(name="X",
#                                     shape=[-1, 23, 48],
#                                     dtype='float32')
# print(new_variable.to_string(True))
# print("=============with detail===============")
# print(new_variable.to_string(True, True))

# with fluid.dygraph.guard():
#     value0 = np.arange(26).reshape(2, 13).astype("float32")
#     value1 = np.arange(6).reshape(2, 3).astype("float32")
#     value2 = np.arange(10).reshape(2, 5).astype("float32")
#     linear = fluid.Linear(13, 5, dtype="float32")
#     linear2 = fluid.Linear(3, 3, dtype="float32")
#     a = fluid.dygraph.to_variable(value0)
#     b = fluid.dygraph.to_variable(value1)
#     c = fluid.dygraph.to_variable(value2)
#     out1 = linear(a)
#     out2 = linear2(b)
#     out1.stop_gradient = True
#     out = fluid.layers.concat(input=[out1, out2, c], axis=1)
#     out.backward()

#     assert linear.weight.gradient() is None
#     assert (out1.gradient() == 0).all()

# cur_program = fluid.Program()
# cur_block = cur_program.current_block()
# new_variable = cur_block.create_var(name="X",
#                                     shape=[-1, 23, 48],
#                                     dtype='float32')
# print("persistable of current Var is: {}".format(new_variable.persistable))

# new_parameter = paddle.static.create_parameter(name="X",
#                                     shape=[10, 23, 48],
#                                     dtype='float32')
# if new_parameter.is_parameter:
#     print("Current var is a Parameter")
# else:
#     print("Current var is not a Parameter")

# cur_program = fluid.Program()
# cur_block = cur_program.current_block()
# new_variable = cur_block.create_var(name="X",
#                                     shape=[-1, 23, 48],
#                                     dtype='float32')
# print("name of current Var is: {}".format(new_variable.name))

# cur_program = fluid.Program()
# cur_block = cur_program.current_block()
# new_variable = cur_block.create_var(name="X",
#                                     shape=[-1, 23, 48],
#                                     dtype='float32')
# print("shape of current Var is: {}".format(new_variable.shape))

# cur_program = fluid.Program()
# cur_block = cur_program.current_block()
# new_variable = cur_block.create_var(name="X",
#                                     shape=[-1, 23, 48],
#                                     dtype='float32')
# print("Dtype of current Var is: {}".format(new_variable.dtype))

# cur_program = fluid.Program()
# cur_block = cur_program.current_block()
# new_variable = cur_block.create_var(name="X",
#                                     shape=[-1, 23, 48],
#                                     dtype='float32')
# print("LoD Level of current Var is: {}".format(new_variable.lod_level))

# cur_program = fluid.Program()
# cur_block = cur_program.current_block()
# new_variable = cur_block.create_var(name="X",
#                                     shape=[-1, 23, 48],
#                                     dtype='float32')
# print("Type of current Var is: {}".format(new_variable.type))

# paddle.enable_static()

# x = paddle.ones(shape=[2, 3, 5])
# x_T = x.T

# exe = paddle.static.Executor()
# x_T_np = exe.run(paddle.static.default_main_program(), fetch_list=[x_T])[0]
# print(x_T_np.shape)

# paddle.enable_static()

# x = paddle.static.data(name='x', shape=[3, 2, 1])

# y = x.clone()

# paddle.enable_static()

# x = static.data(name="x", shape=[10, 10], dtype='float32')

# y = static.nn.fc(x, 10, name='fc')
# place = paddle.CPUPlace()
# exe = static.Executor(place)
# prog = paddle.static.default_main_program()
# exe.run(static.default_startup_program())
# inputs = np.ones((10, 10), dtype='float32')
# exe.run(prog, feed={'x': inputs}, fetch_list=[y, ])
# path = 'temp/tensor_'
# for var in prog.list_vars():
#     if var.persistable:
#         t = var.get_value()
#         paddle.save(t, path+var.name+'.pdtensor')

# for var in prog.list_vars():
#     if var.persistable:
#         t_load = paddle.load(path+var.name+'.pdtensor')
#         var.set_value(t_load)

# paddle.enable_static()

# x = static.data(name="x", shape=[10, 10], dtype='float32')

# y = static.nn.fc(x, 10, name='fc')
# place = paddle.CPUPlace()
# exe = static.Executor(place)
# prog = paddle.static.default_main_program()
# exe.run(static.default_startup_program())
# inputs = np.ones((10, 10), dtype='float32')
# exe.run(prog, feed={'x': inputs}, fetch_list=[y, ])
# path = 'temp/tensor_'
# for var in prog.list_vars():
#     if var.persistable:
#         t = var.get_value()
#         paddle.save(t, path+var.name+'.pdtensor')

# for var in prog.list_vars():
#     if var.persistable:
#         t_load = paddle.load(path+var.name+'.pdtensor')
#         var.set_value(t_load)

# paddle.enable_static()

# x = paddle.static.data(name='x', shape=[3, 2, 1])

# y = x.size()

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.abs(x)
# print(out)

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.acos(x)
# print(out)

# x = paddle.to_tensor([2, 3, 4], 'float64')
# y = paddle.to_tensor([1, 5, 2], 'float64')
# z = paddle.add(x, y)
# print(z)  

# input0 = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='float32')
# input1 = paddle.to_tensor([[7, 8, 9], [10, 11, 12]], dtype='float32')
# output = paddle.add_n([input0, input1])

# x = paddle.ones([2,2])
# y = paddle.ones([2,2])
# input = paddle.ones([2,2])

# out = paddle.addmm( input=input, x=x, y=y, beta=0.5, alpha=5.0 )

# print(out)

# x = paddle.assign(np.array([[1, 0], [1, 1]], dtype='int32'))
# print(x)
# x = paddle.cast(x, 'bool')

# out1 = paddle.all(x)  
# print(out1)

# out2 = paddle.all(x, axis=0)  
# print(out2)

# out3 = paddle.all(x, axis=-1)  
# print(out3)

# out4 = paddle.all(x, axis=1, keepdim=True)
# out4 = paddle.cast(out4, 'int32')  
# print(out4)

# x = paddle.to_tensor([10000., 1e-07])
# y = paddle.to_tensor([10000.1, 1e-08])
# result1 = paddle.allclose(x, y, rtol=1e-05, atol=1e-08,
#                         equal_nan=False, name="ignore_nan")
# np_result1 = result1.numpy()

# result2 = paddle.allclose(x, y, rtol=1e-05, atol=1e-08,
#                             equal_nan=True, name="equal_nan")
# np_result2 = result2.numpy()

# x = paddle.to_tensor([1.0, float('nan')])
# y = paddle.to_tensor([1.0, float('nan')])
# result1 = paddle.allclose(x, y, rtol=1e-05, atol=1e-08,
#                         equal_nan=False, name="ignore_nan")
# np_result1 = result1.numpy()

# result2 = paddle.allclose(x, y, rtol=1e-05, atol=1e-08,
#                             equal_nan=True, name="equal_nan")
# np_result2 = result2.numpy()

# x = paddle.assign(np.array([[1, 0], [1, 1]], dtype='int32'))
# print(x)
# x = paddle.cast(x, 'bool')

# out1 = paddle.any(x)  
# print(out1)

# out2 = paddle.any(x, axis=0)  
# print(out2)

# out3 = paddle.any(x, axis=-1)  
# print(out3)

# out4 = paddle.any(x, axis=1, keepdim=True)
# out4 = paddle.cast(out4, 'int32')  
# print(out4)

# x =  paddle.to_tensor([[5,8,9,5],
#                          [0,0,1,7],
#                          [6,9,2,4]])
# out1 = paddle.argmax(x)
# print(out1) 
# out2 = paddle.argmax(x, axis=1)
# print(out2)

# out3 = paddle.argmax(x, axis=-1)
# print(out3)

# x =  paddle.to_tensor([[5,8,9,5],
#                          [0,0,1,7],
#                          [6,9,2,4]])
# out1 = paddle.argmin(x)
# print(out1) 
# out2 = paddle.argmin(x, axis=1)
# print(out2)

# out3 = paddle.argmin(x, axis=-1)
# print(out3)

# x = paddle.to_tensor([[[5,8,9,5],
#                        [0,0,1,7],
#                        [6,9,2,4]],
#                       [[5,2,4,2],
#                        [4,7,7,9],
#                        [1,7,0,6]]],
#                     dtype='float32')
# out1 = paddle.argsort(x=x, axis=-1)
# out2 = paddle.argsort(x=x, axis=0)
# out3 = paddle.argsort(x=x, axis=1)
# print(out1)

# print(out2)

# print(out3)

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.asin(x)
# print(out)

# startup_prog = fluid.Program()
# main_prog = fluid.Program()
# with fluid.program_guard(startup_prog, main_prog):
#     original_variable = fluid.data(name = "new_variable", shape=[2,2], dtype='float32')
#     new_variable = original_variable.astype('int64')
#     print("new var's dtype is: {}".format(new_variable.dtype))

# x = np.ones([2, 2], np.float32)
# with fluid.dygraph.guard():
#     original_variable = fluid.dygraph.to_variable(x)
#     print("original var's dtype is: {}, numpy dtype is {}".format(original_variable.dtype, original_variable.numpy().dtype))
#     new_variable = original_variable.astype('int64')
#     print("new var's dtype is: {}, numpy dtype is {}".format(new_variable.dtype, new_variable.numpy().dtype))

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.atan(x)
# print(out)

# x = paddle.to_tensor([1, 2, 1, 4, 5])
# result1 = paddle.bincount(x)
# print(result1) 

# w = paddle.to_tensor([2.1, 0.4, 0.1, 0.5, 0.5])
# result2 = paddle.bincount(x, weights=w)
# print(result2) 

# x = paddle.to_tensor([-5, -1, 1])
# y = paddle.to_tensor([4,  2, -3])
# res = paddle.bitwise_and(x, y)
# print(res)  

# x = paddle.to_tensor([-5, -1, 1])
# res = paddle.bitwise_not(x)
# print(res) 

# x = paddle.to_tensor([-5, -1, 1])
# y = paddle.to_tensor([4,  2, -3])
# res = paddle.bitwise_or(x, y)
# print(res)  

# x = paddle.to_tensor([-5, -1, 1])
# y = paddle.to_tensor([4,  2, -3])
# res = paddle.bitwise_xor(x, y)
# print(res) 

# x = paddle.to_tensor([[[1.0, 1.0, 1.0],
#                     [2.0, 2.0, 2.0]],
#                     [[3.0, 3.0, 3.0],
#                     [4.0, 4.0, 4.0]]])
# y = paddle.to_tensor([[[1.0, 1.0],[2.0, 2.0],[3.0, 3.0]],
#                     [[4.0, 4.0],[5.0, 5.0],[6.0, 6.0]]])
# out = paddle.bmm(x, y)

# out_np = out.numpy()

# shape = paddle.broadcast_shape([2, 1, 3], [1, 3, 1])

# x1 = paddle.rand([1, 2, 3, 4]).astype('float32')
# x2 = paddle.rand([1, 2, 1, 4]).astype('float32')
# x3 = paddle.rand([1, 1, 3, 1]).astype('float32')
# out1, out2, out3 = paddle.broadcast_tensors(input=[x1, x2, x3])

# data = paddle.to_tensor([1, 2, 3], dtype='int32')
# out = paddle.broadcast_to(data, shape=[2, 3])
# print(out)

# x = paddle.to_tensor([2, 3, 4], 'float64')
# y = paddle.cast(x, 'uint8')

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.ceil(x)
# print(out)

# a = np.random.rand(3, 3)
# a_t = np.transpose(a, [1, 0])
# x_data = np.matmul(a, a_t) + 1e-03
# x = paddle.to_tensor(x_data)
# out = paddle.cholesky(x, upper=False)
# print(out)

# x_np = np.random.random([3, 9, 5]).astype("int32")
# x = paddle.to_tensor(x_np)

# out0, out1, out2 = paddle.chunk(x, chunks=3, axis=1)

# out0, out1, out2 = paddle.chunk(x, chunks=3, axis=-2)

# x1 = paddle.to_tensor([[1.2, 3.5], [4.5, 6.4]], 'float32')
# out1 = paddle.clip(x1, min=3.5, max=5.0)
# out2 = paddle.clip(x1, min=2.5)
# print(out1)

# print(out2)

# x1 = paddle.to_tensor([[1, 2, 3],
#                        [4, 5, 6]])
# x2 = paddle.to_tensor([[11, 12, 13],
#                        [14, 15, 16]])
# x3 = paddle.to_tensor([[21, 22],
#                        [23, 24]])
# zero = paddle.full(shape=[1], dtype='int32', fill_value=0)

# out1 = paddle.concat(x=[x1, x2, x3], axis=-1)
# out2 = paddle.concat(x=[x1, x2], axis=0)
# out3 = paddle.concat(x=[x1, x2], axis=zero)

# x = paddle.to_tensor([[1., 0, -1], [0, 1, 0], [1, 0, 1]])

# out = paddle.linalg.cond(x)

# out_fro = paddle.linalg.cond(x, p='fro')

# out_nuc = paddle.linalg.cond(x, p='nuc')

# out_1 = paddle.linalg.cond(x, p=1)

# out_minus_1 = paddle.linalg.cond(x, p=-1)

# out_2 = paddle.linalg.cond(x, p=2)

# out_minus_2 = paddle.linalg.cond(x, p=-2)

# out_inf = paddle.linalg.cond(x, p=np.inf)

# out_minus_inf = paddle.linalg.cond(x, p=-np.inf)

# a = paddle.to_tensor(np.random.randn(2, 4, 4).astype('float32'))

# a_cond_fro = paddle.linalg.cond(a, p='fro')

# b = paddle.to_tensor(np.random.randn(2, 3, 4).astype('float64'))

# b_cond_2 = paddle.linalg.cond(b, p=2)

# data=paddle.to_tensor([[1+1j, 2+2j, 3+3j], [4+4j, 5+5j, 6+6j]])

# conj_data=paddle.conj(data)

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.cos(x)
# print(out)

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.cosh(x)
# print(out)

# x = paddle.to_tensor([[1.0, 1.0, 1.0],
#                       [2.0, 2.0, 2.0],
#                       [3.0, 3.0, 3.0]])
# y = paddle.to_tensor([[1.0, 1.0, 1.0],
#                       [1.0, 1.0, 1.0],
#                       [1.0, 1.0, 1.0]])

# z1 = paddle.cross(x, y)

# z2 = paddle.cross(x, y, axis=1)

# data = paddle.arange(12)
# data = paddle.reshape(data, (3, 4))

# y = paddle.cumprod(data, dim=0)

# y = paddle.cumprod(data, dim=-1)

# y = paddle.cumprod(data, dim=1, dtype='float64')

# print(y.dtype)

# data = paddle.arange(12)
# data = paddle.reshape(data, (3, 4))

# y = paddle.cumsum(data)

# y = paddle.cumsum(data, axis=0)

# y = paddle.cumsum(data, axis=-1)

# y = paddle.cumsum(data, dtype='float64')
# print(y.dtype)

# x = paddle.rand([2,2,3],'float32')
# print(x)

# out1 = paddle.diagonal(x)
# print(out1)

# out2 = paddle.diagonal(x, offset=0, axis1=2, axis2=1)
# print(out2)

# out3 = paddle.diagonal(x, offset=1, axis1=0, axis2=1)
# print(out3)

# out4 = paddle.diagonal(x, offset=0, axis1=1, axis2=2)
# print(out4)

# data = paddle.to_tensor([[1, 1.5], [0, -2.2]], dtype='float32')
# res = paddle.digamma(data)
# print(res)

# x = paddle.to_tensor(np.array([[3, 3],[3, 3]]), "float32")
# y = paddle.to_tensor(np.array([[3, 3],[3, 1]]), "float32")
# out = paddle.dist(x, y, 0)
# print(out) 

# out = paddle.dist(x, y, 2)
# print(out) 

# out = paddle.dist(x, y, float("inf"))
# print(out) 

# out = paddle.dist(x, y, float("-inf"))
# print(out) 

# x = paddle.to_tensor([2, 3, 4], dtype='float64')
# y = paddle.to_tensor([1, 5, 2], dtype='float64')
# z = paddle.divide(x, y)
# print(z)  

# x_data = np.random.uniform(0.1, 1, [10]).astype(np.float32)
# y_data = np.random.uniform(1, 3, [10]).astype(np.float32)
# x = paddle.to_tensor(x_data)
# y = paddle.to_tensor(y_data)
# z = paddle.dot(x, y)
# print(z)

# paddle.device.set_device("cpu")

# x_data = np.array([[1.6707249, 7.2249975, 6.5045543],
#                    [9.956216,  8.749598,  6.066444 ],
#                    [4.4251957, 1.7983172, 0.370647 ]]).astype("float32")
# x = paddle.to_tensor(x_data)
# w, v = paddle.linalg.eig(x)
# print(w)

# print(v)

# paddle.set_device("cpu")
# paddle.seed(1234)

# x = paddle.rand(shape=[3, 3], dtype='float64')

# print(paddle.linalg.eigvals(x))

# x_data = np.array([[1, -2j], [2j, 5]])
# x = paddle.to_tensor(x_data)
# out_value = paddle.eigvalsh(x, UPLO='L')
# print(out_value)

# x = paddle.to_tensor([1, 2, 3])
# y = paddle.to_tensor([1, 3, 2])
# result1 = paddle.equal(x, y)
# print(result1)  

# x = paddle.to_tensor([1, 2, 3])
# y = paddle.to_tensor([1, 2, 3])
# z = paddle.to_tensor([1, 4, 3])
# result1 = paddle.equal_all(x, y)
# print(result1) 
# result2 = paddle.equal_all(x, z)
# print(result2) 

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.erf(x)
# print(out)

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.exp(x)
# print(out)

# data = paddle.to_tensor([1, 2, 3], dtype='int32')
# out = paddle.expand(data, shape=[2, 3])
# print(out)

# data_x = paddle.to_tensor([1, 2, 3], 'int32')
# data_y = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], 'int32')
# out = paddle.expand_as(data_x, data_y)
# np_out = out.numpy()

# image_shape=(2, 3, 4, 4)

# x = paddle.arange(end=image_shape[0] * image_shape[1] * image_shape[2] * image_shape[3])
# img = paddle.reshape(x, image_shape)

# out = paddle.flatten(img, start_axis=1, stop_axis=2)

# img[0, 0, 0, 0] = -1
# print(out[0, 0, 0]) 

# image_shape=(3, 2, 2)
# x = np.arange(image_shape[0] * image_shape[1] * image_shape[2]).reshape(image_shape)
# x = x.astype('float32')
# img = paddle.to_tensor(x)
# tmp = paddle.flip(img, [0,1])
# print(tmp) 

# out = paddle.flip(tmp,-1)
# print(out) 

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.floor(x)
# print(out)

# x = paddle.to_tensor([2, 3, 8, 7])
# y = paddle.to_tensor([1, 5, 3, 3])
# z = paddle.floor_divide(x, y)
# print(z)  

# x = paddle.to_tensor([2, 3, 8, 7])
# y = paddle.to_tensor([1, 5, 3, 3])
# z = paddle.remainder(x, y)
# print(z)  

# input = paddle.to_tensor([[1,2],[3,4],[5,6]])
# index = paddle.to_tensor([0,1])
# output = paddle.gather(input, index, axis=0)

# x = paddle.to_tensor([[[1, 2], [3, 4], [5, 6]],
#                       [[7, 8], [9, 10], [11, 12]]])
# index = paddle.to_tensor([[0, 1]])

# output = paddle.gather_nd(x, index) 

# x = paddle.to_tensor([1, 2, 3])
# y = paddle.to_tensor([1, 3, 2])
# result1 = paddle.greater_equal(x, y)
# print(result1)  

# x = paddle.to_tensor([1, 2, 3])
# y = paddle.to_tensor([1, 3, 2])
# result1 = paddle.greater_than(x, y)
# print(result1)  

# inputs = paddle.to_tensor([1, 2, 1])
# result = paddle.histogram(inputs, bins=4, min=0, max=3)
# print(result) 

# x = paddle.to_tensor(
#     [[1 + 6j, 2 + 5j, 3 + 4j], [4 + 3j, 5 + 2j, 6 + 1j]])

# imag_res = paddle.imag(x)

# imag_t = x.imag()

# data = paddle.zeros(shape=[1], dtype='float32')
# counter = paddle.increment(data)

# x = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0],
#                       [5.0, 6.0, 7.0, 8.0],
#                       [9.0, 10.0, 11.0, 12.0]], dtype='float32')
# index = paddle.to_tensor([[0, 1, 2],
#                           [1, 2, 3],
#                           [0, 0, 0]], dtype='int32')
# target = paddle.to_tensor([[100, 200, 300, 400],
#                            [500, 600, 700, 800],
#                            [900, 1000, 1100, 1200]], dtype='int32')
# out_z1 = paddle.index_sample(x, index)
# print(out_z1)

# top_value, top_index = paddle.topk(x, k=2)
# out_z2 = paddle.index_sample(target, top_index)
# print(top_value)

# print(top_index)

# print(out_z2)

# x = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0],
#                       [5.0, 6.0, 7.0, 8.0],
#                       [9.0, 10.0, 11.0, 12.0]])
# index = paddle.to_tensor([0, 1, 1], dtype='int32')
# out_z1 = paddle.index_select(x=x, index=index)

# out_z2 = paddle.index_select(x=x, index=index, axis=1)

# mat = paddle.to_tensor([[2, 0], [0, 2]], dtype='float32')
# inv = paddle.inverse(mat)
# print(inv) 

# input = paddle.rand(shape=[4, 32, 32], dtype='float32')
# res = paddle.is_empty(x=input)
# print("res:", res)

# input1 = paddle.rand(shape=[2, 3, 5], dtype='float32')
# check = paddle.is_tensor(input1)
# print(check)  

# input3 = [1, 4]
# check = paddle.is_tensor(input3)
# print(check)  

# x = paddle.to_tensor([float('-inf'), -2, 3.6, float('inf'), 0, float('-nan'), float('nan')])
# out = paddle.tensor.isfinite(x)
# print(out)  

# x = paddle.to_tensor([float('-inf'), -2, 3.6, float('inf'), 0, float('-nan'), float('nan')])
# out = paddle.tensor.isinf(x)
# print(out)  

# x = paddle.to_tensor([float('-inf'), -2, 3.6, float('inf'), 0, float('-nan'), float('nan')])
# out = paddle.tensor.isnan(x)
# print(out)  

# x = paddle.to_tensor([[1, 2], [3, 4]], dtype='int64')
# y = paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='int64')
# out = paddle.kron(x, y)
# print(out)

# x = paddle.to_tensor([1, 2, 3])
# y = paddle.to_tensor([1, 3, 2])
# result1 = paddle.less_equal(x, y)
# print(result1)  

# x = paddle.to_tensor([1, 2, 3])
# y = paddle.to_tensor([1, 3, 2])
# result1 = paddle.less_than(x, y)
# print(result1)  

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.lgamma(x)
# print(out)

# x = [[2,3,4], [7,8,9]]
# x = paddle.to_tensor(x, dtype='float32')
# res = paddle.log(x)

# x_i = paddle.to_tensor([[1.0], [10.0]])
# res = paddle.log10(x_i) 

# x_i = paddle.full(shape=[1], fill_value=10, dtype='float32')
# paddle.to_tensor(x_i)
# res = paddle.log10(x_i)
# print(res) 

# x_i = paddle.full(shape=[1], fill_value=10, dtype='float64')
# paddle.to_tensor(x_i)
# res = paddle.log10(x_i)
# print(res) 

# data = paddle.to_tensor([[0], [1]], dtype='float32')
# res = paddle.log1p(data)

# x_i = paddle.to_tensor([[1.0], [2.0]])
# res = paddle.log2(x_i) 

# x_i = paddle.full(shape=[1], fill_value=2, dtype='float32')
# paddle.to_tensor(x_i)
# res = paddle.log2(x_i)
# print(res) 

# x_i = paddle.full(shape=[1], fill_value=2, dtype='float64')
# paddle.to_tensor(x_i)
# res = paddle.log2(x_i)
# print(res) 

# x = paddle.to_tensor([True])
# y = paddle.to_tensor([True, False, True, False])
# res = paddle.logical_and(x, y)
# print(res) 

# x = paddle.to_tensor([True, False, True, False])
# res = paddle.logical_not(x)
# print(res) 

# x_data = np.array([True, False], dtype=np.bool).reshape(2, 1)
# y_data = np.array([True, False, True, False], dtype=np.bool).reshape(2, 2)
# x = paddle.to_tensor(x_data)
# y = paddle.to_tensor(y_data)
# res = paddle.logical_or(x, y)
# print(res) 

# x_data = np.array([True, False], dtype=np.bool).reshape([2, 1])
# y_data = np.array([True, False, True, False], dtype=np.bool).reshape([2, 2])
# x = paddle.to_tensor(x_data)
# y = paddle.to_tensor(y_data)
# res = paddle.logical_xor(x, y)
# print(res) 

# x = paddle.to_tensor([[-1.5, 0., 2.], [3., 1.2, -2.4]])
# out1 = paddle.logsumexp(x) 
# out2 = paddle.logsumexp(x, 1) 

# x = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0],
#                       [5.0, 6.0, 7.0, 8.0],
#                       [9.0, 10.0, 11.0, 12.0]])
# mask = paddle.to_tensor([[True, False, False, False],
#                          [True, True, False, False],
#                          [True, False, False, False]])
# out = paddle.masked_select(x, mask)

# x_data = np.random.random([10]).astype(np.float32)
# y_data = np.random.random([10]).astype(np.float32)
# x = paddle.to_tensor(x_data)
# y = paddle.to_tensor(y_data)
# z = paddle.matmul(x, y)
# print(z.numpy().shape)

# x_data = np.random.random([10, 5]).astype(np.float32)
# y_data = np.random.random([5]).astype(np.float32)
# x = paddle.to_tensor(x_data)
# y = paddle.to_tensor(y_data)
# z = paddle.matmul(x, y)
# print(z.numpy().shape)

# x_data = np.random.random([10, 5, 2]).astype(np.float32)
# y_data = np.random.random([2]).astype(np.float32)
# x = paddle.to_tensor(x_data)
# y = paddle.to_tensor(y_data)
# z = paddle.matmul(x, y)
# print(z.numpy().shape)

# x_data = np.random.random([10, 5, 2]).astype(np.float32)
# y_data = np.random.random([10, 2, 5]).astype(np.float32)
# x = paddle.to_tensor(x_data)
# y = paddle.to_tensor(y_data)
# z = paddle.matmul(x, y)
# print(z.numpy().shape)

# x_data = np.random.random([10, 1, 5, 2]).astype(np.float32)
# y_data = np.random.random([1, 3, 2, 5]).astype(np.float32)
# x = paddle.to_tensor(x_data)
# y = paddle.to_tensor(y_data)
# z = paddle.matmul(x, y)
# print(z.numpy().shape)

# x = paddle.to_tensor([[1, 2, 3],
#                       [1, 4, 9],
#                       [1, 8, 27]], dtype='float64')
# print(paddle.linalg.matrix_power(x, 2))

# print(paddle.linalg.matrix_power(x, 0))

# print(paddle.linalg.matrix_power(x, -2))

# x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
#                       [0.1, 0.2, 0.6, 0.7]])
# result1 = paddle.max(x)
# print(result1)

# result2 = paddle.max(x, axis=0)
# print(result2)

# result3 = paddle.max(x, axis=-1)
# print(result3)

# result4 = paddle.max(x, axis=1, keepdim=True)
# print(result4)

# y = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]],
#                       [[5.0, 6.0], [7.0, 8.0]]])
# result5 = paddle.max(y, axis=[1, 2])
# print(result5)

# result6 = paddle.max(y, axis=[0, 1])
# print(result6)

# x = paddle.to_tensor([[1, 2], [7, 8]])
# y = paddle.to_tensor([[3, 4], [5, 6]])
# res = paddle.maximum(x, y)
# print(res)

# x = paddle.to_tensor([[1, 2, 3], [1, 2, 3]])
# y = paddle.to_tensor([3, 0, 4])
# res = paddle.maximum(x, y)
# print(res)

# x = paddle.to_tensor([2, 3, 5], dtype='float32')
# y = paddle.to_tensor([1, np.nan, np.nan], dtype='float32')
# res = paddle.maximum(x, y)
# print(res)

# x = paddle.to_tensor([5, 3, np.inf], dtype='float32')
# y = paddle.to_tensor([1, -np.inf, 5], dtype='float32')
# res = paddle.maximum(x, y)
# print(res)

# x = paddle.to_tensor([[[1., 2., 3., 4.],
#                        [5., 6., 7., 8.],
#                        [9., 10., 11., 12.]],
#                       [[13., 14., 15., 16.],
#                        [17., 18., 19., 20.],
#                        [21., 22., 23., 24.]]])
# out1 = paddle.mean(x)

# out2 = paddle.mean(x, axis=-1)

# out3 = paddle.mean(x, axis=-1, keepdim=True)

# out4 = paddle.mean(x, axis=[0, 2])

# x = paddle.arange(12).reshape([3, 4])

# y1 = paddle.median(x)

# y2 = paddle.median(x, axis=0)

# y3 = paddle.median(x, axis=1)

# y4 = paddle.median(x, axis=0, keepdim=True)

# x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
#                       [0.1, 0.2, 0.6, 0.7]])
# result1 = paddle.min(x)
# print(result1)

# result2 = paddle.min(x, axis=0)
# print(result2)

# result3 = paddle.min(x, axis=-1)
# print(result3)

# result4 = paddle.min(x, axis=1, keepdim=True)
# print(result4)

# y = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]],
#                       [[5.0, 6.0], [7.0, 8.0]]])
# result5 = paddle.min(y, axis=[1, 2])
# print(result5)

# result6 = paddle.min(y, axis=[0, 1])
# print(result6)

# x = paddle.to_tensor([[1, 2], [7, 8]])
# y = paddle.to_tensor([[3, 4], [5, 6]])
# res = paddle.minimum(x, y)
# print(res)

# x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
# y = paddle.to_tensor([3, 0, 4])
# res = paddle.minimum(x, y)
# print(res)

# x = paddle.to_tensor([2, 3, 5], dtype='float32')
# y = paddle.to_tensor([1, np.nan, np.nan], dtype='float32')
# res = paddle.minimum(x, y)
# print(res)

# x = paddle.to_tensor([5, 3, np.inf], dtype='float64')
# y = paddle.to_tensor([1, -np.inf, 5], dtype='float64')
# res = paddle.minimum(x, y)
# print(res)

# input = paddle.arange(1, 7).reshape((3, 2)).astype('float32')
# mat2 = paddle.arange(1, 9).reshape((2, 4)).astype('float32')
# out = paddle.mm(input, mat2)
# print(out)

# x = paddle.to_tensor([2, 3, 8, 7])
# y = paddle.to_tensor([1, 5, 3, 3])
# z = paddle.remainder(x, y)
# print(z)  

# A_data = np.random.random([3, 4]).astype(np.float32)
# B_data = np.random.random([4, 5]).astype(np.float32)
# A = paddle.to_tensor(A_data)
# B = paddle.to_tensor(B_data)
# out = paddle.linalg.multi_dot([A, B])
# print(out.numpy().shape)

# A_data = np.random.random([10, 5]).astype(np.float32)
# B_data = np.random.random([5, 8]).astype(np.float32)
# C_data = np.random.random([8, 7]).astype(np.float32)
# A = paddle.to_tensor(A_data)
# B = paddle.to_tensor(B_data)
# C = paddle.to_tensor(C_data)
# out = paddle.linalg.multi_dot([A, B, C])
# print(out.numpy().shape)

# img1 = np.array([[1, 2], [3, 4]]).astype(np.float32)
# img2 = np.array([[5, 6], [7, 8]]).astype(np.float32)
# inputs = [paddle.to_tensor(img1), paddle.to_tensor(img2)]
# index = paddle.to_tensor(np.array([[1], [0]]).astype(np.int32))
# res = paddle.multiplex(inputs, index)
# print(res) 

# x = paddle.to_tensor([[1, 2], [3, 4]])
# y = paddle.to_tensor([[5, 6], [7, 8]])
# res = paddle.multiply(x, y)
# print(res) 

# x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
# y = paddle.to_tensor([2])
# res = paddle.multiply(x, y)
# print(res) 

# x_data = np.array([[2, 1, 3], [3, 0, 1]]).astype("float64")
# x = paddle.to_tensor(x_data)
# vec_data = np.array([3, 5, 1])
# vec = paddle.to_tensor(vec_data).astype("float64")
# out = paddle.mv(x, vec)

# paddle.enable_static()

# x = paddle.static.data(name='x', shape=[3, 2, 1])

# print(x.ndim)

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.neg(x)
# print(out)

# x1 = paddle.to_tensor([[1.0, 0.0, 0.0],
#                        [0.0, 2.0, 0.0],
#                        [0.0, 0.0, 3.0]])
# x2 = paddle.to_tensor([0.0, 1.0, 0.0, 3.0])
# out_z1 = paddle.nonzero(x1)
# print(out_z1)

# out_z1_tuple = paddle.nonzero(x1, as_tuple=True)
# for out in out_z1_tuple:
#     print(out)

# out_z2 = paddle.nonzero(x2)
# print(out_z2)

# out_z2_tuple = paddle.nonzero(x2, as_tuple=True)
# for out in out_z2_tuple:
#     print(out)

# shape=[2, 3, 4]
# np_input = np.arange(24).astype('float32') - 12
# np_input = np_input.reshape(shape)
# x = paddle.to_tensor(np_input)

# out_fro = paddle.norm(x, p='fro', axis=[0,1])

# out_pnorm = paddle.norm(x, p=2, axis=-1)

# out_pnorm = paddle.norm(x, p=2, axis=[0,1])

# out_pnorm = paddle.norm(x, p=np.inf)

# out_pnorm = paddle.norm(x, p=np.inf, axis=0)

# out_pnorm = paddle.norm(x, p=-np.inf)

# out_pnorm = paddle.norm(x, p=-np.inf, axis=0)

# x = paddle.to_tensor([1, 2, 3])
# y = paddle.to_tensor([1, 3, 2])
# result1 = paddle.not_equal(x, y)
# print(result1)  

# x = paddle.full(shape=[4, 5, 7], fill_value=0, dtype='int32')
# numel = paddle.numel(x) 

# x = paddle.to_tensor([1, 2, 3], dtype='float32')

# res = paddle.pow(x, 2)
# print(res)

# res = paddle.pow(x, 2.5)
# print(res)

# y = paddle.to_tensor([2], dtype='float32')
# res = paddle.pow(x, y)
# print(res)

# x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
#                       [0.1, 0.2, 0.6, 0.7]])
# out1 = paddle.prod(x)

# out2 = paddle.prod(x, -1)

# out3 = paddle.prod(x, 0)

# out4 = paddle.prod(x, 0, keepdim=True)

# out5 = paddle.prod(x, 0, dtype='int64')

# y = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]],
#                       [[5.0, 6.0], [7.0, 8.0]]])
# out6 = paddle.prod(y, [0, 1])

# out7 = paddle.prod(y, (1, 2))

# x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).astype('float64')
# q, r = paddle.linalg.qr(x)
# print (q)
# print (r)

# input = paddle.rand((3, 100, 100))
# rank = paddle.rank(input)
# print(rank)

# x = paddle.to_tensor(
#     [[1 + 6j, 2 + 5j, 3 + 4j], [4 + 3j, 5 + 2j, 6 + 1j]])

# real_res = paddle.real(x)

# real_t = x.real()

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.reciprocal(x)
# print(out)

# x = paddle.to_tensor([2, 3, 8, 7])
# y = paddle.to_tensor([1, 5, 3, 3])
# z = paddle.remainder(x, y)
# print(z)  

# x = paddle.rand([2, 4, 6], dtype="float32")
# positive_four = paddle.full([1], 4, "int32")

# out = paddle.reshape(x, [-1, 0, 3, 2])
# print(out)

# out = paddle.reshape(x, shape=[positive_four, 12])
# print(out)

# shape_tensor = paddle.to_tensor(np.array([8, 6]).astype("int32"))
# out = paddle.reshape(x, shape=shape_tensor)
# print(out)

# x[0, 0, 0] = 10.
# print(out[0, 0])

# image_shape=(3, 2, 2)
# x = np.arange(image_shape[0] * image_shape[1] * image_shape[2]).reshape(image_shape)
# x = x.astype('float32')
# img = paddle.to_tensor(x)
# tmp = paddle.flip(img, [0,1])
# print(tmp) 

# out = paddle.flip(tmp,-1)
# print(out) 

# x = paddle.to_tensor([[1.0, 2.0, 3.0],
#                       [4.0, 5.0, 6.0],
#                       [7.0, 8.0, 9.0]])
# out_z1 = paddle.roll(x, shifts=1)
# print(out_z1)

# out_z2 = paddle.roll(x, shifts=1, axis=0)
# print(out_z2)

# x = paddle.to_tensor([-0.5, -0.2, 0.6, 1.5])
# out = paddle.round(x)
# print(out)

# x = paddle.to_tensor([0.1, 0.2, 0.3, 0.4])
# out = paddle.rsqrt(x)
# print(out)

# data = paddle.randn(shape=[2,3], dtype='float32')
# res = paddle.scale(data, scale=2.0, bias=1.0)

# data = paddle.randn(shape=[2, 3], dtype='float32')
# factor = paddle.to_tensor([2], dtype='float32')
# res = paddle.scale(data, scale=factor, bias=1.0)

# x = np.array([[1, 1], [2, 2], [3, 3]])
# index = np.array([2, 1, 0, 1])

# updates = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
# overwrite = False

# if not overwrite:
#     for i in range(len(index)):
#         x[index[i]] = np.zeros((2))
# for i in range(len(index)):
#     if (overwrite):
#         x[index[i]] = updates[i]
#     else:
#         x[index[i]] += updates[i]

# out = np.array([[3, 3], [6, 6], [1, 1]])
# out.shape 

# x = paddle.to_tensor([[1, 1], [2, 2], [3, 3]], dtype='float32')
# index = paddle.to_tensor([2, 1, 0, 1], dtype='int64')
# updates = paddle.to_tensor([[1, 1], [2, 2], [3, 3], [4, 4]], dtype='float32')

# output1 = paddle.scatter(x, index, updates, overwrite=False)

# output2 = paddle.scatter(x, index, updates, overwrite=True)

# index_data = np.array([[1, 1],
#                        [0, 1],
#                        [1, 3]]).astype(np.int64)
# index = paddle.to_tensor(index_data)
# updates = paddle.rand(shape=[3, 9, 10], dtype='float32')
# shape = [3, 5, 9, 10]

# output = paddle.scatter_nd(index, updates, shape)

# x = paddle.rand(shape=[3, 5, 9, 10], dtype='float32')
# updates = paddle.rand(shape=[3, 9, 10], dtype='float32')
# index_data = np.array([[1, 1],
#                        [0, 1],
#                        [1, 3]]).astype(np.int64)
# index = paddle.to_tensor(index_data)
# output = paddle.scatter_nd_add(x, index, updates)
# shard_size = (index_num + nshards - 1) // nshards
# v = v - shard_id * shard_size if shard_id * shard_size <= v < (shard_id+1) * shard_size else ignore_value

# label = paddle.to_tensor([[16], [1]], "int64")
# shard_label = paddle.shard_index(input=label,
#                                  index_num=20,
#                                  nshards=2,
#                                  shard_id=0)
# print(shard_label)

# x = paddle.to_tensor([3.0, 0.0, -2.0, 1.7], dtype='float32')
# out = paddle.sign(x=x)
# print(out)  

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.sin(x)
# print(out)

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.sinh(x)
# print(out)

# input = paddle.rand(shape=[4, 5, 6], dtype='float32')

# axes = [0, 1, 2]
# starts = [-3, 0, 2]
# ends = [3, 2, 4]
# sliced_1 = paddle.slice(input, axes=axes, starts=starts, ends=ends)

# minus_3 = paddle.full([1], -3, "int32")
# sliced_2 = paddle.slice(input, axes=axes, starts=[minus_3, 0, 2], ends=ends)

# x = paddle.to_tensor([[[5,8,9,5],
#                        [0,0,1,7],
#                        [6,9,2,4]],
#                       [[5,2,4,2],
#                        [4,7,7,9],
#                        [1,7,0,6]]],
#                      dtype='float32')
# out1 = paddle.sort(x=x, axis=-1)
# out2 = paddle.sort(x=x, axis=0)
# out3 = paddle.sort(x=x, axis=1)
# print(out1)

# print(out2)

# print(out3)

# x = paddle.rand([3, 9, 5])

# out0, out1, out2 = paddle.split(x, num_or_sections=3, axis=1)
# print(out0.shape)  
# print(out1.shape)  
# print(out2.shape)  

# out0, out1, out2 = paddle.split(x, num_or_sections=[2, 3, 4], axis=1)
# print(out0.shape)  
# print(out1.shape)  
# print(out2.shape)  

# out0, out1, out2 = paddle.split(x, num_or_sections=[2, 3, -1], axis=1)
# print(out0.shape)  
# print(out1.shape)  
# print(out2.shape)  

# out0, out1, out2 = paddle.split(x, num_or_sections=3, axis=-2)
# print(out0.shape)  
# print(out1.shape)  
# print(out2.shape)  

# x = paddle.to_tensor([0.1, 0.2, 0.3, 0.4])
# out = paddle.sqrt(x)
# print(out)

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.square(x)
# print(out)

# x = paddle.rand([5, 1, 10])
# output = paddle.squeeze(x, axis=1)

# print(x.shape)  
# print(output.shape)  

# x[0, 0, 0] = 10.
# print(output[0, 0]) 

# x1 = paddle.to_tensor([[1.0, 2.0]])
# x2 = paddle.to_tensor([[3.0, 4.0]])
# x3 = paddle.to_tensor([[5.0, 6.0]])
# out = paddle.stack([x1, x2, x3], axis=0)
# print(out.shape)  
# print(out)

# x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
# out = paddle.stanh(x, scale_a=0.67, scale_b=1.72) 

# x = paddle.to_tensor([[1.0, 2.0, 3.0], [1.0, 4.0, 5.0]])
# out1 = paddle.std(x)

# out2 = paddle.std(x, axis=1)

# x = paddle.zeros(shape=[3,4,5,6], dtype="float32")

# axes = [1, 2, 3]
# starts = [-3, 0, 2]
# ends = [3, 2, 4]
# strides_1 = [1, 1, 1]
# strides_2 = [1, 1, 2]
# sliced_1 = paddle.strided_slice(x, axes=axes, starts=starts, ends=ends, strides=strides_1)

# minus_3 = paddle.full(shape=[1], fill_value=-3, dtype='int32')
# sliced_2 = paddle.strided_slice(x, axes=axes, starts=[minus_3, 0, 2], ends=ends, strides=strides_2)

# x = paddle.to_tensor([[1, 2], [7, 8]])
# y = paddle.to_tensor([[5, 6], [3, 4]])
# res = paddle.subtract(x, y)
# print(res)

# x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
# y = paddle.to_tensor([1, 0, 4])
# res = paddle.subtract(x, y)
# print(res)

# x = paddle.to_tensor([2, np.nan, 5], dtype='float32')
# y = paddle.to_tensor([1, 4, np.nan], dtype='float32')
# res = paddle.subtract(x, y)
# print(res)

# x = paddle.to_tensor([5, np.inf, -np.inf], dtype='float64')
# y = paddle.to_tensor([1, 4, 5], dtype='float64')
# res = paddle.subtract(x, y)
# print(res)

# x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
#                       [0.1, 0.2, 0.6, 0.7]])
# out1 = paddle.sum(x)  
# out2 = paddle.sum(x, axis=0)  
# out3 = paddle.sum(x, axis=-1)  
# out4 = paddle.sum(x, axis=1, keepdim=True)  

# y = paddle.to_tensor([[[1, 2], [3, 4]],
#                       [[5, 6], [7, 8]]])
# out5 = paddle.sum(y, axis=[1, 2]) 
# out6 = paddle.sum(y, axis=[0, 1]) 

# x = paddle.to_tensor([[True, True, True, True],
#                       [False, False, False, False]])
# out7 = paddle.sum(x)  
# out8 = paddle.sum(x, axis=0)  
# out9 = paddle.sum(x, axis=1)  

# x = paddle.ones(shape=[2, 3], dtype='int32')
# x_transposed = paddle.t(x)
# print(x_transposed.shape)

# x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
# out = paddle.tanh(x)
# print(out)

# data_type = 'float64'

# x = paddle.arange(4, dtype=data_type).reshape([2, 2])
# y = paddle.arange(4, dtype=data_type).reshape([2, 2])
# z = paddle.tensordot(x, y, axes=0)

# x = paddle.arange(10, dtype=data_type)
# y = paddle.arange(10, dtype=data_type)
# z1 = paddle.tensordot(x, y, axes=1)
# z2 = paddle.dot(x, y)

# x = paddle.arange(6, dtype=data_type).reshape([2, 3])
# y = paddle.arange(12, dtype=data_type).reshape([3, 4])
# z1 = paddle.tensordot(x, y, axes=1)
# z2 = paddle.matmul(x, y)

# x = paddle.arange(24, dtype=data_type).reshape([2, 3, 4])
# y = paddle.arange(36, dtype=data_type).reshape([3, 3, 4])
# z = paddle.tensordot(x, y, axes=[1, 2])

# x = paddle.arange(60, dtype=data_type).reshape([3, 4, 5])
# y = paddle.arange(24, dtype=data_type).reshape([4, 3, 2])
# z = paddle.tensordot(x, y, axes=([1, 0], [0, 1]))

# x = paddle.arange(720, dtype=data_type).reshape([2, 3, 4, 5, 6])
# y = paddle.arange(720, dtype=data_type).reshape([3, 2, 4, 5, 6])
# z = paddle.tensordot(x, y, axes=[[0, 1, 3, 4], [1, 0]])

# data = paddle.to_tensor([1, 2, 3], dtype='int32')
# out = paddle.tile(data, repeat_times=[2, 1])
# np_out = out.numpy()

# out = paddle.tile(data, repeat_times=[2, 2])
# np_out = out.numpy()

# repeat_times = paddle.to_tensor([2, 1], dtype='int32')
# out = paddle.tile(data, repeat_times=repeat_times)
# np_out = out.numpy()

# tensor_1 = paddle.to_tensor([1, 4, 5, 7])
# value_1, indices_1 = paddle.topk(tensor_1, k=1)
# print(value_1)

# print(indices_1)

# tensor_2 = paddle.to_tensor([[1, 4, 5, 7], [2, 6, 2, 5]])
# value_2, indices_2 = paddle.topk(tensor_2, k=1)
# print(value_2)

# print(indices_2)

# value_3, indices_3 = paddle.topk(tensor_2, k=1, axis=-1)
# print(value_3)

# print(indices_3)

# value_4, indices_4 = paddle.topk(tensor_2, k=1, axis=0)
# print(value_4)

# print(indices_4)

# case1 = paddle.randn([2, 3])
# case2 = paddle.randn([3, 10, 10])
# case3 = paddle.randn([3, 10, 5, 10])
# data1 = paddle.trace(case1) 
# data2 = paddle.trace(case2, offset=1, axis1=1, axis2=2) 
# data3 = paddle.trace(case3, offset=-3, axis1=1, axis2=-1) 

# x = paddle.randn([2, 3, 4])
# x_transposed = paddle.transpose(x, perm=[1, 0, 2])
# print(x_transposed.shape)

# input = paddle.rand([2,2],'float32')
# print(input)

# output = paddle.trunc(input)
# print(output)

# np_input = np.random.rand(3, 4, 5).astype('float32')
# input = paddle.to_tensor(np_input)
# [x0, x1, x2] = paddle.unbind(input, axis=0)

# [x0, x1, x2, x3] = paddle.unbind(input, axis=1)

# x = paddle.ones(shape=[3, 4])
# x.uniform_()
# print(x)

# x = paddle.to_tensor([2, 3, 3, 1, 5, 3])
# unique = paddle.unique(x)
# np_unique = unique.numpy() 
# _, indices, inverse, counts = paddle.unique(x, return_index=True, return_inverse=True, return_counts=True)
# np_indices = indices.numpy() 
# np_inverse = inverse.numpy() 
# np_counts = counts.numpy() 

# x = paddle.to_tensor([[2, 1, 3], [3, 0, 1], [2, 1, 3]])
# unique = paddle.unique(x)
# np_unique = unique.numpy() 

# unique = paddle.unique(x, axis=0)
# np_unique = unique.numpy()

# x = paddle.to_tensor([1, 1, 2, 2, 3, 1, 1, 2])
# output = paddle.unique_consecutive(x) 
# np_output = output.numpy() 
# _, inverse, counts = paddle.unique_consecutive(x, return_inverse=True, return_counts=True)
# np_inverse = inverse.numpy() 
# np_counts = inverse.numpy() 

# x = paddle.to_tensor([[2, 1, 3], [3, 0, 1], [2, 1, 3], [2, 1, 3]])
# output = paddle.unique_consecutive(x, axis=0) 
# np_output = output.numpy() 

# x = paddle.to_tensor([[2, 1, 3], [3, 0, 1], [2, 1, 3], [2, 1, 3]])
# output = paddle.unique_consecutive(x, axis=0) 
# np_output = output.numpy()

# x = paddle.rand([5, 10])
# print(x.shape)  

# out1 = paddle.unsqueeze(x, axis=0)
# print(out1.shape)  

# out2 = paddle.unsqueeze(x, axis=[0, 2])
# print(out2.shape)  

# axis = paddle.to_tensor([0, 1, 2])
# out3 = paddle.unsqueeze(x, axis=axis)
# print(out3.shape)  

# x[0, 0] = 10.
# print(out1[0, 0, 0]) 
# print(out2[0, 0, 0, 0]) 
# print(out3[0, 0, 0, 0, 0]) 

# x = paddle.ones(name='x', shape=[2, 3, 5], dtype='float32')  
# y = paddle.unstack(x, axis=1)  

# x = paddle.to_tensor([[1.0, 2.0, 3.0], [1.0, 4.0, 5.0]])
# out1 = paddle.var(x)

# out2 = paddle.var(x, axis=1)

# x = paddle.to_tensor([0.9383, 0.1983, 3.2, 1.2])
# y = paddle.to_tensor([1.0, 1.0, 1.0, 1.0])
# out = paddle.where(x>1, x, y)

# print(out)

# paddle.enable_static()

# data = paddle.static.data(name="data", shape=[3, 32, 32], dtype="float32")

# fc = paddle.static.nn.fc(x=data,
#                          size=1000,
#                          weight_attr=paddle.static.WeightNormParamAttr(
#                              dim=None,
#                              name='weight_norm_param',
#                              initializer=paddle.nn.initializer.Constant(1.0),
#                              learning_rate=1.0,
#                              regularizer=paddle.regularizer.L2Decay(0.1),
#                              trainable=True,
#                              do_model_average=False,
#                              need_clip=True))

# paddle.enable_static()
# xpu_places = static.xpu_places()

# include_dir = paddle.sysconfig.get_include()

# include_dir = paddle.sysconfig.get_lib()

# original_tensor = paddle.ones([2, 2])
# print("original tensor's dtype is: {}".format(original_tensor.dtype))
# new_tensor = original_tensor.astype('float32')
# print("new tensor's dtype is: {}".format(new_tensor.dtype))

# x = paddle.to_tensor(5., stop_gradient=False)
# for i in range(5):
#     y = paddle.pow(x, 4.0)
#     y.backward()
#     print("{}: {}".format(i, x.grad))

# x.clear_grad()
# print("{}".format(x.grad))

# grad_tensor=paddle.to_tensor(2.)
# for i in range(5):
#     y = paddle.pow(x, 4.0)
#     y.backward(grad_tensor)
#     print("{}: {}".format(i, x.grad))

# tensor = paddle.to_tensor([0, 1, 2, 3, 4])

# tensor.fill_(0)
# print(tensor.tolist())   

# x = paddle.ones((4, 3)) * 2
# x.fill_diagonal_(1.0)
# print(x.tolist())   

# x = paddle.ones((4, 3)) * 2
# y = paddle.ones((3,))
# nx = x.fill_diagonal_tensor(y)
# print(nx.tolist())   

# x = paddle.ones((4, 3)) * 2
# y = paddle.ones((3,))
# x.fill_diagonal_tensor_(y)
# print(x.tolist())   

# x = paddle.to_tensor(5., stop_gradient=False)
# y = paddle.pow(x, 4.0)
# y.backward()
# print("grad of x: {}".format(x.gradient()))

# x = paddle.to_tensor(1)
# print(x.item())             
# print(type(x.item()))       

# x = paddle.to_tensor(1.0)
# print(x.item())             
# print(type(x.item()))       

# x = paddle.to_tensor(True)
# print(x.item())             
# print(type(x.item()))       

# x = paddle.to_tensor(1+1j)
# print(x.item())             
# print(type(x.item()))       

# x = paddle.to_tensor([[1.1, 2.2, 3.3]])
# print(x.item(2))            
# print(x.item(0, 2))         

# def print_hook_fn(grad):
#     print(grad)

# def double_hook_fn(grad):
#     grad = grad * 2
#     return grad

# x = paddle.to_tensor([0., 1., 2., 3.], stop_gradient=False)
# y = paddle.to_tensor([4., 5., 6., 7.], stop_gradient=False)
# z = paddle.to_tensor([1., 2., 3., 4.])

# h = x.register_hook(print_hook_fn)
# x.register_hook(double_hook_fn)

# w = x + y

# w.register_hook(lambda grad: grad * 2)

# o = z.matmul(w)
# o.backward()

# print("w.grad:", w.grad) 
# print("x.grad:", x.grad) 
# print("y.grad:", y.grad) 

# h.remove()

# data = np.ones([3, 1024], dtype='float32')
# with fluid.dygraph.guard():
#     linear = fluid.dygraph.Linear(1024, 4)
#     t = to_variable(data)
#     linear(t)  
#     custom_weight = np.random.randn(1024, 4).astype("float32")
#     linear.weight.set_value(custom_weight)  
#     out = linear(t)  

# x = paddle.ones(shape=[3, 4])
# x.uniform_()
# print(x)

# tensor = paddle.to_tensor([0, 1, 2, 3, 4])

# tensor.zero_()
# print(tensor.tolist())   

# class SimpleNet(paddle.nn.Layer):
#     def __init__(self):
#         super(SimpleNet, self).__init__()

#     def forward(self, pred_idx, mark, label):
#         return paddle.sum(pred_idx), paddle.sum(mark), paddle.sum(label)

# conll05st = Conll05st()

# for i in range(10):
#     pred_idx, mark, label= conll05st[i][-3:]
#     pred_idx = paddle.to_tensor(pred_idx)
#     mark = paddle.to_tensor(mark)
#     label = paddle.to_tensor(label)

#     model = SimpleNet()
#     pred_idx, mark, label= model(pred_idx, mark, label)
#     print(pred_idx.numpy(), mark.numpy(), label.numpy())

# conll05st = Conll05st()
# word_dict, predicate_dict, label_dict = conll05st.get_dict()

# conll05st = Conll05st()
# emb_file = conll05st.get_embedding()

# class SimpleNet(paddle.nn.Layer):
#     def __init__(self):
#         super(SimpleNet, self).__init__()

#     def forward(self, doc, label):
#         return paddle.sum(doc), label

# imdb = Imdb(mode='train')

# for i in range(10):
#     doc, label = imdb[i]
#     doc = paddle.to_tensor(doc)
#     label = paddle.to_tensor(label)

#     model = SimpleNet()
#     image, label = model(doc, label)
#     print(doc.numpy().shape, label.numpy().shape)

# class SimpleNet(paddle.nn.Layer):
#     def __init__(self):
#         super(SimpleNet, self).__init__()

#     def forward(self, src, trg):
#         return paddle.sum(src), paddle.sum(trg)

# imikolov = Imikolov(mode='train', data_type='SEQ', window_size=2)

# for i in range(10):
#     src, trg = imikolov[i]
#     src = paddle.to_tensor(src)
#     trg = paddle.to_tensor(trg)

#     model = SimpleNet()
#     src, trg = model(src, trg)
#     print(src.numpy().shape, trg.numpy().shape)

# class SimpleNet(paddle.nn.Layer):
#     def __init__(self):
#         super(SimpleNet, self).__init__()

#     def forward(self, category, title, rating):
#         return paddle.sum(category), paddle.sum(title), paddle.sum(rating)

# movielens = Movielens(mode='train')

# for i in range(10):
#     category, title, rating = movielens[i][-3:]
#     category = paddle.to_tensor(category)
#     title = paddle.to_tensor(title)
#     rating = paddle.to_tensor(rating)

#     model = SimpleNet()
#     category, title, rating = model(category, title, rating)
#     print(category.numpy().shape, title.numpy().shape, rating.numpy().shape)

# class SimpleNet(paddle.nn.Layer):
#     def __init__(self):
#         super(SimpleNet, self).__init__()

#     def forward(self, feature, target):
#         return paddle.sum(feature), target

# paddle.disable_static()

# uci_housing = UCIHousing(mode='train')

# for i in range(10):
#     feature, target = uci_housing[i]
#     feature = paddle.to_tensor(feature)
#     target = paddle.to_tensor(target)

#     model = SimpleNet()
#     feature, target = model(feature, target)
#     print(feature.numpy().shape, target.numpy())

# paddle.seed(102)
# batch_size, seq_len, num_tags = 2, 4, 3
# emission = paddle.rand((batch_size, seq_len, num_tags), dtype='float32')
# length = paddle.randint(1, seq_len + 1, [batch_size])
# tags = paddle.randint(0, num_tags, [batch_size, seq_len])
# transition = paddle.rand((num_tags, num_tags), dtype='float32')
# scores, path = paddle.text.viterbi_decode(emission, transition, length, False) 

# paddle.seed(102)
# batch_size, seq_len, num_tags = 2, 4, 3
# emission = paddle.rand((batch_size, seq_len, num_tags), dtype='float32')
# length = paddle.randint(1, seq_len + 1, [batch_size])
# tags = paddle.randint(0, num_tags, [batch_size, seq_len])
# transition = paddle.rand((num_tags, num_tags), dtype='float32')
# decoder = paddle.text.ViterbiDecoder(transition, include_bos_eos_tag=False)
# scores, path = decoder(emission, length) 

# class MyLayer(paddle.nn.Layer):
#     def __init__(self):
#         super(MyLayer, self).__init__()
#         self._linear = paddle.nn.Linear(1, 1)
#         w_tmp = self.create_parameter([1,1])
#         self.add_parameter("w_tmp", w_tmp)

#     def forward(self, input):
#         return self._linear(input)

# mylayer = MyLayer()
# for name, param in mylayer.named_parameters():
#     print(name, param)      

# class MySequential(paddle.nn.Layer):
#     def __init__(self, *layers):
#         super(MySequential, self).__init__()
#         if len(layers) > 0 and isinstance(layers[0], tuple):
#             for name, layer in layers:
#                 self.add_sublayer(name, layer)
#         else:
#             for idx, layer in enumerate(layers):
#                 self.add_sublayer(str(idx), layer)

#     def forward(self, input):
#         for layer in self._sub_layers.values():
#             input = layer(input)
#         return input

# fc1 = paddle.nn.Linear(10, 3)
# fc2 = paddle.nn.Linear(3, 10, bias_attr=False)
# model = MySequential(fc1, fc2)
# for prefix, layer in model.named_sublayers():
#     print(prefix, layer)

# net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))

# def init_weights(layer):
#     if type(layer) == nn.Linear:
#         print('before init weight:', layer.weight.numpy())
#         new_weight = paddle.full(shape=layer.weight.shape, dtype=layer.weight.dtype, fill_value=0.9)
#         layer.weight.set_value(new_weight)
#         print('after init weight:', layer.weight.numpy())

# net.apply(init_weights)

# print(net.state_dict())

# linear = paddle.nn.Linear(10, 3)
# value = np.array([0]).astype("float32")
# buffer = paddle.to_tensor(value)
# linear.register_buffer("buf_name", buffer, persistable=True)

# print(linear.buffers())     

# linear1 = paddle.nn.Linear(10, 3)
# linear2 = paddle.nn.Linear(3, 10, bias_attr=False)
# model = paddle.nn.Sequential(linear1, linear2)

# layer_list = list(model.children())

# print(layer_list)   

# value = np.arange(26).reshape(2, 13).astype("float32")
# a = paddle.to_tensor(value)
# linear = paddle.nn.Linear(13, 5)
# adam = paddle.optimizer.Adam(learning_rate=0.01,
#                             parameters=linear.parameters())
# out = linear(a)
# out.backward()
# adam.step()
# linear.clear_gradients()

# class MyLayer(paddle.nn.Layer):
#     def __init__(self):
#         super(MyLayer, self).__init__()
#         self._linear = paddle.nn.Linear(1, 1)
#         w_tmp = self.create_parameter([1,1])
#         self.add_parameter("w_tmp", w_tmp)

#     def forward(self, input):
#         return self._linear(input)

# mylayer = MyLayer()
# for name, param in mylayer.named_parameters():
#     print(name, param)      

# class MyLinear(paddle.nn.Layer):
#     def __init__(self,
#                 in_features,
#                 out_features):
#         super(MyLinear, self).__init__()
#         self.linear = paddle.nn.Linear( 10, 10)

#         self.back_var = self.create_tensor(name = "linear_tmp_0", dtype=self._dtype)

#     def forward(self, input):
#         out = self.linear(input)
#         paddle.assign( out, self.back_var)

#         return out

# class MyLinear(paddle.nn.Layer):
#     def __init__(self,
#                 in_features,
#                 out_features):
#         super(MyLinear, self).__init__()
#         self.linear = paddle.nn.Linear( 10, 10)

#         self.back_var = self.create_variable(name = "linear_tmp_0", dtype=self._dtype)

#     def forward(self, input):
#         out = self.linear(input)
#         paddle.assign( out, self.back_var)

#         return out

# class MyLayer(paddle.nn.Layer):
#     def __init__(self):
#         super(MyLayer, self).__init__()
#         self._linear = paddle.nn.Linear(1, 1)
#         self._dropout = paddle.nn.Dropout(p=0.5)

#     def forward(self, input):
#         temp = self._linear(input)
#         temp = self._dropout(temp)
#         return temp

# x = paddle.randn([10, 1], 'float32')
# mylayer = MyLayer()
# mylayer.eval()  
# out = mylayer(x)
# print(out)

# class LinearNet(paddle.nn.Layer):
#     def __init__(self):
#         super(LinearNet, self).__init__(name_scope = "demo_linear_net")
#         self._linear = paddle.nn.Linear(1, 1)

#     def forward(self, x):
#         return self._linear(x)

# linear_net = LinearNet()
# print(linear_net.full_name())   

# emb = paddle.nn.Embedding(10, 10)

# state_dict = emb.state_dict()
# paddle.save(state_dict, "paddle_dy.pdparams")
# para_state_dict = paddle.load("paddle_dy.pdparams")
# emb.set_state_dict(para_state_dict)

# fc1 = paddle.nn.Linear(10, 3)
# buffer1 = paddle.to_tensor(np.array([0]).astype("float32"))

# fc1.register_buffer("buf_name_1", buffer1, persistable=True)

# fc2 = paddle.nn.Linear(3, 10)
# buffer2 = paddle.to_tensor(np.array([1]).astype("float32"))

# fc2.buf_name_2 = buffer2

# model = paddle.nn.Sequential(fc1, fc2)

# for name, buffer in model.named_buffers():
#     print(name, buffer)

# linear1 = paddle.nn.Linear(10, 3)
# linear2 = paddle.nn.Linear(3, 10, bias_attr=False)
# model = paddle.nn.Sequential(linear1, linear2)
# for prefix, layer in model.named_children():
#     print(prefix, layer)
    
    

# fc1 = paddle.nn.Linear(10, 3)
# fc2 = paddle.nn.Linear(3, 10, bias_attr=False)
# model = paddle.nn.Sequential(fc1, fc2)
# for name, param in model.named_parameters():
#     print(name, param)

# fc1 = paddle.nn.Linear(10, 3)
# fc2 = paddle.nn.Linear(3, 10, bias_attr=False)
# model = paddle.nn.Sequential(fc1, fc2)
# for prefix, layer in model.named_sublayers():
#     print(prefix, layer)

# linear = paddle.nn.Linear(10, 3)
# value = np.array([0]).astype("float32")
# buffer = paddle.to_tensor(value)
# linear.register_buffer("buf_name", buffer, persistable=True)

# print(linear.buf_name)

# def forward_post_hook(layer, input, output):
    

    
#     return output * 2

# linear = paddle.nn.Linear(13, 5)

# forward_post_hook_handle = linear.register_forward_post_hook(forward_post_hook)

# value1 = np.arange(26).reshape(2, 13).astype("float32")
# in1 = paddle.to_tensor(value1)

# out0 = linear(in1)

# forward_post_hook_handle.remove()

# out1 = linear(in1)

# assert (out0.numpy() == (out1.numpy()) * 2).any()

# def forward_pre_hook(layer, input):
    

    
#     input_return = (input[0] * 2)
#     return input_return

# linear = paddle.nn.Linear(13, 5)

# forward_pre_hook_handle = linear.register_forward_pre_hook(forward_pre_hook)

# value0 = np.arange(26).reshape(2, 13).astype("float32")
# in0 = paddle.to_tensor(value0)
# out0 = linear(in0)

# forward_pre_hook_handle.remove()

# value1 = value0 * 2
# in1 = paddle.to_tensor(value1)
# out1 = linear(in1)

# assert (out0.numpy() == out1.numpy()).any()

# emb = paddle.nn.Embedding(10, 10)

# state_dict = emb.state_dict()
# paddle.save(state_dict, "paddle_dy.pdparams")
# para_state_dict = paddle.load("paddle_dy.pdparams")
# emb.set_state_dict(para_state_dict)

# emb = paddle.nn.Embedding(10, 10)

# state_dict = emb.state_dict()
# paddle.save(state_dict, "paddle_dy.pdparams")
# para_state_dict = paddle.load("paddle_dy.pdparams")
# emb.set_state_dict(para_state_dict)

# emb = paddle.nn.Embedding(10, 10)

# state_dict = emb.state_dict()
# paddle.save( state_dict, "paddle_dy.pdparams")

# class MyLayer(paddle.nn.Layer):
#     def __init__(self):
#         super(MyLayer, self).__init__()
#         self._linear = paddle.nn.Linear(1, 1)
#         self._dropout = paddle.nn.Dropout(p=0.5)

#     def forward(self, input):
#         temp = self._linear(input)
#         temp = self._dropout(temp)
#         return temp

# mylayer = MyLayer()
# print(mylayer.sublayers())  

# linear=paddle.nn.Linear(2, 2)
# linear.weight

# linear.to(dtype='float64')
# linear.weight

# linear.to(device='cpu')
# linear.weight

# linear.to(device=paddle.CUDAPinnedPlace(), blocking=False)
# linear.weight

# emb = paddle.nn.Embedding(10, 10)

# state_dict = emb.to_static_state_dict()
# paddle.save( state_dict, "paddle_dy.pdparams")

# class MyLayer(paddle.nn.Layer):
#     def __init__(self):
#         super(MyLayer, self).__init__()
#         self._linear = paddle.nn.Linear(1, 1)
#         self._dropout = paddle.nn.Dropout(p=0.5)

#     def forward(self, input):
#         temp = self._linear(input)
#         temp = self._dropout(temp)
#         return temp

# x = paddle.randn([10, 1], 'float32')
# mylayer = MyLayer()
# mylayer.eval()  
# out = mylayer(x)
# mylayer.train()  
# out = mylayer(x)

# class SimpleNet(paddle.nn.Layer):
#     def __init__(self):
#         super(SimpleNet, self).__init__()

#     def forward(self, src_ids, trg_ids, trg_ids_next):
#         return paddle.sum(src_ids), paddle.sum(trg_ids), paddle.sum(trg_ids_next)

# wmt14 = WMT14(mode='train', dict_size=50)

# for i in range(10):
#     src_ids, trg_ids, trg_ids_next = wmt14[i]
#     src_ids = paddle.to_tensor(src_ids)
#     trg_ids = paddle.to_tensor(trg_ids)
#     trg_ids_next = paddle.to_tensor(trg_ids_next)

#     model = SimpleNet()
#     src_ids, trg_ids, trg_ids_next = model(src_ids, trg_ids, trg_ids_next)
#     print(src_ids.numpy(), trg_ids.numpy(), trg_ids_next.numpy())

# wmt14 = WMT14(mode='train', dict_size=50)
# src_dict, trg_dict = wmt14.get_dict()

# class SimpleNet(paddle.nn.Layer):
#     def __init__(self):
#         super(SimpleNet, self).__init__()

#     def forward(self, src_ids, trg_ids, trg_ids_next):
#         return paddle.sum(src_ids), paddle.sum(trg_ids), paddle.sum(trg_ids_next)

# paddle.disable_static()

# wmt16 = WMT16(mode='train', src_dict_size=50, trg_dict_size=50)

# for i in range(10):
#     src_ids, trg_ids, trg_ids_next = wmt16[i]
#     src_ids = paddle.to_tensor(src_ids)
#     trg_ids = paddle.to_tensor(trg_ids)
#     trg_ids_next = paddle.to_tensor(trg_ids_next)

#     model = SimpleNet()
#     src_ids, trg_ids, trg_ids_next = model(src_ids, trg_ids, trg_ids_next)
#     print(src_ids.numpy(), trg_ids.numpy(), trg_ids_next.numpy())

# wmt16 = WMT16(mode='train', src_dict_size=50, trg_dict_size=50)
# en_dict = wmt16.get_dict('en')

# fluid.require_version('0.1.0')

# fluid.require_version(min_version='0.1.0', max_version='10.0.0')

# paddle.utils.run_check()

# paddle.version.cuda()

# paddle.version.cudnn()

# backend = get_image_backend()
# print(backend)

# fake_img = Image.fromarray((np.random.random((32, 32, 3)) * 255).astype('uint8'))

# path = 'temp.png'
# fake_img.save(path)

# set_image_backend('pil')

# pil_img = image_load(path).convert('RGB')

# print(type(pil_img))

# set_image_backend('pil')

# def make_fake_dir():
#     data_dir = tempfile.mkdtemp()

#     for i in range(2):
#         sub_dir = os.path.join(data_dir, 'class_' + str(i))
#         if not os.path.exists(sub_dir):
#             os.makedirs(sub_dir)
#         for j in range(2):
#             fake_img = Image.fromarray((np.random.random((32, 32, 3)) * 255).astype('uint8'))
#             fake_img.save(os.path.join(sub_dir, str(j) + '.png'))
#     return data_dir

# temp_dir = make_fake_dir()

# pil_data_folder = DatasetFolder(temp_dir)

# for items in pil_data_folder:
#     break

# print(type(items[0]))

# shutil.rmtree(temp_dir)
