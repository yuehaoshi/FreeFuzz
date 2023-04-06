import paddle
from paddle.distribution import Categorical

paddle.seed(100) # on CPU device
x = paddle.rand([6])
print(x)
# [0.5535528  0.20714243 0.01162981
#  0.51577556 0.36369765 0.2609165 ]

cat = Categorical(x)

value = paddle.to_tensor([2,1,3])
cat.probs(value)
# [0.00608027 0.108298 0.269656]