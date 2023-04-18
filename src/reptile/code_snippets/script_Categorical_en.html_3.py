import paddle
from paddle.distribution import Categorical

paddle.seed(100) # on CPU device
x = paddle.rand([6])
print(x)
# [0.5535528  0.20714243 0.01162981
#  0.51577556 0.36369765 0.2609165 ]

paddle.seed(200) # on CPU device
y = paddle.rand([6])
print(y)
# [0.77663314 0.90824795 0.15685187
#  0.04279523 0.34468332 0.7955718 ]

cat = Categorical(x)
cat2 = Categorical(y)

cat.kl_divergence(cat2)
# [0.071952]