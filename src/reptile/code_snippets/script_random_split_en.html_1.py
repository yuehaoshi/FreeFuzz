import paddle
from paddle.io import random_split

a_list = paddle.io.random_split(range(10), [3, 7])
print(len(a_list))
# 2

for idx, v in enumerate(a_list[0]):
    print(idx, v)

# output of the first subset
# 0 1
# 1 3
# 2 9

for idx, v in enumerate(a_list[1]):
    print(idx, v)
# output of the second subset
# 0 5
# 1 7
# 2 8
# 3 6
# 4 0
# 5 2
# 6 4