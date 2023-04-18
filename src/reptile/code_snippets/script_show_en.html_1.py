import paddle

# Case 1: paddle is tagged with 2.2.0
paddle.version.show()
# full_version: 2.2.0
# major: 2
# minor: 2
# patch: 0
# rc: 0
# cuda: '10.2'
# cudnn: '7.6.5'

# Case 2: paddle is not tagged
paddle.version.show()
# commit: cfa357e984bfd2ffa16820e354020529df434f7d
# cuda: '10.2'
# cudnn: '7.6.5'