import paddle.fluid as fluid

# any version >= 0.1.0 is acceptable.
fluid.require_version('0.1.0')

# if 0.1.0 <= version <= 10.0.0, it is acceptable.
fluid.require_version(min_version='0.1.0', max_version='10.0.0')