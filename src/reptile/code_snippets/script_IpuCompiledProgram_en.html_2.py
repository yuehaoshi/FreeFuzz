# required: ipu

import paddle
import paddle.static as static

paddle.enable_static()

a = static.data(name='data', shape=[None, 1], dtype='int32')
b = a + 1
main_prog = static.default_main_program()

ipu_strategy = static.IpuStrategy()
ipu_strategy.set_graph_config(num_ipus=1, is_training=True, micro_batch_size=1)
ipu_strategy.set_pipelining_config(enable_pipelining=False, batches_per_step=1, enable_gradient_accumulation=False, accumulation_factor=1)
ipu_strategy.set_precision_config(enable_fp16=False)

program = static.IpuCompiledProgram(
    main_prog,
    ipu_strategy=ipu_strategy).compile([a.name], [b.name])