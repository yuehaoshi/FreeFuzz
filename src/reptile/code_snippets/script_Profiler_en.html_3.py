# required: gpu
import paddle.profiler as profiler
p = profiler.Profiler()
p.start()
for iter in range(10):
    #train()
    p.step()
p.stop()
p.summary()