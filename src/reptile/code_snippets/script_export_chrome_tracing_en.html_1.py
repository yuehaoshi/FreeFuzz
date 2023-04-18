# required: gpu
import paddle.profiler as profiler
with profiler.Profiler(
        targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU],
        scheduler = (3, 10),
        on_trace_ready=profiler.export_protobuf('./log')) as p:
    for iter in range(10):
        #train()
        p.step()