# required: gpu
import paddle.profiler as profiler
with profiler.Profiler(
        targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU],
        scheduler = (3, 10)) as p:
    for iter in range(10):
        #train()
        p.step()
p.export('test_export_protobuf.pb', format='pb')
profiler_result = profiler.load_profiler_result('test_export_protobuf.pb')