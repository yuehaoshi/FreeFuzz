# required: gpu
import paddle.profiler as profiler
prof = profiler.Profiler(
    targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU],
    scheduler = (3, 7))
prof.start()
for iter in range(10):
    #train()
    prof.step()
prof.stop()
prof.export(path="./profiler_data.json", format="json")