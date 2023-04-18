import paddle.profiler as profiler
prof = profiler.Profiler(timer_only=True)
prof.start()
for iter in range(20):
    #train()
    prof.step()
    if iter % 10 == 0:
        print("Iter {}: {}".format(iter, prof.step_info()))
        # The example does not call the DataLoader, so there is no "reader_cost".
        # Iter 0:  batch_cost: 0.00001 s ips: 86216.623 steps/s
        # Iter 10:  batch_cost: 0.00001 s ips: 103645.034 steps/s
prof.stop()
# Time Unit: s, IPS Unit: steps/s
# |                 |       avg       |       max       |       min       |
# |    batch_cost   |     0.00000     |     0.00002     |     0.00000     |
# |       ips       |   267846.19437  |   712030.38727  |   45134.16662   |