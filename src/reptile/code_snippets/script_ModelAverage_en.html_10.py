# train on default dynamic graph mode
import paddle
import numpy as np
emb = paddle.nn.Embedding(10, 3)

## example1: LRScheduler is not used, return the same value is all the same
adam = paddle.optimizer.Adam(0.01, parameters = emb.parameters())
for batch in range(10):
    input = paddle.randint(low=0, high=5, shape=[5])
    out = emb(input)
    out.backward()
    print("Learning rate of step{}: {}".format(batch, adam.get_lr())) # 0.01
    adam.step()

## example2: StepDecay is used, return the scheduled learning rate
scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.5, step_size=2, gamma=0.1)
adam = paddle.optimizer.Adam(scheduler, parameters = emb.parameters())
for batch in range(10):
    input = paddle.randint(low=0, high=5, shape=[5])
    out = emb(input)
    out.backward()
    print("Learning rate of step{}: {}".format(batch, adam.get_lr())) # 0.5->0.05...
    adam.step()
    scheduler.step()

# train on static graph mode
paddle.enable_static()
main_prog = paddle.static.Program()
start_prog = paddle.static.Program()
with paddle.static.program_guard(main_prog, start_prog):
    x = paddle.static.data(name='x', shape=[None, 10])
    z = paddle.static.nn.fc(x, 100)
    loss = paddle.mean(z)
    scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.5, step_size=2, gamma=0.1)
    adam = paddle.optimizer.Adam(learning_rate=scheduler)
    adam.minimize(loss)

exe = paddle.static.Executor()
exe.run(start_prog)
for batch in range(10):
    print("Learning rate of step{}: {}", adam.get_lr())     # 0.5->0.05->0.005...
    out = exe.run(main_prog, feed={'x': np.random.randn(3, 10).astype('float32')})
    scheduler.step()