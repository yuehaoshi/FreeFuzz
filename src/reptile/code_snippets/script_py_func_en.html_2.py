# example 2:
# This example shows how to turn Tensor into numpy array and
# use numpy API to register an Python OP
import paddle
import numpy as np

paddle.enable_static()

def element_wise_add(x, y):
    # Tensor must be actively converted to numpy array, otherwise,
    # numpy.shape can't be used.
    x = np.array(x)
    y = np.array(y)

    if x.shape != y.shape:
        raise AssertionError("the shape of inputs must be the same!")

    result = np.zeros(x.shape, dtype='int32')
    for i in range(len(x)):
        for j in range(len(x[0])):
            result[i][j] = x[i][j] + y[i][j]

    return result

def create_tmp_var(name, dtype, shape):
    return paddle.static.default_main_program().current_block().create_var(
                name=name, dtype=dtype, shape=shape)

def py_func_demo():
    start_program = paddle.static.default_startup_program()
    main_program = paddle.static.default_main_program()

    # Input of the forward function
    x = paddle.static.data(name='x', shape=[2,3], dtype='int32')
    y = paddle.static.data(name='y', shape=[2,3], dtype='int32')

    # Output of the forward function, name/dtype/shape must be specified
    output = create_tmp_var('output','int32', [3,1])

    # Multiple Variable should be passed in the form of tuple(Variale) or list[Variale]
    paddle.static.py_func(func=element_wise_add, x=[x,y], out=output)

    exe=paddle.static.Executor(paddle.CPUPlace())
    exe.run(start_program)

    # Feed numpy array to main_program
    input1 = np.random.randint(1, 10, size=[2,3], dtype='int32')
    input2 = np.random.randint(1, 10, size=[2,3], dtype='int32')
    out = exe.run(main_program,
                feed={'x':input1, 'y':input2},
                fetch_list=[output.name])
    print("{0} + {1} = {2}".format(input1, input2, out))

py_func_demo()

# Reference output:
# [[5, 9, 9]   + [[7, 8, 4]  =  [array([[12, 17, 13]
#  [7, 5, 2]]     [1, 3, 3]]            [8, 8, 5]], dtype=int32)]