from classes.library import Library

import numpy as np
from os.path import join

from classes.argument import Argument, ArgType
from classes.paddle_api import PaddleAPI, PaddleArgument
from classes.library import Library
from classes.database import PaddleDatabase
from constants.enum import OracleType
from constants.keys import ERR_CPU_KEY, ERR_GPU_KEY, ERR_HIGH_KEY, ERR_LOW_KEY, ERROR_KEY, RES_CPU_KEY, RES_GPU_KEY, \
    TIME_HIGH_KEY, TIME_LOW_KEY, RES_KEY
import paddle


# class PaddleLibrary(Library):
#     def __init__(self, output_dir, diff_bound=1e-5, time_bound=10, time_thresold=1e-3) -> None:
#         super().__init__(output_dir)
#         self.diff_bound = diff_bound
#         self.time_bound = time_bound
#         self.time_thresold = time_thresold
#
#     def test_with_oracle(self, api: PaddleAPI, oracle: OracleType):
#         if oracle == OracleType.CRASH:
#             # We need call another process to catch the crash error
#             code = "import paddle\n"
#             code += self.generate_code(api, oracle)
#
#             with open(join(self.directory, "temp.py"), "w") as f:
#                 f.write(code)
#             results, error = self.run_code(code)
#             if error == None:
#                 self.write_to_dir(join(self.output[oracle], "success"), api.api, code)
#             else:
#                 self.write_to_dir(join(self.output[oracle], "fail"), api.api, code)
#         elif oracle == OracleType.CUDA:
#             code = "import paddle\n"
#             code += self.generate_code(api, oracle)
#
#             write_code = "results = dict()\n" + code + "\nprint(results)\n"
#             with open(join(self.directory, "temp.py"), "w") as f:
#                 f.write(write_code)
#
#             results, error = self.run_code(code)
#             err_cpu = results[ERR_CPU_KEY]
#             err_gpu = results[ERR_GPU_KEY]
#             write_dir = ""
#             if error is None:
#                 if (err_cpu is None) != (err_gpu is None):
#                     write_dir = join(self.output[oracle], "potential-bug")
#                 elif err_cpu == None:
#                     res_cpu = results[RES_CPU_KEY]
#                     res_gpu = results[RES_GPU_KEY]
#                     if self.is_equal(res_cpu, res_gpu):
#                         write_dir = join(self.output[oracle], "success")
#                     else:
#                         write_dir = join(self.output[oracle], "potential-bug")
#                 elif "SystemError" in err_cpu or "SystemError" in err_gpu:
#                     write_dir = join(self.output[oracle], "potential-bug")
#                 else:
#                     write_dir = join(self.output[oracle], "success")
#             elif "SystemError" in error:
#                 write_dir = join(self.output[oracle], "potential-bug")
#             else:
#                 write_dir = join(self.output[oracle], "fail")
#             self.write_to_dir(write_dir, api.api, write_code)
#         elif oracle == OracleType.PRECISION:
#             code = "import paddle\n"
#             code += "import time\n"
#             code += self.generate_code(api, oracle)
#
#             write_code = "results = dict()\n" + code + "\nprint(results)\n"
#             with open(join(self.directory, "temp.py"), "w") as f:
#                 f.write(write_code)
#
#             results, error = self.run_code(code)
#             err_high = results[ERR_HIGH_KEY]
#             err_low = results[ERR_LOW_KEY]
#             write_dir = ""
#             if error is None:
#                 if (err_high is None) != (err_low is None):
#                     write_dir = join(self.output[oracle], "potential-bug")
#                 elif err_high == None:
#                     time_high = results[TIME_HIGH_KEY]
#                     time_low = results[TIME_LOW_KEY]
#                     if time_low >= self.time_bound * time_high and time_high >= self.time_thresold:
#                         write_dir = join(self.output[oracle], "potential-bug")
#                     else:
#                         write_dir = join(self.output[oracle], "success")
#                 elif "SystemError" in err_high or "SystemError" in err_low:
#                     write_dir = join(self.output[oracle], "potential-bug")
#                 else:
#                     write_dir = join(self.output[oracle], "success")
#             elif "SystemError" in error:
#                 write_dir = join(self.output[oracle], "potential-bug")
#             else:
#                 write_dir = join(self.output[oracle], "fail")
#             self.write_to_dir(write_dir, api.api, write_code)
#
#     @staticmethod
#     def generate_code(api: PaddleAPI, oracle: OracleType) -> str:
#         code = ""
#         if oracle == OracleType.CRASH:
#             code += api.to_code_oracle(oracle=oracle)
#             return code
#         elif oracle == OracleType.CUDA:
#             code += api.to_code_oracle(oracle=oracle)
#             return code
#         elif oracle == OracleType.PRECISION:
#             code += api.to_code_oracle(oracle=oracle)
#             return code
#         else:
#             assert (0)
#
#     @staticmethod
#     def run_code(code):
#         results = dict()
#         results[ERR_CPU_KEY] = None
#         results[ERR_GPU_KEY] = None
#         results[ERR_HIGH_KEY] = None
#         results[ERR_LOW_KEY] = None
#
#         exec(code)
#         error = results[ERROR_KEY] if ERROR_KEY in results else None
#         return results, error
#
#     @staticmethod
#     def get_type(x):
#         res = Argument.get_type(x)
#         if res != None:
#             return res
#         if isinstance(x, paddle.Tensor):
#             return ArgType.PADDLE_TENSOR
#         elif isinstance(x, paddle.dtype):
#             return ArgType.PADDLE_DTYPE
#         else:
#             return ArgType.PADDLE_OBJECT
#
#     # @staticmethod
#     # def _eval_k(x):
#     #     return tf.convert_to_tensor(x).numpy()
#
#     @staticmethod
#     def get_tensor_value(t):
#         return t.numpy()
#
#     @staticmethod
#     def is_equal(x, y):
#         x_type = PaddleArgument.get_type(x)
#         y_type = PaddleArgument.get_type(y)
#         if x_type != y_type:
#             return False
#         if x_type == ArgType.PADDLE_TENSOR:
#             try:
#                 np_x = PaddleLibrary.get_tensor_value(x)
#                 np_y = PaddleLibrary.get_tensor_value(y)
#                 if x.dtype.is_floating:
#                     return np.allclose(np_x, np_y, rtol=1e-5, atol=1e-5)
#                 elif x.dtype.is_integer:
#                     return np.equal(np_x, np_y).all()
#             except:
#                 raise ValueError(f"Comparison between {type(x)} is not supported now.")
#             return True
#         elif x_type == ArgType.FLOAT:
#             return abs(x - y) < 1e-5
#         elif x_type in [ArgType.LIST, ArgType.TUPLE]:
#             if len(x) != len(y):
#                 return False
#             for i in range(len(x)):
#                 if not PaddleLibrary.is_equal(x[i], y[i]):
#                     return False
#             return True
#
#         else:
#             try:
#                 flag = x == y
#             except:
#                 return True
#
#             if isinstance(flag, np.ndarray):
#                 flag = flag.all()
#             try:
#                 if flag:
#                     pass
#             except:
#                 flag = True
#             return flag

class PaddleLibrary(Library):
    def __init__(self, output_dir, diff_bound=1e-5, time_bound=10, time_thresold=1e-3) -> None:
        super().__init__(output_dir)
        self.diff_bound = diff_bound
        self.time_bound = time_bound
        self.time_thresold = time_thresold

    def test_with_oracle(self, api: PaddleAPI, oracle: OracleType):
        if oracle == OracleType.CRASH:
            # We need call another process to catch the crash error
            code = "import paddle\n"
            code += self.generate_code(api, oracle)
            with open(join(self.directory, "temp.py"), "w") as f:
                f.write(code)
            _, error = self.run_code(code)
            if error == None:
                self.write_to_dir(join(self.output[oracle], "success"), api.api, code)
            elif self.is_crash_msg(error):
                self.write_to_dir(join(self.output[oracle], "potential-bug"), api.api, code)
            else:
                self.write_to_dir(join(self.output[oracle], "fail"), api.api, code)
        elif oracle == OracleType.CUDA:
            code = "import paddle\n"
            code += api.to_code(res=f"{RES_KEY}[\"{RES_CPU_KEY}\"]", use_try=True,
                                error_res=f"{RES_KEY}[\"{ERR_CPU_KEY}\"]")
            code += api.to_diff_code(oracle, res=f"{RES_KEY}[\"{RES_GPU_KEY}\"]", use_try=True,
                                     error_res=f"{RES_KEY}[\"{ERR_GPU_KEY}\"]")

            write_code = "results = dict()\n" + code + "\nprint(results)\n"
            with open(join(self.directory, "temp.py"), "w") as f:
                f.write(write_code)

            results, error = self.run_code(code)

            write_dir = ""
            if error == None:
                # first check the correctness
                if results[ERR_CPU_KEY] == None and results[ERR_GPU_KEY] == None:
                    try:
                        is_equal = self.is_equal(results[RES_CPU_KEY], results[RES_GPU_KEY], self.diff_bound)
                    except Exception:
                        write_dir = join(self.output[oracle], "compare-bug")
                    else:
                        if is_equal:
                            write_dir = join(self.output[oracle], "success")
                        else:
                            write_dir = join(self.output[oracle], "potential-bug")
                elif self.is_crash_msg(results[ERR_CPU_KEY]) or self.is_crash_msg(results[ERR_GPU_KEY]):
                    write_dir = join(self.output[oracle], "potential-bug")
                elif results[ERR_CPU_KEY] and results[ERR_GPU_KEY]:
                    write_dir = join(self.output[oracle], "success")
                    pass
                elif self.is_error_msg(results[ERR_CPU_KEY]) != self.is_error_msg(results[ERR_GPU_KEY]):
                    write_dir = join(self.output[oracle], "potential-bug")
                else:
                    write_dir = join(self.output[oracle], "success")
            elif self.is_crash_msg(error):
                write_dir = join(self.output[oracle], "potential-bug")
            else:
                write_dir = join(self.output[oracle], "fail")
            self.write_to_dir(write_dir, api.api, write_code)
        elif oracle == OracleType.PRECISION:
            code = "import paddle\n"
            code += "import time\n"
            code += api.to_code(res=f"results[\"{TIME_LOW_KEY}\"]", low_precision=True)
            code += api.to_diff_code(oracle, res=f"results[\"{TIME_HIGH_KEY}\"]")

            write_code = "results = dict()\n" + code + "\nprint(results)\n"
            with open(join(self.directory, "temp.py"), "w") as f:
                f.write(write_code)

            results, error = self.run_code(code)
            if error == None:
                if isinstance(results[TIME_LOW_KEY], float) and isinstance(results[TIME_HIGH_KEY], float):
                    if results[TIME_LOW_KEY] > self.time_bound * results[TIME_HIGH_KEY] and results[
                        TIME_HIGH_KEY] > self.time_thresold:
                        write_dir = join(self.output[oracle], "potential-bug")
                    else:
                        write_dir = join(self.output[oracle], "success")
                else:
                    write_dir = join(self.output[oracle], "fail")
            else:
                write_dir = join(self.output[oracle], "fail")
            self.write_to_dir(write_dir, api.api, write_code)

    @staticmethod
    def generate_code(api: PaddleAPI, oracle: OracleType) -> str:
        if oracle == OracleType.CRASH:
            return api.to_code()
        elif oracle == OracleType.CUDA:
            code = api.to_code(res="cpu_res", use_try=True)
            code += api.to_diff_code(oracle, res="cuda_res", use_try=True)
            return code
        elif oracle == OracleType.PRECISION:
            code = api.to_code(res="low_res", low_precision=True)
            code += api.to_diff_code(oracle, res="high_res")
            return code
        else:
            assert (0)

    @staticmethod
    def run_code(code):
        results = dict()
        results[ERR_CPU_KEY] = None
        results[ERR_GPU_KEY] = None
        error = None
        try:
            exec(code)
        except Exception as e:
            error = str(e)
        return results, error

    @staticmethod
    def is_equal(x, y, diff_bound):
        def eq_float_tensor(x, y):
            # not strictly equal
            return np.allclose(x, y, rtol=1e-5, atol=1e-5)

        x_type = PaddleArgument.get_type(x)
        y_type = PaddleArgument.get_type(y)
        if x_type != y_type:
            if x_type == ArgType.PADDLE_TENSOR and y_type in [ArgType.LIST, ArgType.TUPLE]:
                flag = False
                for temp in y:
                    flag = flag or PaddleLibrary.is_equal(x, temp, diff_bound)
                return flag
            elif y_type == ArgType.PADDLE_TENSOR and x_type in [ArgType.LIST, ArgType.TUPLE]:
                flag = False
                for temp in x:
                    flag = flag or PaddleLibrary.is_equal(y, temp, diff_bound)
                return flag
            return False
        if x_type == ArgType.PADDLE_TENSOR:
            x = x.cpu()
            y = y.cpu()
            if x.dtype != y.dtype or x.shape != y.shape:
                return False
            if x.is_sparse:
                x = x.to_dense()
            if y.is_sparse:
                y = y.to_dense()
            if x.is_complex():
                if not y.is_complex(): return False
                return eq_float_tensor(x.real, y.real) and eq_float_tensor(
                    x.imag, y.imag)
            if not x.dtype.is_floating_point:
                return paddle.equal(x.cpu(), y.cpu())
            return eq_float_tensor(x, y)
        elif x_type == ArgType.FLOAT:
            return abs(x - y) < diff_bound
        elif x_type in [ArgType.LIST, ArgType.TUPLE]:
            if len(x) != len(y):
                return False
            for i in range(len(x)):
                if PaddleLibrary.is_equal(x[i], y[i], diff_bound) == False:
                    return False
            return True
        else:
            return x == y

    @staticmethod
    def is_error_msg(error_msg):
        allowed_msgs = ["not implement", "not support"]

        if error_msg == None:
            return False
        for msg in allowed_msgs:
            if msg in error_msg:
                return False
        return True

    @staticmethod
    def is_crash_msg(error_msg):
        if error_msg == None:
            return False
        if "INTERNAL ASSERT" in error_msg:
            return True
        else:
            return False
