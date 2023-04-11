import paddle
from classes.argument import *
from classes.api import *
from classes.database import PaddleDatabase

class PaddleArgument(Argument):
    _support_types = [ArgType.PADDLE_DTYPE, ArgType.PADDLE_TENSOR, ArgType.PADDLE_OBJECT]

    _dtypes = [
        paddle.bfloat16, paddle.bool, paddle.complex128, paddle.complex64,
        paddle.uint8, paddle.int8, paddle.int16, paddle.int32, paddle.int64,
        paddle.float32, paddle.float64, paddle.float16
    ]

    def __init__(self,
                 value,
                 type: ArgType,
                 shape=None,
                 dtype=None,
                 max_value=1,
                 min_value=0):
        super().__init__(value, type)
        self.shape = shape
        self.dtype = dtype
        self.max_value = max_value
        self.min_value = min_value
    def to_code(self, var_name, low_precision=False, is_cuda=False) -> str:
        if self.type in [ArgType.LIST, ArgType.TUPLE]:
            code = ""
            arg_name_list = ""
            for i in range(len(self.value)):
                code += self.value[i].to_code(f"{var_name}_{i}", low_precision,
                                              is_cuda)
                arg_name_list += f"{var_name}_{i},"

            if self.type == ArgType.LIST:
                code += f"{var_name} = [{arg_name_list}]\n"
            else:
                code += f"{var_name} = ({arg_name_list})\n"
            return code
        elif self.type == ArgType.PADDLE_TENSOR:
            dtype = self.dtype
            max_value = self.max_value
            min_value = self.min_value
            if low_precision:
                dtype = self.low_precision_dtype(dtype)
                max_value, min_value = self.random_tensor_value(dtype)
            suffix = ""
            if is_cuda:
                suffix = ".cuda()"
            if dtype.is_floating_point:
                code = f"{var_name}_tensor = paddle.rand({self.shape}, dtype={dtype})\n"
            elif dtype.is_complex:
                code = f"{var_name}_tensor = paddle.rand({self.shape}, dtype={dtype})\n"
            elif dtype == paddle.bool:
                code = f"{var_name}_tensor = paddle.randint(0,2,{self.shape}, dtype={dtype})\n"
            else:
                code = f"{var_name}_tensor = paddle.randint({min_value},{max_value},{self.shape}, dtype={dtype})\n"
            code += f"{var_name} = {var_name}_tensor.clone(){suffix}\n"
            return code
        elif self.type == ArgType.PADDLE_OBJECT:
            return f"{var_name} = {self.value}\n"
        elif self.type == ArgType.PADDLE_DTYPE:
            return f"{var_name} = {self.value}\n"
        return super().to_code(var_name)

    def to_diff_code(self, var_name, oracle):
        """differential testing with oracle"""
        code = ""
        if self.type in [ArgType.LIST, ArgType.TUPLE]:
            code = ""
            arg_name_list = ""
            for i in range(len(self.value)):
                code += self.value[i].to_diff_code(f"{var_name}_{i}", oracle)
                arg_name_list += f"{var_name}_{i},"
            if self.type == ArgType.LIST:
                code += f"{var_name} = [{arg_name_list}]\n"
            else:
                code += f"{var_name} = ({arg_name_list})\n"
        elif self.type == ArgType.PADDLE_TENSOR:
            if oracle == OracleType.CUDA:
                code += f"{var_name} = {var_name}_tensor.clone().cuda()\n"
            elif oracle == OracleType.PRECISION:
                code += f"{var_name} = {var_name}_tensor.clone().type({self.dtype})\n"
        return code

    def mutate_value(self) -> None:
        if self.type == ArgType.PADDLE_OBJECT:
            pass
        elif self.type == ArgType.PADDLE_DTYPE:
            self.value = choice(self._dtypes)
        elif self.type == ArgType.PADDLE_TENSOR:
            self.max_value, self.min_value = self.random_tensor_value(
                self.dtype)
        elif self.type in super()._support_types:
            super().mutate_value()
        else:
            print(self.type, self.value)
            assert (0)

    def mutate_type(self) -> None:
        if self.type == ArgType.NULL:
            # choose from all types
            new_type = choice(self._support_types + super()._support_types)
            self.type = new_type
            if new_type == ArgType.LIST or new_type == ArgType.TUPLE:
                self.value = [
                    PaddleArgument(2, ArgType.INT),
                    PaddleArgument(3, ArgType.INT)
                ]
            elif new_type == ArgType.PADDLE_TENSOR:
                self.shape = [2, 2]
                self.dtype = paddle.float32
            elif new_type == ArgType.PADDLE_DTYPE:
                self.value = choice(self._dtypes)
            elif new_type == ArgType.PADDLE_OBJECT:
                pass
            else:
                self.value = super().initial_value(new_type)
        elif self.type == ArgType.PADDLE_TENSOR:
            new_size = list(self.shape)
            # change the dimension of tensor
            if change_tensor_dimension():
                if add_tensor_dimension():
                    new_size.append(1)
                elif len(new_size) > 0:
                    new_size.pop()
            # change the shape
            for i in range(len(new_size)):
                if change_tensor_shape():
                    new_size[i] = self.mutate_int_value(new_size[i], _min=0)
            self.shape = new_size
            # change dtype
            if change_tensor_dtype():
                self.dtype = choice(self._dtypes)
                self.max_value, self.min_value = self.random_tensor_value(self.dtype)
        elif self.type == ArgType.PADDLE_OBJECT:
            pass
        elif self.type == ArgType.PADDLE_DTYPE:
            self.value = choice(self._dtypes)
        elif self.type in super()._support_types:
            super().mutate_type()
        else:
            print(self.type, self.value)
            assert (0)

    # @staticmethod
    # def generate_arg_from_signature(signature):
    #     """Generate a Torch argument from the signature"""
    #     if signature == "paddleTensor":
    #         return PaddleArgument(None,
    #                              ArgType.PADDLE_TENSOR,
    #                              shape=[2, 2],
    #                              dtype=paddle.float32)
    #     if signature == "paddledtype":
    #         return PaddleArgument(choice(PaddleArgument._dtypes),
    #                              ArgType.PADDLE_DTYPE)
    #     if isinstance(signature, str) and signature == "paddledevice":
    #         value = paddle.set_device("cpu")
    #         return PaddleArgument(value, ArgType.TORCH_OBJECT)
    #     paddle.strided_slice()
    #     if isinstance(signature, str) and signature == "torch.strided":
    #         return TorchArgument("torch.strided", ArgType.TORCH_OBJECT)
    #     if isinstance(signature, str) and signature.startswith("torch."):
    #         value = eval(signature)
    #         if isinstance(value, torch.dtype):
    #             return TorchArgument(value, ArgType.TORCH_DTYPE)
    #         elif isinstance(value, torch.memory_format):
    #             return TorchArgument(value, ArgType.TORCH_OBJECT)
    #         print(signature)
    #         assert (0)
    #     if isinstance(signature, bool):
    #         return TorchArgument(signature, ArgType.BOOL)
    #     if isinstance(signature, int):
    #         return TorchArgument(signature, ArgType.INT)
    #     if isinstance(signature, str):
    #         return TorchArgument(signature, ArgType.STR)
    #     if isinstance(signature, float):
    #         return TorchArgument(signature, ArgType.FLOAT)
    #     if isinstance(signature, tuple):
    #         value = []
    #         for elem in signature:
    #             value.append(TorchArgument.generate_arg_from_signature(elem))
    #         return TorchArgument(value, ArgType.TUPLE)
    #     if isinstance(signature, list):
    #         value = []
    #         for elem in signature:
    #             value.append(TorchArgument.generate_arg_from_signature(elem))
    #         return TorchArgument(value, ArgType.LIST)
    #     # signature is a dictionary
    #     if isinstance(signature, dict):
    #         if not ('shape' in signature.keys()
    #                 and 'dtype' in signature.keys()):
    #             raise Exception('Wrong signature {0}'.format(signature))
    #         shape = signature['shape']
    #         dtype = signature['dtype']
    #         # signature is a ndarray or tensor.
    #         if isinstance(shape, (list, tuple)):
    #             if not dtype.startswith("torch."):
    #                 dtype = f"torch.{dtype}"
    #             dtype = eval(dtype)
    #             max_value, min_value = TorchArgument.random_tensor_value(dtype)
    #             return TorchArgument(None,
    #                                  ArgType.TORCH_TENSOR,
    #                                  shape,
    #                                  dtype=dtype,
    #                                  max_value=max_value,
    #                                  min_value=min_value)
    #         else:
    #             return TorchArgument(None,
    #                                  ArgType.TORCH_TENSOR,
    #                                  shape=[2, 2],
    #                                  dtype=torch.float32)
    #     return TorchArgument(None, ArgType.NULL)

    def to_code_oracle(self, prefix="arg", oracle=OracleType.CRASH) -> str:
        pass

    @staticmethod
    def low_precision_dtype(dtype):
        if dtype in [paddle.int16, paddle.int32, paddle.int64]:
            return paddle.int8
        elif dtype in [paddle.float32, paddle.float64]:
            return paddle.float16
        elif dtype in [paddle.complex128]:
            return paddle.complex64
        return dtype

    @staticmethod
    def random_tensor_value(dtype):
        max_value = 1
        min_value = 0
        if dtype == paddle.bool:
            max_value = 2
            min_value = 0
        elif dtype == paddle.uint8:
            max_value = 1 << randint(0, 9)
            min_value = 0
        elif dtype == paddle.int8:
            max_value = 1 << randint(0, 8)
            min_value = -1 << randint(0, 8)
        elif dtype == paddle.int16:
            max_value = 1 << randint(0, 16)
            min_value = -1 << randint(0, 16)
        else:
            max_value = 1 << randint(0, 16)
            min_value = -1 << randint(0, 16)
        return max_value, min_value

    @staticmethod
    def get_type(x):
        res = Argument.get_type(x)
        if res != None:
            return res
        if isinstance(x, paddle.Tensor):
            return ArgType.PADDLE_TENSOR
        elif isinstance(x, paddle.dtype):
            return ArgType.PADDLE_DTYPE
        else:
            return ArgType.PADDLE_OBJECT
# TODO: create paddle_api class
class PADDLEAPI(API):
    def __init__(self, api_name, record=None) -> None:
        super().__init__(api_name)
        pass

    def mutate(self, enable_value=True, enable_type=True, enable_db=True):
        pass

    def to_code_oracle(self, prefix="arg", oracle=OracleType.CRASH) -> str:
        pass
