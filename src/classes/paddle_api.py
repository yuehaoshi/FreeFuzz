import torch
from classes.argument import *
from classes.api import *
from classes.database import TorchDatabase
class PADDLEAPI(API):
    def __init__(self, api_name, record=None) -> None:
        super().__init__(api_name)
        self.record = TFDatabase.get_rand_record(api_name) if record is None else record
        self.args = TFAPI.generate_args_from_record(self.record)
        self.is_class = inspect.isclass(eval(self.api))

    def mutate(self, enable_value=True, enable_type=True, enable_db=True):
        num_arg = len(self.args)
        if num_arg == 0:
            return
        num_Mutation = randint(1, num_arg + 1)
        for _ in range(num_Mutation):
            arg_name = choice(list(self.args.keys()))
            arg = self.args[arg_name]

            if enable_type and do_type_mutation():
                arg.mutate_type()
            do_value_mutation = True
            if enable_db and do_select_from_db():
                new_arg, success = TFDatabase.select_rand_over_db(self.api, arg_name)
                if success:
                    new_arg = TFArgument.generate_arg_from_signature(new_arg)
                    self.args[arg_name] = new_arg
                    do_value_mutation = False
            if enable_value and do_value_mutation:
                arg.mutate_value()

    def to_code_oracle(self,
                       prefix="arg", oracle=OracleType.CRASH) -> str:

        if oracle == OracleType.CRASH:
            code = self.to_code(prefix=prefix, res_name=RESULT_KEY)
            return self.wrap_try(code, ERROR_KEY)
        elif oracle == OracleType.CUDA:
            cpu_code = self.to_code(prefix=prefix, res_name=RES_CPU_KEY,
                                    use_try=True, err_name=ERR_CPU_KEY, wrap_device=True, device_name="CPU")
            gpu_code = self.to_diff_code(prefix=prefix, res_name=RES_GPU_KEY,
                                         use_try=True, err_name=ERR_GPU_KEY, wrap_device=True, device_name="GPU:0")

            code = cpu_code + gpu_code
            return self.wrap_try(code, ERROR_KEY)
        elif oracle == OracleType.PRECISION:
            low_code = self.to_code(prefix=prefix, res_name=RES_LOW_KEY, low_precision=True,
                                    use_try=True, err_name=ERR_LOW_KEY, time_it=True, time_var=TIME_LOW_KEY)
            high_code = self.to_diff_code(prefix=prefix, res_name=RES_HIGH_KEY,
                                          use_try=True, err_name=ERR_HIGH_KEY, time_it=True, time_var=TIME_HIGH_KEY)
            code = low_code + high_code
            return self.wrap_try(code, ERROR_KEY)
        return ''
