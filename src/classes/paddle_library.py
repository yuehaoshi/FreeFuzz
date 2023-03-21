from classes.library import Library

import numpy as np

from classes.argument import Argument, ArgType
from classes.paddle_api import PADDLEAPI, PaddleArgument
from classes.library import Library
from classes.database import PaddleDatabase
from constants.enum import OracleType
from constants.keys import ERR_CPU_KEY, ERR_GPU_KEY, ERR_HIGH_KEY, ERR_LOW_KEY, ERROR_KEY, RES_CPU_KEY, RES_GPU_KEY, TIME_HIGH_KEY, TIME_LOW_KEY


#TODO: create paddle_library class
class TFLibrary(Library):
    @staticmethod
    def generate_code(api: PADDLEAPI, oracle: OracleType) -> str:
        code = ""
        if oracle == OracleType.CRASH:
            code += api.to_code_oracle(oracle=oracle)
            return code
        elif oracle == OracleType.CUDA:
            code += api.to_code_oracle(oracle=oracle)
            return code
        elif oracle == OracleType.PRECISION:
            code += api.to_code_oracle(oracle=oracle)
            return code
        else:
            assert 0

