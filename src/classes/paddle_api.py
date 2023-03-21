import torch
from classes.argument import *
from classes.api import *
from classes.database import TorchDatabase

# TODO: create paddle_api class
class PADDLEAPI(API):
    def __init__(self, api_name, record=None) -> None:
        super().__init__(api_name)
        pass

    def mutate(self, enable_value=True, enable_type=True, enable_db=True):
        pass

    def to_code_oracle(self, prefix="arg", oracle=OracleType.CRASH) -> str:
        pass
