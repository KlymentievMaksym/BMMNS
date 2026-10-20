import argparse
from dataclasses import fields

import numpy as np

from labs.lab2.entities import WaveSimConfig


def parse_math_expr(val: str) -> float:
    try:
        return float(val)
    except ValueError:
        pass

    allowed_names = {
        "np": np,
        "pi": np.pi,
        "e": np.e,
        "sin": np.sin,
        "cos": np.cos,
    }

    try:
        code = compile(val, "<string>", "eval")
        for name in code.co_names:
            if name not in allowed_names:
                raise NameError(f"Use of '{name}' is not allowed in expression")
        return float(eval(code, {"__builtins__": {}}, allowed_names))
    except Exception as e:  # noqa: BLE001
        raise argparse.ArgumentTypeError(
            f"Invalid float or math expression: '{val}'. Error: {e}"
        )

def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="2D Wave Equation Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    for field in fields(WaveSimConfig):
        arg_type = parse_math_expr if field.type is float else field.type
        parser.add_argument(
            f"--{field.name}",
            type=arg_type,
            default=field.default,
            help=f"Parameter {field.name}",
        )
        
    return parser.parse_args()