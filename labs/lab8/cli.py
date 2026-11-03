import argparse
from dataclasses import fields, is_dataclass
from typing import get_args, get_origin, get_type_hints

from labs.lab8.entities import Lab8Config


def add_dataclass_args(parser: argparse.ArgumentParser, cls, prefix: str = "") -> None:
    """Add one --flag per leaf field; nested dataclasses become dotted prefixes."""
    hints = get_type_hints(cls)
    defaults = cls()
    for f in fields(cls):
        tp, name = hints[f.name], f"{prefix}{f.name}"
        if is_dataclass(tp):
            add_dataclass_args(parser, tp, f"{name}.")
        elif get_origin(tp) is tuple:
            args = get_args(tp)
            parser.add_argument(
                f"--{name}", dest=name, type=args[0], nargs=len(args),
                default=None, help=f"default: {getattr(defaults, f.name)}",
            )
        else:
            parser.add_argument(
                f"--{name}", dest=name, type=tp,
                default=None, help=f"default: {getattr(defaults, f.name)}",
            )


def build_dataclass(cls, ns: argparse.Namespace, prefix: str = ""):
    """Construct cls, overriding defaults only with flags the user passed."""
    hints = get_type_hints(cls)
    kwargs = {}
    for f in fields(cls):
        tp, name = hints[f.name], f"{prefix}{f.name}"
        if is_dataclass(tp):
            kwargs[f.name] = build_dataclass(tp, ns, f"{name}.")
        elif (value := vars(ns)[name]) is not None:
            kwargs[f.name] = tuple(value) if get_origin(tp) is tuple else value
    return cls(**kwargs)


def parse_config() -> Lab8Config:
    parser = argparse.ArgumentParser(description="Lab 8: epidemic models")
    add_dataclass_args(parser, Lab8Config)
    return build_dataclass(Lab8Config, parser.parse_args())
