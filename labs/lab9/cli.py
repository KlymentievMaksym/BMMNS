import argparse
import types
from dataclasses import fields, is_dataclass
from typing import Union, get_args, get_origin, get_type_hints

from labs.lab9.entities import Lab9Config


def add_dataclass_args(parser: argparse.ArgumentParser, cls, prefix: str = "") -> None:
    hints = get_type_hints(cls)
    defaults = cls()
    for f in fields(cls):
        tp, name = hints[f.name], f"{prefix}{f.name}"
        help_text = f"default: {getattr(defaults, f.name)}"

        if is_dataclass(tp):
            add_dataclass_args(parser, tp, f"{name}.")

        elif get_origin(tp) is tuple:
            args = get_args(tp)
            variadic = args[-1] is Ellipsis
            parser.add_argument(
                f"--{name}",
                dest=name,
                type=args[0],
                nargs="+" if variadic else len(args),
                default=None,
                help=help_text,
            )

        elif get_origin(tp) in (Union, types.UnionType):
            inner = [a for a in get_args(tp) if a is not type(None)]
            arg_type = (
                inner[0] if len(inner) == 1 and inner[0] in (int, float, str) else str
            )
            parser.add_argument(
                f"--{name}", dest=name, type=arg_type, default=None, help=help_text
            )

        else:
            parser.add_argument(
                f"--{name}", dest=name, type=tp, default=None, help=help_text
            )


def build_dataclass(cls, ns: argparse.Namespace, prefix: str = ""):
    hints = get_type_hints(cls)
    kwargs = {}
    for f in fields(cls):
        tp, name = hints[f.name], f"{prefix}{f.name}"

        if is_dataclass(tp):
            kwargs[f.name] = build_dataclass(tp, ns, f"{name}.")

        elif (value := vars(ns)[name]) is not None:
            kwargs[f.name] = tuple(value) if get_origin(tp) is tuple else value

    return cls(**kwargs)


def parse_config() -> Lab9Config:
    parser = argparse.ArgumentParser(description="Lab 9: social processes")
    add_dataclass_args(parser, Lab9Config)
    return build_dataclass(Lab9Config, parser.parse_args())
