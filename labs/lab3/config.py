import argparse
from dataclasses import MISSING, Field, dataclass, field, fields, is_dataclass
from typing import Any, TypeVar

T = TypeVar("T", bound="Args")


def _parse_supports(raw: str) -> list[tuple[float, float]]:
    if not raw.strip():
        return []
    result: list[tuple[float, float]] = []
    for item in raw.split(","):
        pos_str, val_str = item.strip().split(":")
        result.append((float(pos_str), float(val_str)))
    return result


@dataclass(frozen=True)
class Args:
    @classmethod
    def from_namespace(
        cls: type[T], namespace: argparse.Namespace, prefix: str = ""
    ) -> T:
        kwargs: dict[str, Any] = {}
        for _field in fields(cls):
            if isinstance(_field.type, type) and issubclass(_field.type, Args):
                kwargs[_field.name] = _field.type.from_namespace(
                    namespace, prefix=f"{prefix}{_field.name}_"
                )
            else:
                kwargs[_field.name] = getattr(namespace, f"{prefix}{_field.name}")
        return cls(**kwargs)


# 📏 Одновимірна балка:
# довжина балки $L$, жорсткість на згин $EI$, інтенсивність навантаження $q$,
# кількість вузлів сітки $N$ та список координат або індексів внутрішніх опор.
@dataclass(frozen=True)
class BeamConfig(Args):
    L: float = field(
        default=1.0,
        metadata={"help": "Length of the 1D beam in meters."},
    )
    EI: float = field(
        default=1.0,
        metadata={"help": "Flexural rigidity of the beam."},
    )
    q: float = field(
        default=1.0,
        metadata={"help": "Distributed load intensity multiplier."},
    )
    N: int = field(
        default=1000,
        metadata={"help": "Number of internal grid nodes along the beam."},
    )
    supports: list[tuple[float, float]] = field(
        default_factory=lambda: [(1.0 / 3.0, 0.0), (2.0 / 3.0, 0.0)],
        metadata={
            "help": "Internal supports as 'x:deflection' pairs separated by commas.",
            "type": _parse_supports,
        },
    )


# 🟦 Двовимірна пластина:
# розміри сторін $L_x$ і $L_y$, кількість вузлів $N_x$ і $N_y$,
# циліндрична жорсткість $D$, навантаження $q$, точність зупинки $\varepsilon$,
# максимальна кількість ітерацій та параметр релаксації $\omega$.
@dataclass(frozen=True)
class PlateConfig(Args):
    L_x: float = field(
        default=1.0,
        metadata={"help": "Plate dimension along the X axis."},
    )
    L_y: float = field(
        default=1.0,
        metadata={"help": "Plate dimension along the Y axis."},
    )
    N_x: int = field(
        default=46,
        metadata={"help": "Total number of grid nodes along the X axis."},
    )
    N_y: int = field(
        default=46,
        metadata={"help": "Total number of grid nodes along the Y axis."},
    )
    D: float = field(
        default=1.0,
        metadata={"help": "Flexural rigidity of the 2D plate."},
    )
    q: float = field(
        default=-100.0,
        metadata={"help": "Uniform transverse load on the plate."},
    )
    epsilon: float = field(
        default=1e-9,
        metadata={"help": "Convergence tolerance for SOR iterations."},
    )
    max_iter: int = field(
        default=50000,
        metadata={"help": "Maximum number of SOR iterations."},
    )
    omega: float = field(
        default=1.85,
        metadata={"help": "Successive over-relaxation factor in range (1, 2)."},
    )


# ⭕ Кругла пластина в полярних координатах:
# радіуси кільця $r_a$ та $r_b$, кількість вузлів за радіусом $N_r$,
# циліндрична жорсткість $D$ та навантаження $q$.
@dataclass(frozen=True)
class PolarConfig(Args):
    r_a: float = field(
        default=1.0,
        metadata={"help": "Inner radius of the annular plate."},
    )
    r_b: float = field(
        default=2.0,
        metadata={"help": "Outer radius of the annular plate."},
    )
    N_r: int = field(
        default=200,
        metadata={"help": "Number of internal radial grid nodes."},
    )
    D: float = field(
        default=1.0,
        metadata={"help": "Flexural rigidity in polar coordinates."},
    )
    q: float = field(
        default=0.0,
        metadata={"help": "Axisymmetric transverse load intensity."},
    )


@dataclass(frozen=True)
class Config(Args):
    task: str = field(
        default="all",
        metadata={
            "help": "Simulation model to execute.",
            "choices": ("beam", "plate", "polar", "all"),
        },
    )
    beam: BeamConfig = field(
        default_factory=BeamConfig,
        metadata={"help": "1D continuous beam parameters"},
    )
    plate: PlateConfig = field(
        default_factory=PlateConfig,
        metadata={"help": "2D rectangular plate parameters"},
    )
    polar: PolarConfig = field(
        default_factory=PolarConfig,
        metadata={"help": "Axisymmetric polar plate parameters"},
    )


def _get_field_default(_field: Field[Any]) -> Any:
    if _field.default is not MISSING:
        return _field.default
    if _field.default_factory is not MISSING:
        return _field.default_factory()
    return MISSING


def _add_dataclass_arguments(
    target: argparse._ActionsContainer,
    cls: type[Any],
    prefix: str = "",
) -> None:
    for _field in fields(cls):
        if isinstance(_field.type, type) and is_dataclass(_field.type):
            group = target.add_argument_group(_field.metadata.get("help", _field.name))
            _add_dataclass_arguments(
                group, _field.type, prefix=f"{prefix}{_field.name}-"
            )
            continue

        default_val = _get_field_default(_field)
        flag_name = f"--{prefix}{_field.name.replace('_', '-')}"
        dest_name = f"{prefix.replace('-', '_')}{_field.name}"
        arg_type = _field.metadata.get("type", _field.type)
        help_text = _field.metadata.get("help", "")

        kwargs: dict[str, Any] = {
            "dest": dest_name,
            "type": arg_type,
            "help": f"{help_text} (default: {default_val})",
        }
        if default_val is not MISSING:
            kwargs["default"] = default_val
        if "choices" in _field.metadata:
            kwargs["choices"] = _field.metadata["choices"]

        target.add_argument(flag_name, **kwargs)


def parse_args(argv: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(
        description="Numerical modeling of elastic beam and plate bending."
    )
    _add_dataclass_arguments(parser, Config)
    args = parser.parse_args(argv)
    return Config.from_namespace(args)
