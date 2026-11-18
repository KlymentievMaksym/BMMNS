# 📏 Одновимірна балка: довжина балки $L$, жорсткість на згин $EI$, інтенсивність розподіленого навантаження $q$, кількість вузлів сітки $N$ та список координат або індексів внутрішніх опор.
# 🟦 Двовимірна пластина: розміри сторін $L_x$ і $L_y$, кількість вузлів $N_x$ і $N_y$, циліндрична жорсткість $D$, навантаження $q$, точність зупинки $\varepsilon$, максимальна кількість ітерацій та параметр релаксації $\omega$.
# ⭕ Кругла пластина в полярних координатах: радіус диска $R$, кількість вузлів за радіусом $N_r$, циліндрична жорсткість $D$ та навантаження $q$.
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class BeamConfig:
    pass

@dataclass(frozen=True)
class PlateConfig:
    pass

@dataclass(frozen=True)
class PolarConfig:
    pass

@dataclass(frozen=True)
class Config:
    pass
