# src/game/types.py
from typing import TypeAlias, List, Tuple, Callable

StateDeltaChange: TypeAlias = Tuple[str, ...]
StateDelta: TypeAlias = List[StateDeltaChange]
UndoInfo: TypeAlias = Callable[[], None]
