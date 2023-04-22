
from dataclasses import dataclass

from .alias import nano
from .сharacteristic import WindowCharacteristic


@dataclass(frozen=True)
class Filter:
    span: tuple[nano, nano]

