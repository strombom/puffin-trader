
from enum import Enum


class Direction(Enum):
    up = 1
    down = 2


class OrderSide(Enum):
    long = 1
    short = -1


class RunnerEvent(Enum):
    nothing = 0
    direction_change_up = 1
    direction_change_down = -1
    overshoot_up = 2
    overshoot_down = 3


class EventType(Enum):
    direction_change = 1
    overshoot = 2

