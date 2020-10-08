
from enum import Enum


class Direction(Enum):
    up = 1
    down = -1


class OrderSide(Enum):
    long = 1
    short = -1


class RunnerEvent(Enum):
    overshoot_up = 2
    direction_change_up = 1
    nothing = 0
    direction_change_down = -1
    overshoot_down = -2


class EventType(Enum):
    direction_change = 1
    overshoot = 2

