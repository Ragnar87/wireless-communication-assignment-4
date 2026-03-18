from dataclasses import dataclass

@dataclass(frozen=True)
class Node:
    id: int
    room_label: str
    coordinates: tuple[int, int]