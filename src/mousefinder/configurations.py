"""A collection of dataclasses that hold data related to the type and dimensions
of specific recording chambers recorded from a top-down angle.

Currently supported chambers are:
    PCG:
        This is Pinnacle's circular gravel bottom chamber.
"""

from dataclasses import dataclass


@dataclass
class Configuration:
    """A base dataclass configuration specifying the minimal attribute set
    required of all configurations.

    Attributes:
        name:
            The descriptive name of this dataclass.
        manufacturer:
            The name of this chamber's manufacturer.
        material:
            The string name of the material used in this chamber.
        bottom:
            The string name of the material that lines the chamber bottom.
        shape:
            The shape of the arena within the chamber.
        height:
            The vertical dimension of the chamber.
        width:
            The horizontal dimension of the chamber.
    """

    name: str
    manufacturer: str
    material: str
    bottom: str
    shape: str
    height: float
    width: float


@dataclass
class PCGC(Configuration):
    """A representation of Pinnacle's circular gravel bottomed chamber.

    Attributes:
        name:
            The descriptive name of this dataclass.
        manufacturer:
            The name of this chamber's manufacturer.
        material:
            The string name of the material used in this chamber.
        bottom:
            The string name of the material that lines the chamber bottom.
        shape:
            The shape of the arena within the chamber.
        height:
            The vertical dimension of the chamber.
        width:
            The horizontal dimension of the chamber.
    """

    name: str = 'Pinnacle Circular Gravel'
    manufacturer: str = 'Pinnacle'
    material: str = 'plastic'
    bottom: str = 'gravel'
    shape: str = 'circle'
    height: float = 24
    width: float = 24
