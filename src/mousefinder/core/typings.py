"""A collection of protocols and custom types."""

from typing import protocol

# protocols are not part of public API
# pylint: disable = too-few-public-methods

class ImgThreshold(Protocol):
    """Protocol determining scalar float thresholds from 2D images."""

    def __call__(
        self,
        arr: npt.NDArray,
        *args,
        **kwargs,
    ) -> np.float64: ...

class ImgFilter(Protocol):
    """Protocol for functions that filter images."""

    def __call__(
        self,
        arr: npt.NDArray[np.float64],
        *args,
        **kwargs,
    ) -> npt.NDArray[np.float64]: ...

