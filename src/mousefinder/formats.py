"""A collection of pyav VideoFrame instance reformatters for converting VideoFrames to
numpy arrays.

PYAV provides a method called 'to_ndarray' for VideoFrame reformatting. However, this
method can be slow if converting between pixel formats or changing the size of
the arrays since a reformatter is called. These functions bypass the
reformatter and hence have more limited capabilities but greater speed.
"""

import typing

import numpy as np
import numpy.typing as npt
from av.video.frame import VideoFrame  # pylint: disable=no-name-in-module


# protocols are not part of public API
# pylint: disable-next = too-few-public-methods
class Formatter(typing.Protocol):
    """Protocol for functions that convert pyav VideoFrames to 2D np.uint8 arrays."""

    def __call__(self, frame: VideoFrame, *args, **kwargs) -> npt.NDArray: ...


# the pyav frame.to_ndarray return type is broad but yuvj420p stores 8-bit
# for read speed we need to ignore rather than narrow types here
@typing.no_type_check
def from_yuvj420p(frame: VideoFrame) -> npt.NDArray[np.uint8]:
    """Converts yuvj420p planar format VideoFrames to a 8-bit grayscale array.

    The YUV planar formats store the grayscale value in the Y luminance plane.
    This plane can represent pixels as full-swing (0-255) or limited (16-235).
    Traditionally, a 'j' in the format name distinguishes between full-swing and
    limited data but modern encoders describe the format as yuv420p even
    though the data is full-swing. PYAV automatically scales yuv420p data to
    be full-swing always. This rescaling takes time and in most cases is not necessary
    for data processing. This function ignores rescaling, so the range of the
    output may be (0, 255) or (16, 235) depending on whether the frame is
    full-swing or limited. Ignoring the rescaling speeds up reformatting ~10X.

    Args:
        frame:
            A pyav VideoFrame instance.

    Returns:
        A 2-D numpy array of 8-bit unsigned integers that may be in the range
        (0-255) or (16-235) depending on whether the frame represents pixels as
        full-swing or limited range.
    """

    planes = frame.to_ndarray()
    img = planes[:frame.height]

    return img

# the pyav frame.to_ndarray return type is broad but yuvj420p stores 8-bit
# for read speed we need to ignore rather than narrow types here
@typing.no_type_check
def from_yuv420p(frame: VideoFrame) -> npt.NDArray[np.uint8]:
    """Converts a limited range yuv420p VideoFrame to a full-swing 8-bit grayscale
    image.

    The yuv420p format is ambiguous about the range of the pixels since modern
    encoders call full-swing and limited pixel formats the same name. If the
    format is known to be limited range and you want the range rescaled to
    full-swing this is the function to use. It is a thin wrapper around pyav
    to_ndarray. It is ~10X slower than from_yuv420j due to the rescaling.

    Args:
        frame:
            A pyav VideoFrame instance.

    Returns:
        A 2-D numpy array of 8-bit unsigned integers rescaled to be in the range
        (0, 255).
    """

    return frame.to_ndarray('gray8')
