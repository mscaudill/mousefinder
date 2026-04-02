"""A collection of dataclasses that hold data related to the type and dimensions
of specific recording chambers recorded from a top-down angle.

Currently supported chambers are:
    PCG:
        This is Pinnacle's circular gravel bottom chamber.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy import ndimage
from skimage import filters, measure, morphology

from mousefinder import readers
from mousefinder.rois import ROI


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
        path:
            The path to a video file for roi determination.
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
        size:
            The size of the kernel for smoothing away the dark spots in the
            gravel of the chamber and the electrode wires.

    """

    name: str = 'Pinnacle Circular Gravel'
    manufacturer: str = 'Pinnacle'
    material: str = 'plastic'
    bottom: str = 'gravel'
    shape: str = 'circle'
    height: float = 24
    width: float = 24
    size: int = 10

    # TODO protocol for filt and thresholder but skimage has not typed
    def roi(
        self,
        path: Path | str,
        frames: Sequence[int] = (0, -1),
        filt=filters.sobel,
        thresholder=filters.threshold_li,
        **kwargs,
    ) -> ROI:
        """Returns the circular region of interest of this chamber.

        Args:
            path:
                Path to a video file with this chamber configuration.
            frames:
                A sequence of frames whose max intensity projection will be
                taken as the background image (i.e. no mouse present). These
                frames should be separated by enough time so that the mouse is
                unlikely to be in the same position in each frame. Defaults to
                the first and last keyframes of the video.
            filt:
                An edge detection filter function consisting of two kernels for
                estimating the vertical and horizontal gradients.
            thresholder:
                Am skimage thresholding function for separating the mouse from
                the background.
            kwargs:
                An optional 'size' may be passed for roi detection that will
                supersede this configuration's size attribute and any
                valid kwarg for the thresholder function.

        Returns:
            An ROI instance.
        """

        size = kwargs.pop('size', self.size)

        reader = readers.WebmReader(path)
        # remove mouse by MIP across 2 separated images
        mip = np.max(reader.keyseek(frames), axis=0)
        # get edges & convolve removing wire & small structures
        edges = filt(mip, mode='constant', cval=0)
        convolved = ndimage.maximum_filter(edges, size=size)
        # threshold to binary image
        binary = convolved > thresholder(convolved, **kwargs)
        # label and return the largest region's bounding box
        labeled = morphology.label(binary)
        props = measure.regionprops(labeled)
        largest = props[np.argmax([r.area for r in props])]
        rows = largest.bbox[0], largest.bbox[2]
        columns = largest.bbox[1], largest.bbox[3]
        region = (slice(*rows), slice(*columns))
        scale = tuple(
            [
                (region[0].stop - region[0].start) / self.height,
                (region[1].stop - region[1].start) / self.width,
            ]
        )

        return ROI(region, scale=scale)


if __name__ == '__main__':

    base = '/media/matt/Magnus/PAC_Data/'
    # name = '5879_Left_group B-S_no rest_video.webm'
    # name = '5895_Right_group B-S_video.webm'
    # name = 'No.6489 left_2022-02-09_13_55_22 (2).webm'
    name = 'No.6503 right_2022-02-08_15_27_48.webm'
    path = base + name

    reader = readers.WebmReader(path)
    cfg = PCG()
    roi = cfg.roi(path)

    roi.plot(reader.keyseek(1000))
