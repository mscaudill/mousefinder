"""A Region of Interest storing the row and column slices of an image where
mouse detection will occur."""

from typing import Self

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.patches import Rectangle

from mousefinder.readers import VideoReader


class ROI:
    """A region of interest for mouse position detection.

    Attrs:
        region:
            A 2-tuple of row and column slice instances that span the image
            pixels containing the roi.
        scale:
            The pixels per cm conversion.
    """

    def __init__(
        self,
        region: tuple[slice, slice],
        scale: tuple[float, float],
    ) -> None:
        """Initialize this ROI with row & columns slices & pixel to cm scale.

        Args:
            region:
                A tuple of slices the row slice and the column slice.
            scale:
                The float number of pixels per cm.
        """

        self.region = region
        self.scale = scale

    def as_mask(self, inscribed='circle') -> npt.NDArray[np.bool_]:
        """Returns a boolean image of this ROI in which pixels within inscribed
        are True and False otherwise.

        Args:
            inscribed:
                A string shape of the region to inscribe inside this ROI's
                region where the mask will be True.

        Returns:
            A boolean image of the same shape as this ROI's region.
        """

        if inscribed.lower() not in 'circle square rectangle':
            msg = f'Mask of shape {inscribed} are not supported'
            raise ValueError(msg)

        shape = tuple(sl.stop - sl.start for sl in self.region)
        # rectangular mask
        if inscribed.lower() in ['square', 'rectangle']:
            return np.ones(shape)

        # circular mask
        center = np.array((shape)) // 2
        y, x = np.ogrid[: shape[0], : shape[1]]
        squared_dist = (y - center[0]) ** 2 + (x - center[1]) ** 2
        squared_radius = min(shape - center) ** 2
        bool_img: npt.NDArray[np.bool_] = squared_dist <= squared_radius

        return bool_img

    def plot(
        self,
        img: npt.NDArray,
        ax: plt.Axes | None = None,
        **kwargs,
    ) -> None:
        """Plots the region on top of an image.

        Args:
            img:
                A 2-D numpy array image of a plate to draw ROIs on top of.
            ax:
                A matplotlib axis instance. If None, a new axis, and figure, are
                created.
            kwargs:
                Any kwargs for matplotlib's Rectangle constructor.

        Returns:
            None
        """

        if ax is None:
            _, ax = plt.subplots()

        corner = self.region[1].start, self.region[0].start
        width = self.region[1].stop - self.region[1].start
        height = self.region[0].stop - self.region[0].start
        # default colors for rectangles if not provided
        ec = kwargs.pop('edgecolor', 'r')
        fc = kwargs.pop('facecolor', 'none')
        rect = Rectangle(corner, width, height, ec=ec, fc=fc, **kwargs)
        ax.imshow(img, cmap='gray')
        ax.add_patch(rect)

        plt.show()

    @classmethod
    def from_draw(
        cls,
        reader: VideoReader,
        frame: int,
        scale: tuple[float, float],
    ) -> Self:
        """Alternative constructor for creating an ROI from a drawn rectangle.

        Args:
            reader:
                A VideoReader instance.
            frame:
                The frame on which to draw the ROI.
            scale:
                The pixel per cm float value of frame.

        Returns:
            An ROI instance.
        """

        raise NotImplementedError
