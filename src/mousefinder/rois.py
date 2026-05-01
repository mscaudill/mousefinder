"""A Region of Interest storing the row and column slices of images where
mouse detection will occur."""

from collections.abc import Callable, Sequence
from typing import Self

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.patches import Rectangle
from scipy import ndimage
from skimage import filters, measure, morphology

from mousefinder import configurations as configs
from mousefinder import readers


class ROI:
    """A region of interest for mouse position detection.

    Attrs:
        region:
            A 2-tuple of row and column slice instances.
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
        """Returns a boolean image of this ROI.

        Args:
            inscribed:
                A string shape name in {circle, square or rectangle} descrbing
                the shape to inscribe in the ROI with True values.

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
                A 2-D numpy array image to draw this ROI on top of.
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
    def from_PCG(  # pylint: disable=invalid-name, too-many-positional-arguments
        cls,
        reader: readers.VideoReader,
        config: configs.Configuration,
        frames: Sequence[int] = (0, 1000),
        filt=filters.sobel,
        size: int = 10,
        thresholder: Callable[[npt.NDArray], float] = filters.threshold_li,
        **kwargs,
    ) -> Self:
        """Returns the region of interest for a Pinnacle circular gravel
        bottomed chamber.

        Args:
            path:
                Path to a video file with this chamber configuration.
            frames:
                A sequence of frames whose max intensity projection will be
                taken as the background image (i.e. no mouse present). These
                frames should be separated by enough time so that the mouse is
                unlikely to be in the same position in each frame. Defaults to
                the first and 1000-th frames.
            filt:
                An edge detection filter function from ndimage or skimage
                libraries. Defaults to a sobel filter.
            size:
                The size of scipy's maximum_filter kernel in pixels for removing
                the gravel bed texture and possible electrode wires. This size
                should be smaller than the mouse but larger than the variations
                in the gravel bed. The default is 10 pixels.
            thresholder:
                A callable expected to accept an image and return a float
                threshold that segments the chambers bottom. This defaults to
                skimage's threshold_li function.
            kwargs:
                Keyword args are passed to thresholder function.

        Returns:
            An ROI instance.
        """

        # remove mouse by MIP across separated images
        imgs = np.stack([img for _, img in reader.keyseek(frames)])
        mip = np.max(imgs, axis=0)
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
                (region[0].stop - region[0].start) / config.height,
                (region[1].stop - region[1].start) / config.width,
            ]
        )

        return cls(region, scale=scale)

    @classmethod
    def from_draw(
        cls,
        reader: readers.VideoReader,
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
