"""A Region of Interest storing the row and column slices of an image where
mouse detection will occur."""

from collections.abc import Sequence
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
    def from_PCG(
        cls,
        reader: readers.VideoReader,
        config: configs.Configuration,
        frames: Sequence[int] = (0, -1),
        filt=filters.sobel,
        size: int = 10,
        thresholder=filters.threshold_li,
        **kwargs,
    ) -> Self:
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
            size:
                The size of the kernel for smoothing away the dark spots in the
                gravel of the chamber and the electrode wires.
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

if __name__ == '__main__':

    base = '/media/matt/Magnus/PAC_Data/videos/'
    name = '5879_Left_group B-S_no rest_video.webm'
    # name = '5895_Right_group B-S_video.webm'
    # name = 'No.6489 left_2022-02-09_13_55_22 (2).webm'
    #name = 'No.6503 right_2022-02-08_15_27_48.webm'
    path = base + name

    reader = readers.WebmReader(path)
    config = configs.PCGC()
    roi = ROI.from_PCG(reader, config)
    roi.plot(reader.keyseek(10000))

