""" """

from typing import Self
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.patches import Rectangle
from scipy.ndimage import maximum_filter
from skimage import exposure, filters, measure, morphology


from mousefinder.readers import WebmReader

class ROI:
    """ """

    def __init__(self, region: tuple[slice, slice], scale: float) -> None:
        """ """

        self.region = region
        self.scale = scale

    @property
    def center(self):
        """Returns the row, column center coordinate of this ROI."""

        return [(sl.start + sl.stop) // 2 for sl in self.region]

    @property
    def as_mask(self):
        """Returns a boolean array with a circular region that inscribes this
        ROI."""

        shape = [sl.stop - sl.start for sl in self.region]
        center = np.array((shape)) // 2
        y, x = np.ogrid[: shape[0], : shape[1]]
        squared_dist = (y - center[0]) ** 2 + (x - center[1]) ** 2
        squared_radius = min(shape - center) ** 2

        return squared_dist <= squared_radius

    @classmethod
    def from_video(
        cls,
        path: Path | str,
        scale: float,
        frames=[0, 10000],
        filt='sobel',
        size=10,
        Reader=WebmReader,
        thresholder=filters.threshold_li,
        **kwargs,
    ) -> Self:
        """ """

        reader = Reader(path)
        # remove mouse by MIP across 2 seperated images
        mip = np.max(reader.keyseek(frames), axis=0)
        # get edges & convolve removing wire & small structures
        filtfunc = getattr(filters, filt)
        edges = filtfunc(mip, mode='constant', cval=0)
        convolved = maximum_filter(edges, size=size)
        # threshold to binary image
        binary = convolved > thresholder(convolved, **kwargs)
        #label and return the largest region's bounding box
        labeled = morphology.label(binary)
        props = measure.regionprops(labeled)
        largest = props[np.argmax([r.area for r in props])]
        rows = largest.bbox[0], largest.bbox[2]
        columns = largest.bbox[1], largest.bbox[3]
        region = (slice(*rows), slice(*columns))

        return cls(region, scale)

    @classmethod
    def from_draw(cls, path: Path | str, frame: int=0) -> Self:
        """ """

        pass

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


if __name__ == '__main__':
    
    from mousefinder.readers import WebmReader
    from scipy.ndimage import maximum_filter
    from skimage.filters.thresholding import threshold_minimum
    import matplotlib.pyplot as plt
    import time


    plt.ion()
    base = '/media/matt/Magnus/PAC_Data/'
    #name = '5879_Left_group B-S_no rest_video.webm'
    #name = '5895_Right_group B-S_video.webm'
    #name = 'No.6489 left_2022-02-09_13_55_22 (2).webm'
    name = 'No.6503 right_2022-02-08_15_27_48.webm'
    path = base + name
    reader = WebmReader(path)
    img = reader.keyseek(20000)

    roi = ROI.from_video(path, scale=1)
    fig0, ax0 = plt.subplots()
    roi.plot(img, ax=ax0)

    fig1, ax1 = plt.subplots()
    sliced = img[*roi.region]
    convolved = maximum_filter(sliced, size=20)
    thr = threshold_minimum(convolved)
    binary = convolved<thr
    masked = binary * roi.as_mask
    ax1.imshow(masked)

    plt.show()

