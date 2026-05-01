"""A collection of center-of-mass detection models.

This collection consist of:
    PCGTop:
        A model for detecting the mouse center-of-mass in the Pinnacle circular
        chamber with a gravel bed and top-down camera viewing angle.
"""

import multiprocessing as mp
import time
import typing
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter, label, minimum_filter
from skimage import measure
from skimage.filters.thresholding import threshold_li

from mousefinder import readers
from mousefinder.configurations import Configuration
from mousefinder.core import mixins
from mousefinder.core.resources import allocate
from mousefinder.rois import ROI


class PCGTop(mixins.ReprMixin, mixins.SavingMixin, mixins.PrintMixin):
    """A model for center-of-mass detection in Pinnacle's circular
    chamber with a gravel bottom and a top-down camera angle.

    Attrs:
        reader:
            An iterable VideoReader instance (see mousefinder.readers).
        roi:
            A region of interest (ROI) instance (see mousefinder.rois)
        config:
            A chamber configuration data class. For this model, this will usually
            be the PCGC configuration.
    """

    def __init__(
        self,
        reader: readers.VideoReader,
        roi: ROI,
        config: Configuration,
    ) -> None:
        """Initialize this model with a reader, an roi and configuration."""

        self.reader = reader
        self.roi = roi
        self.configuration = config
        self._ctx = mp.get_context('spawn')

        self.sigma_: float | None = None
        self.threshold_: float | None = None

    def estimate(
        self,
        sigma: int | None = None,
        thresholder: Callable[[npt.NDArray], float] = threshold_li,
        **kwargs,
    ) -> None:
        """Estimates the float threshold that best distinguishes the mouse
        from the background from an illumniation normalized image.

        The threshold computed by this method is the threshold after adjusting
        for light intensity changes by gaussian blur correction.

        Args:
            sigma:
                The standard deviation of the gaussian for correcting the uneven
                illumination. If None, this value defaults to 1/20 the height of
                the image.
            thresholder:
                A callable thresholding function that accepts an image and
                returns a integer or float value. The default is the
                threshold_li function from the skimage library.
            kwargs:
                All keyword arguments are passed to the thresholder.

        Returns:
            None
        """

        _, frame = self.reader.keyseek(0)[0]
        self.sigma_ = frame.shape[0] // 20 if sigma is None else sigma

        img = frame[*self.roi.region]
        blurred = gaussian_filter(img, sigma=self.sigma_)
        corrected = img / blurred
        self.threshold_ = thresholder(corrected, **kwargs)

    @typing.no_type_check
    def _worker(
        self,
        indexed_frame: tuple[int, npt.NDArray],
        minsize: int,
    ) -> npt.NDArray:
        """Detects the mouse's center-of-mass on a single frame of data.

        Args:
            indexed_frame:
                A frame index and image frame 2-tuple yielded by this Model's
                reader during iteration.
            minsize:
                Objects below minsize x minsize pixels will be removed prior to
                segmentation.

        Returns:
            A 2-el array of row and column indices where the mouse's
            center-of-mass is predicted to be relative to this model's roi.
        """

        _, frame = indexed_frame
        x = frame[*self.roi.region].astype(float)
        # correct lighting, smooth out gravel and threshold
        corrected = x / gaussian_filter(x, self.sigma_)
        smoothed = gaussian_filter(corrected, sigma=minsize)  # type: ignore
        bool_img = smoothed < self.threshold_  # type: ignore

        # remove small detections
        bool_img = minimum_filter(bool_img, size=minsize)

        labeled, cnt = label(bool_img)
        if cnt == 0:
            return np.nan * np.ones(2)

        if cnt == 1:
            return np.mean(np.nonzero(bool_img), axis=-1)

        # get largest 2 regions
        regions = measure.regionprops(labeled)
        sorted_regions = sorted(regions, key=lambda r: r.area, reverse=True)[:2]
        ratios = [
            r.area * r.perimeter / max(1, measure.perimeter(r.image_convex))
            for r in sorted_regions
        ]

        return sorted_regions[np.argmax(ratios)].centroid

    def detect(
        self,
        *,
        minsize: int = 10,
        path: Path | str | None = None,
        ncores: int | None = None,
        chunksize: int = 100,
        verbose: bool = True,
        saving: bool = True,
    ) -> npt.NDArray:
        """Concurrently detects mouse coordinates from each frame of this
        Detector's data.

        Args:
            minsize:
                The minimum size in pixels along rows and columns that objects
                within the images must achieve to be included in the
                segmentation. Objects below minsize x minsize pixels are
                excluded from detection. The default is 10 pixels.
            path:
                A path or string to a dir where the coordinates will be saved
                to. If None, the coordinates will be saved to the same directory
                as the video data. The name of the saved file will match the
                this Model's reader path name but with '_coordinates' appended
                to the name.
            ncores:
                The number of processing cores to utilize. If None, the total
                number of available cores will be used. If 1, a single core will
                be used and the data will be processed non-concurrently.
            chunksize:
                The (approximate) number of frames to submit to a processing
                worker for detection at one time. This reduces data transfers
                and can significantly speed-up the detection but consumes
                memory. The default value of 100 frames per processor is a good
                balance.
            verbose:
                Boolean indicating if progress should be printed to stdout.
            saving:
                A boolean indicating if the coordinates should be saved to path
                before returning them.

        Returns:
            An ndarray of shape (n, 2) containing the float row and
            column coordinates for each of the n frames of this Model's reader.
        """

        if self.threshold_ is None:
            msg = (
                "Method 'estimate' must be called prior to calling this Model."
            )
            raise RuntimeError(msg)

        # narrow the estimated parameter types
        assert isinstance(self.threshold_, np.float64)
        assert isinstance(self.sigma_, (int, np.float64))

        # no hyperthread as VideoReader's already hyperthread
        core_cnt = allocate(self.reader.shape[0], ncores, hyperthread=False)
        func = partial(self._worker, minsize=minsize)

        results = []
        start = time.perf_counter()
        if core_cnt > 1:

            msg = f'Initializing Detection with {core_cnt} cores.'
            self.printable(msg, verbose)
            with self._ctx.Pool(core_cnt) as pool:

                mapped = pool.imap(func, self.reader, chunksize)
                for idx, coords in enumerate(mapped, 1):
                    results.append(coords)
                    # if verbose print every chunk completed
                    if idx % chunksize == 0:
                        msg = f'Completed {idx} / {len(self.reader)} frames.'
                        self.printable(msg, verbose, end='\r')

        else:

            for idx, arr in enumerate(self.reader, 1):
                results.append(func(arr))
                msg = f'Frames completed {idx} / {len(self.reader)}'
                self.printable(msg, verbose, end='\r')

        msg = f'Detection completed in {time.perf_counter() - start} secs.'
        self.printable(msg, verbose)

        # shift the coordinates relative to ROI upper-left
        corner = np.array([self.roi.region[0].start, self.roi.region[1].start])
        result: npt.NDArray = np.array(results) + corner

        if saving:

            # use video data's dir if no path given
            target = self.reader.path.parent if not path else Path(path)
            # if the target is a file get its parent
            target = target.parent if not target.is_dir() else target
            # add a stem to target and save
            target = target / Path(self.reader.path.stem + '_coordinates')

            self.save(
                target,
                result,
                path=self.reader.path,
                timestamp=self.reader.creation_time(),
                sample_rate=self.reader.sample_rate,
                scale=self.roi.scale,
                model=self.__class__.__name__,
            )

        return result
