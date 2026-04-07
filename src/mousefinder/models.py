"""A collection of mouse center-of-mass detection models.

This collection consist of:
    PCG:
        A model for detecting the mouse center-of-mass in the Pinnacle circular
        chamber with a gravel bed and camera viewing angle from top.
"""

import multiprocessing as mp
import time
from functools import partial
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.ndimage import maximum_filter
from skimage import measure
from skimage.filters.thresholding import threshold_minimum

from mousefinder import readers
from mousefinder.configurations import PCGC, Configuration
from mousefinder.core import mixins
from mousefinder.core.resources import allocate
from mousefinder.rois import ROI


class PCG(mixins.ReprMixin, mixins.SavingMixin, mixins.PrintMixin):
    """A model for mouse center-of-mass detection for Pinnacle's circular
    chamber with a gravel bottom and a top-down camera angle.

    Attrs:
        reader:
            An iterable VideoReader instance (see mousefinder.readers).
        roi:
            A region of interest (ROI) instance (see mousefinder.rois)
        config:
            A chamber configuration data class. For this model this will usually
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

        self.threshold_: int | None = None
        self.mask_: npt.NDArray[np.bool_] | None = None

    def estimate(self, size: int = 10, **kwargs) -> None:
        """Estimates the integer threshold that best distinguishes the mouse
        pixels from the background pixels.

        Args:
            size:
                The size of the kernel used to smooth the pixel value
                discontinuities in the gravel and the electrode wires. The
                default value of 10 pixels is appropriate for the PCGC
                configuration.
            kwargs:
                Any valid kwarg for scipy's ndimage.maximum_filter that is used
                for smoothing.

        Returns:
            None
        """

        img = self.reader.keyseek(0)
        x = img[*self.roi.region]
        x = maximum_filter(x, size=size, **kwargs)

        # store threshold and roi mask to this instance and narrow types
        self.mask_ = self.roi.as_mask()
        self.threshold_ = threshold_minimum(x[self.mask_])

    def _worker(
        self,
        frame_tuple: tuple[int, npt.NDArray],
        size: int,
    ) -> npt.NDArray:
        """Detects the mouse's center-of-mass on a single frame of data.

        Args:
            frame_tuple:
                A frame index and image frame 2-tuple yielded by this Model's
                reader during iteration.
            size:
                The size of the kernel used to reduce discontinuities in the
                pixel values of the frame.

        Returns:
            A 2-el array of row and column indices where the mouse's
            center-of-mass is predicted to be.
        """

        _, frame = frame_tuple
        x = frame[*self.roi.region]
        smoothed = maximum_filter(x, size=size)
        thresholded = smoothed < self.threshold_
        # no typing here for speed, instance in __call__ ensures mask not None
        bool_im = np.logical_and(thresholded, self.mask_)  # type: ignore

        # get the largest labeled regions centroid
        labeled = measure.label(bool_im)
        regions = measure.regionprops(labeled, intensity_image=x)
        idx = np.argmax([r.intensity_std for r in regions])

        result: npt.NDArray[np.float64] = regions[idx].centroid
        return result

    def __call__(
        self,
        *,
        size: int = 20,
        path: Path | str | None = None,
        ncores: int | None = None,
        chunksize: int = 100,
        verbose: bool = True,
        saving: bool = True,
    ) -> npt.NDArray:
        """Concurrently detects mouse coordinates from each frame of this
        Detector's data.

        Args:
            size:
                The size of the kernel used to reduce discontinuities in the
                pixel values of the frame. This value should be large enough to
                smooth neighboring dark values in the gravel bed while being
                small enough to not blur the mouse position unreasonably. For
                the PCGC configuration, the default value is 20 pixel smoothing.
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

        if any([el is None for el in (self.threshold_, self.mask_)]):
            msg = (
                "Method 'estimate' must be called prior to calling this Model."
            )
            raise RuntimeError(msg)

        # narrow the estimated paramter types
        assert isinstance(self.mask_, np.ndarray)
        assert isinstance(self.threshold_, np.int64)

        # no hyperthread as VideoReader's already hyperthread
        core_cnt = allocate(self.reader.shape[0], ncores, hyperthread=False)
        func = partial(self._worker, size=size)

        results = []
        start = time.perf_counter()
        if core_cnt > 1:

            msg = f'Initializing Detection with {core_cnt} cores.'
            self.printable(msg, verbose)
            with mp.Pool(core_cnt) as pool:

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
        result = np.array(results) + corner

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


if __name__ == '__main__':

    base = '/media/matt/compute/PAC_Data/videos/'
    #name = '5879_Left_group B-S_no rest_video.webm'
    name = '5895_Right_group B-S_video.webm'
    # name = 'No.6489 left_2022-02-09_13_55_22 (2).webm'
    # name = 'No.6503 right_2022-02-08_15_27_48.webm'
    path = base + name
    reader = readers.WebmReader(path)
    config = PCGC()
    roi = ROI.from_PCG(reader, config)
    model = PCG(reader, roi, config)
    model.estimate()
    results = model(ncores=2, saving=False, chunksize=100)
