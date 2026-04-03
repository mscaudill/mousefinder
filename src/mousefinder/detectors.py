""" """

import abc
from pathlib import Path
import time
from functools import partial
import multiprocessing as mp

import numpy as np
import numpy.typing as npt
from scipy.ndimage import maximum_filter
from skimage.filters.thresholding import threshold_minimum

from mousefinder.core.resources import allocate
from mousefinder.core import mixins
from mousefinder.configurations import Configuration, PCGC
from mousefinder import readers
from mousefinder.rois import ROI


class PCG(mixins.ReprMixin, mixins.SavingMixin):
    """ """

    def __init__(
        self,
        reader: readers.VideoReader,
        roi: ROI,
        config: Configuration,
    ) -> None:
        """ """

        self.reader = reader
        self.roi = roi
        self.configuration = config

        # parameters determined from estimate method
        self.threshold_: int | None = None
        self.mask_: npt.NDArray[np.bool_] | None = None

    def estimate(self, size=10, **kwargs):
        """kwargs passed to ndimage maximum filter and default footprint size is
        same as configuration's default"""

        img = self.reader.keyseek(0)
        x = img[*self.roi.region]
        x = maximum_filter(x, size=size, **kwargs)

        self.mask_ = self.roi.as_mask()
        self.threshold_ = threshold_minimum(x[self.mask_])

    def printable(self, msg: str, verbose: bool, end='\n', flush=True) -> None:
        """Prints a msg to std out if verbose.

        Args:
            msg:
                A string message to print.
            verbose:
                A boolean indicating if the message should be printed to stdout.
            end:
                The line ending following the message.
            flush:
                Boolean indicating if results should be flushed immediately to
                stdout.

        Returns:
            None
        """

        # pylint: disable-next=expression-not-assigned
        print(msg, end=end, flush=flush) if verbose else None

    def _detect(
        self,
        frame_tuple: tuple[int, npt.NDArray],
        size: int,
    ) -> npt.NDArray:
        #size is the smoothing size and mask is the circlar roi mask
        
        _, frame = frame_tuple
        x = frame[*self.roi.region]
        smoothed = maximum_filter(x, size=size)
        thresholded = smoothed < self.threshold_
        bool_im = np.logical_and(thresholded, self.mask_)
        result = np.mean(np.nonzero(bool_im), axis=-1)

        return result

    def __call__(
        self,
        *,
        path: Path | str | None = None,
        ncores: int | None = None,
        chunksize: int = 100,
        verbose: bool = True,
        returning: bool = True,
        **kwargs,
    ) -> npt.NDArray:
        """Concurrently detects mouse coordinates from each frame of this
        Detector's data.

        Args:
            path:
                A path or string to a dir where the coordinates will be saved
                to. If None, the coordinates will be saved to the same directory
                as the video data.
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
            returning:
                A boolean indicating if this call should return the coordinates
                in addition to saving them.

        Returns:
            An ndarray of shape (rois, 2, frames) containing the float row and
            column coordinates.
        """

        if self.threshold_ is None:
            msg = "Method 'fit' must be called prior to calling this Detector."
            raise RuntimeError(msg)

        # no hyperthread as VideoReader's already hyperthread
        core_cnt = allocate(self.reader.shape[0], ncores, hyperthread=False)
        size = kwargs.pop('size', 10)
        func = partial(
                self._detect,
                size=size,
        )

        results = []
        start = time.perf_counter()
        if core_cnt > 1:

            msg = f'Initializing Detector with {core_cnt} cores.'
            self.printable(msg, verbose)
            with mp.Pool(core_cnt) as pool:

                mapped = pool.imap(func, self.reader, chunksize)
                for idx, coords in enumerate(mapped, 1):
                    results.append(coords)
                    if idx > 1000:
                        break
                    # if verbose print every chunk completed
                    if idx % chunksize == 0:
                        #msg = f'Completed {idx} / {len(self.reader)} frames.'
                        msg = f'Completed {idx} frames.'
                        self.printable(msg, verbose, end='\r')

        else:

            for idx, arr in enumerate(self.reader, 1):
                if idx > 100:
                    break
                results.append(func(arr))
                msg = f'Frames completed {idx} / {len(self.reader)}'
                self.printable(msg, verbose, end='\r')

        msg = f'Detection completed in {time.perf_counter() - start} secs.'
        self.printable(msg, verbose)

        # shift the coordinates relative to ROI upper-left
        corner = np.array([self.roi.region[0].start, self.roi.region[1].start])
        result = np.array(results) + corner

        # use video data's dir if no path given
        target = self.reader.path.parent if not path else Path(path)
        # if the target is a file get its parent
        target = target.parent if not target.is_dir() else target
        # add a stem to target and save
        target = target / Path(self.reader.path.stem + '_coordinates')
        """
        self.save(
            target,
            result,
            path=self.reader.path,
            timestamp=self.reader.creation_time(),
            sample_rate=self.reader.sample_rate,
            scale=self.roi.scale,
            model=self.__class__.__name__,
        )
        """

        return result if returning else None




if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from matplotlib import animation

    base = '/media/matt/Magnus/PAC_Data/'
    name = '5879_Left_group B-S_no rest_video.webm'
    #name = '5895_Right_group B-S_video.webm'
    #name = 'No.6489 left_2022-02-09_13_55_22 (2).webm'
    #name = 'No.6503 right_2022-02-08_15_27_48.webm'
    path = base + name
    reader = readers.WebmReader(path)
    config = PCGC()
    roi = ROI.from_PCG(reader, config)
    model = PCG(reader, roi, config)
    model.estimate()
    results = model(ncores=2)
