""" """

from collections.abc import Iterator
from itertools import islice
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import numpy.typing as npt

from mousefinder.core import mixins
from mousefinder.readers import VideoReader


class MPLWriter(mixins.ReprMixin):
    """A Matplotlib based writer of video and coordinate data.

    Attributes:
        reader:
            An iterable VideoReader instance.
        coords:
            A 3D array of shape (rois, 2, frames) of coordinates in row, column
            format along axis 1.
        indices:
            A range instance of the frames to write.
    """

    def __init__(
        self,
        reader: VideoReader,
        coords: npt.NDArray | None,
        indices: range | None = None,
        show_frame: bool = False,
        label: bool = False,
        figsize: tuple[int, int] | None = None,
        imageprops: dict[str, str] | None = None,
        **markerprops: dict[str, str],
    ) -> None:
        """Initialize this writer with a reader, an array of coordinates and
        marker kwargs.

        Args:
            reader:
                An iterable VideoReader instance.
            coords:
                A rois x 2 x frames array of coordinates in row, col format. May
                be None if no coordinate data to be drawn.
            indices:
                A range instance of the frames to write or None. If None, all
                image frames in reader and coordinates will be written.
            show_frame:
                Boolean indicating if a matplotlib axis should be drawn. If
                drawn, the axis is drawn to each frame during save or plot.
            label:
                Boolean indicating if the frame index should be placed into
                a legend in the upper right of each frame.
            figsize:
                The size if the images to write. If None, the original image
                size is used.
            imageprops:
                Any valid kwarg for matplotlib's imshow function.
            **markerprops:
                Optional marker properties for coordinate data passed to
                matplotlib scatter function.

        Returns:
            None
        """

        self.reader = reader
        self.coords = np.full((1,2,1), np.nan) if coords is None else coords
        self.indices = range(len(self.reader) - 1) if not indices else indices
        self.show_frame = show_frame
        self.label = label
        self._imageprops = imageprops if imageprops else {}
        self._markerprops = markerprops

        # make an iterator of image numbers and images to draw
        self._data = self._initialize_data(self.indices)

        # make figure and construct initial draw
        self._fig, self._ax = self._configure(figsize)
        self._imgplot, self._dotplot, self._legend = self._initial_draw()

    def _initialize_data(
        self,
        indices: range,
    ) -> Iterator[tuple[int, npt.NDArray]]:
        """Returns a iterator of image numbers and image arrays from this
        Writer's reader that are within indices.

        Args:
            indices:
                A range instance of enumerated image frames to draw.

        Returns:
            An iterator of tuples.
        """

        enumerated = iter(self.reader)
        return islice(enumerated, indices.start, indices.stop, indices.step)

    def _configure(self, figsize):
        """Configures the figure and axis of this Writer."""

        fig, ax = plt.subplots(figsize=figsize)
        if self.show_frame:
            return fig, ax

        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.set_axis_off()

        return fig, ax

    def _initial_draw(self):
        """Make the initial draw of the first frame and first coordinate

        Returns:
            Handles to the image plot, the scatter plot and a legend which may
            be None type.
        """

        idx, img = next(self._data)
        imgplot = self._ax.imshow(img, **self._imageprops)
        xs, ys = self.coords[0, 1], self.coords[0, 0]
        dotplot = self._ax.scatter(xs, ys, **self._markerprops)
        legend = None
        if self.label:
            dotplot.set_label(f'Frame {idx}')
            legend = self._ax.legend(loc='upper right')

        return imgplot, dotplot, legend

    def _update(self, data: tuple[int, npt.NDArray], verbose: bool):
        """Update the image, coordinates and possible legends with new data.

        Args:
            data:
                A tuple, the image number and image data to draw on this update.
            verbose:
                Boolean to indicate how many frames update has completed as
                a print to stdout. This is useful when saving data to know this
                Writer's progress.
        """

        idx, img = data
        if verbose:
            count = idx - self.indices.start
            msg = f'Frame {count / len(self.indices) * 100:0.2f}% complete'
            print(msg, end='\r', flush=True)

        self._imgplot.set_array(img)
        try:
            # + 1 because init_draw has shown first image
            xs, ys = self.coords[idx + 1, 1], self.coords[idx + 1, 0]
        except IndexError:
            # get the default nan coords if we run out of coords
            xs, ys = self.coords[1, 0], self.coords[0, 0]

        offsets = np.stack((xs, ys), axis=-1)
        self._dotplot.set_offsets(offsets)
        if self.label:
            self._legend.get_texts()[0].set_text(f'Frame {idx}')

    def preview(self, fps: int | None = None):
        """Opens an animation that displays the data and coordinates to be
        written.

        Args:
            fps:
                The frames per second if the animation. If None, the reader's
                sample rate will be used.
        """

        func = partial(self._update, verbose=False)
        fps = self.reader.sample_rate if not fps else fps
        interval = int(1 / fps * 1000)
        self.ani = animation.FuncAnimation(
                self._fig,
                func,
                frames=self._data,
                interval=interval,
                blit=False,
                repeat=False,
                cache_frame_data=False,
        )
        plt.show()

    def write(self, path: Path | str | None = None, fps: int = None):
        """Writes this Writer's frame and coordinates to a new mp4 video file.

        Args:
            path:
                A path to the new video file. If None, this Writer's path will
                be used with the str 'tracking' added to the name.
            fps:
                The frames per second of the new file. If None, the fps will
                match the reader's sample rate.
        """

        if path is None:
            name = reader.path.stem + '_tracking'
            path = reader.path.with_name(name).with_suffix('.mp4')

        print(f'Writing video to: {path}')

        func = partial(self._update, verbose=True)
        fps = self.reader.sample_rate if not fps else fps
        interval = int(1 / fps * 1000)
        self.ani = animation.FuncAnimation(
                self._fig,
                func,
                frames=self._data,
                interval=interval,
                blit=False,
                repeat=False,
                cache_frame_data=False,
        )
        self.ani.save(path)


if __name__ == '__main__':

    import pickle
    from mousefinder.readers import WebmReader

    video_base = '/media/matt/compute/PAC_Data/videos/'
    coord_base = '/media/matt/compute/PAC_Data/videos/'
    name = '5879_Left_group B-S_no rest_video.webm'
    #name = '5895_Right_group B-S_video.webm'
    video_path = video_base + name
    coords_path = coord_base + '5879_Left_group B-S_no rest_video_coordinates.pkl'
    #coords_path = coord_base + '5895_Right_group B-S_video_coordinates.pkl'

    reader = WebmReader(video_path)
    with open(coords_path, 'rb') as infile:
        data_dict = pickle.load(infile)
        coords = data_dict['coordinates']

    writer = MPLWriter(reader, coords, show_frame=False, label=True,
            imageprops={'cmap': 'viridis'}, marker='o', color='red')
    writer.preview()



