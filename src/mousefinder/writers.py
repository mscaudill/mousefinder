""" """

import itertools
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import animation

import numpy.typing as npt

from mousefinder.core import mixins
from mousefinder.readers import VideoReader, WebmReader

class MP4Writer(mixins.ReprMixin):
    """A Video writer for writing the frames and detected coordinates to an mp4
    video file.

    Attributes:
        data:
            An iterable VideoReader instance.
        coordinates:
            A 2D array of coordinate positions in row, column format.
        marker:
            The shape of the marker to draw at each coordinate.
        color:
            The color of the marker to draw at each coordinate.
    """

    def __init__(
        self,
        reader: VideoReader,
        coordinates: npt.NDArray | None,
        **kwargs,
    ) -> None:
        """Initialize this writer with a reader, an array of coordinates and
        marker kwargs.

        Args:
            reader:
                An iterable VideoReader instance.
            coordinates:
                A frames x 2 array of coordinates in row, col format.
            fig:
                A matplotlib figure instance where the frame drawing
                & coordinate drawing will occur.
            ax:
                A matplotlib axis instance where the frame & coordinate drawing
                will occur.
            **kwargs:
                Optional marker properties include marker shape and color.

        Returns:
            None
        """

        self.data = reader
        self.coordinates = coordinates
        self.marker = kwargs.pop('marker', '+')
        self.color = kwargs.pop('color', 'r')
        self.fig, self.ax = plt.subplots()
        self._init_draw()
        self._images = iter(self.data)

    def _init_draw(self) -> None:
        """Make the initial draw of the first frame and first coordinate."""

        _, img = next(iter(self.data))
        self._imgplot = self.ax.imshow(img, cmap='gray')
        if self.coordinates is not None:
            # plot the positions
            self._dotplot = self.ax.scatter(
                self.coordinates[0, 1],
                self.coordinates[0, 0],
                marker='+',
                color='r',
                label='Frame 0',
            )
        self.legend = self.ax.legend(loc='upper right')

    def _default_dir(self):
        """Determines the default save dir where this Writer will write to."""

        # use video data's dir if no path given
        target = self.data.path.parent
        # if the target is a file get its parent
        target = target.parent if not target.is_dir() else target
        # add a stem to target and save
        target = target / Path(self.data.path.stem + '_tracking_video')

        return target.with_suffix('.mp4')

    def _update(self, enumerated, indices):
        """Frame and coordinate data callback function for matplotlib's function
        animation.

        Args:
            

        Returns:
            The updated image plot, coordinate plot and legend text.
        """

        index, img = enumerated
        written = index - indices.start
        if written % 100 == 0:
            msg = f'Writing: {written / len(indices) * 100:0.2f}% complete'
            print(msg, end='\r', flush=True)

        self._imgplot.set_array(img)
        self._dotplot.set_offsets(self.coordinates[index + 1, :][::-1])
        legend_texts = self.legend.get_texts()
        legend_texts[0].set_text(f'Frame: {index}')

        return self._imgplot, self._dotplot, legend_texts[0]

    def write(
        self,
        indices: range | None = None,
        path: Path | str | None = None,
        fps=None,
    ) -> None:
        """ """

        if not Path(path).is_dir():
            msg = 'path must be a directory'
            raise ValueError(msg)

        fps = self.data.sample_rate if not fps else fps
        interval = int(1 / fps * 1000)
        indices = range(reader.shape[0] - 1) if indices is None else indices
        # slice the reader's enumerated images
        enumerated = itertools.islice(
                self._images,
                indices.start,
                indices.stop,
                indices.step,
        )
        func = partial(self._update, indices=indices)
        ani = animation.FuncAnimation(
                self.fig,
                func,
                frames=enumerated,
                interval=interval,
                blit=False,
                repeat=False,
                cache_frame_data=False,
        )

        writedir = Path(self.data.path).parent if not path else Path(path)
        fp = writedir / Path(self.data.path.name + '_tracking').with_suffix('.mp4')
        ani.save(fp)
        

if __name__ == '__main__':

    import pickle
    video_base = '/media/matt/compute/PAC_Data/videos/'
    coord_base = '/media/matt/compute/PAC_Data/videos/coordinates/'
    name = '5879_Left_group B-S_no rest_video.webm'
    video_path = video_base + name
    coords_path = coord_base + '5879_Left_group B-S_no rest_video_coordinates.pkl'

    reader = WebmReader(video_path)
    with open(coords_path, 'rb') as infile:
        data_dict = pickle.load(infile)
        coords = data_dict['coordinates']

    writer = MP4Writer(reader, coords)
    writer.write(indices=range(12000),
    path='/media/matt/compute/PAC_Data/videos/tracking/',
    )



