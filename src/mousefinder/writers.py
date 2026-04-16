""" """

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import animation

import numpy.typing as npt

from mousefinder.core import mixins
from mousefinder.readers import WebmReader

class MonoWriter(mixins.ReprMixin):
    """ """

    def __init__(
        self,
        reader: WebmReader,
        coordinates: npt.NDArray | None,
        **kwargs,
    ) -> None:
        """ """

        self.data = reader
        self.coordinates = coordinates
        self.marker = kwargs.pop('marker', '+')
        self.color = kwargs.pop('color', 'r')
        self.fig, self.ax = plt.subplots()
        self._init_draw()
        self._images = iter(self.data)

    def _init_draw(self) -> None:
        """ """

        _, img = next(iter(self.data))
        self._imgplot = self.ax.imshow(img, cmap='gray')
        if self.coordinates is not None:
            # plot the positions
            self._dotplot = self.ax.scatter(
                coords[0, 1],
                coords[0, 0],
                marker='+',
                color='r',
                label='Frame 0',
            )
        self.legend = self.ax.legend(loc='upper right')

    def _default_path(self):
        """ """

        # use video data's dir if no path given
        target = self.data.path.parent
        # if the target is a file get its parent
        target = target.parent if not target.is_dir() else target
        # add a stem to target and save
        target = target / Path(self.data.path.stem + '_tracking_video')

        return target.with_suffix('.mp4')

    def update(self, index):
        """ """

        if index % 500 == 0:
            print(f'{index} frames saved')

        idx, img = next(self._images)

        self._imgplot.set_array(img)
        self._dotplot.set_offsets(coords[index + 1, :][::-1])
        legend_texts = self.legend.get_texts()
        legend_texts[0].set_text(f'Frame: {index}')

        return self._imgplot, self._dotplot, legend_texts[0]

    def write(self, path: Path | str | None = None, fps=None):
        """ """

        fps = self.data.sample_rate if not fps else fps
        interval = int(1 / fps * 1000)
        frames = range(reader.shape[0] - 1)
        ani = animation.FuncAnimation(
                self.fig,
                self.update,
                frames=frames,
                interval=interval,
                blit=False,
        )

        fp = Path(path) if path else self._default_path()
        ani.save(fp)

if __name__ == '__main__':

    import pickle
    base = '/media/matt/Magnus/data/PAC_Data/videos/'
    name = '5876_Left_group B-S_no rest_video.webm'
    video_path = base + name
    #coords_path = base + '5879_Left_group B-S_no rest_video_coordinates.pkl'

    reader = WebmReader(video_path)

    """
    with open(coords_path, 'rb') as infile:
        data_dict = pickle.load(infile)
        coords = data_dict['coordinates']

    writer = MonoWriter(reader, coords)
    writer.write()
    """


