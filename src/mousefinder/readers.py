"""A VideoReader supporting iterative reading of single video stream data from any
container and codec type supported by pyav.
"""

import re
import warnings
from collections.abc import Iterator, Sequence
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from zoneinfo import ZoneInfo

import av
import numpy.typing as npt

from mousefinder import formats
from mousefinder.core import mixins
from mousefinder.formats import Formatter


class VideoReader(mixins.ReprMixin):
    """An iterable video stream reader of pyav supported video containers.

    This reader provides simple iteration of the frames in the container's video
    stream and random access to keyframes.

    Attributes:
        path:
            The path to the video.
        stream_id:
            The integer id of the video stream to read.
        formatter:
            A function for formatting each pyav Frame instance into a grayscale
            image.
    """

    def __init__(
        self,
        path: str | Path,
        stream_id: int = 0,
        formatter: Formatter = formats.from_yuvj420p,
    ) -> None:
        """Initialize this VideoReader."""

        self.path = Path(path)
        self.stream_id = stream_id
        self.formatter = formatter

    @property
    def shape(self) -> tuple[int, int, int]:
        """Returns a shape tuple containing the number, height & width of frames
        in this VideoReader.

        Some video formats do not store the video's frame count. In this case,
        the number of frames is estimated from the stream or container duration.
        """

        with av.open(self.path) as container:
            container.seek(0)
            stream = container.streams.video[self.stream_id]
            frame = next(container.decode(video=self.stream_id))

            # if the stream has the frames ready to return
            if stream.frames > 0:
                return stream.frames, frame.height, frame.width

            base = stream.time_base
            rate = stream.average_rate
            assert isinstance(base, Fraction)
            assert isinstance(rate, Fraction)
            # estimate from stream if it has a duration
            if isinstance(stream.duration, int):
                count = int(stream.duration * base * rate)
                return count, frame.height, frame.width

            # fallback to less accurate container duration estimate
            cduration = container.duration
            assert isinstance(cduration, int)
            assert isinstance(av.time_base, int)
            count = int(cduration / av.time_base * rate)
            return count, frame.height, frame.width

    @property
    def format(self) -> str:
        """Returns the name of the color format of this VideoReader."""

        with av.open(self.path) as container:
            stream = container.streams.video[self.stream_id]
            fmt = stream.format.name

        return fmt

    @property
    def sample_rate(self) -> float:
        """Returns FFMPEGs best guess of the sample_rate."""

        with av.open(self.path) as container:
            stream = container.streams.video[self.stream_id]
            result = stream.guessed_rate

        if result is None:
            msg = 'FFMPEG could not determine the sample rate.'
            raise AttributeError(msg)

        return float(result)

    @property
    def key_spacing(self) -> int:
        """Returns the integer spacing between keyframes.

        This property assumes an even spacing of the keyframes which is
        generally True in streaming videos used in scientific applications.
        """

        sid = self.stream_id
        with av.open(self.path) as container:
            container.streams.video[sid].thread_type = 'AUTO'
            for idx, frame in enumerate(container.decode(video=sid)):
                if frame.key_frame and idx > 0:
                    spacing = idx
                    break

        return spacing

    def creation_time(
        self,
        fmt: str = '%m%d%Y%H%M%S',
        timezone: str = 'Etc/GMT+6',
    ) -> datetime | None:
        """Returns the start of the video acquisition datetime instance.

        Args:
            fmt:
                A python datetime format string. This is only used if the
                datetime is stored to the filename and ignored otherwise.

        Return:
            A datetime instance or None.
        """

        with av.open(self.path) as container:
            # assumes creation_time is an iso compliant UTC time
            start = container.metadata.get('creation_time', None)
            if start:
                utc_time = datetime.fromisoformat(start)
                zone = ZoneInfo(timezone) if timezone else None
                return utc_time.astimezone(zone)

            # Time on path is assumed to be local time
            match = re.search(r'(\d+)(\.)', str(self.path))
            start = match.group(1) if match else None
            if start:
                return datetime.strptime(start, fmt)

        return None

    def keyseek(
        self,
        index: int | Sequence[int],
        **kwargs,
    ) -> list[tuple[int, npt.NDArray]]:
        """Returns the closest keyframe preceding each indexed frame.

        This method assumes that the spacing of the keyframes is constant. This
        is generally True for streaming videos in scientific applications.

        Args:
            index:
                An integer frame index or sequence of frame indices. They
                key frames closest but preceding each index will be returned
            kwargs:
                All keyword arguments are passed to the container's seek method
                see pyav container docs for details.

        Returns:
            A list of tuples of frame indices and color formatted image arrays.
        """

        # list the indices and normalize negatives & convert to keyframes
        indices = [index] if isinstance(index, int) else index
        indices = [len(self) + idx if idx < 0 else idx for idx in indices]
        spacing = self.key_spacing
        key_indices = [idx // spacing * spacing for idx in indices]

        # warn if keyseek is seeking the same keyframe multiple times
        if len(set(key_indices)) < len(key_indices):
            msg = 'keyseek is returning the same keyframe multiple times'
            warnings.warn(msg)

        result = []
        with av.open(self.path) as container:

            vstream = container.streams.video[self.stream_id]
            # pyav allows time_base, rate and start to be None type
            base = vstream.time_base
            rate = vstream.average_rate
            start = vstream.start_time
            assert isinstance(base, Fraction)
            assert isinstance(rate, Fraction)
            assert isinstance(start, int)
            for key_idx in key_indices:

                # seek to presentation time = sec / base
                pts = int(key_idx / rate / base + start)
                container.seek(pts, stream=vstream, **kwargs)

                frame = next(container.decode(video=self.stream_id))
                img = self.formatter(frame)
                result.append((key_idx, img))

        return result

    def __iter__(self) -> Iterator[tuple[int, npt.NDArray]]:
        """Yields frames from this VideoReader."""

        # fetch attrs just once for performance
        sid, fmt = self.stream_id, self.formatter
        with av.open(self.path) as container:
            container.streams.video[sid].thread_type = 'AUTO'
            for idx, frame in enumerate(container.decode(video=sid)):
                yield idx, fmt(frame)

    def __len__(self):
        """Returns the number of frames in this VideoReader."""

        return self.shape[0]
