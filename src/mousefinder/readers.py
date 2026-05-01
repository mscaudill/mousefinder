"""A Video reader iterable supporting single video stream reading from any
container and codec type supported by pyav.
"""

import re
import typing
import warnings
from collections.abc import Iterator, Sequence
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import av
import numpy as np
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
        in this VideoReader."""

        with av.open(self.path) as container:
            stream = container.streams.video[self.stream_id]
            shape = stream.frames, stream.height, stream.width

        return shape

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

    # type checking disabled here as pyav allows None types that disrupt keyseek
    @typing.no_type_check
    def keyseek(
        self,
        index: int | Sequence[int],
        **kwargs,
    ) -> list[int, npt.NDArray]:
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
            for key_idx in key_indices:

                # seek to presentation time = sec / base
                pts = int(key_idx / rate / base + start)
                container.seek(pts, stream=vstream, **kwargs)

                frame = next(container.decode(video=self.stream_id))
                img = self.formatter(frame)
                index = round(frame.pts * base * rate)
                result.append((index, img))

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


class WebmReader(VideoReader):
    """An iterable video stream reader of webm video files.

    This reader provides simple iteration of the frames in the container's video
    stream and random access to keyframes.

    Attributes:
        path:
            The path to the video.
        stream_id:
            The integer id of the video stream to read.
        convert:
            The color format each frame will be converted to.
    """

    @property
    def shape(self):
        """Returns a shape tuple containing the estimated number of frames,
        height and width of data in this VideoReader.

        The frame number is estimated from the duration and sample rate since
        webm does not reliably store the number of frames to the metadata. This
        means the frame number can be off from the actual number of frames.
        """

        with av.open(self.path) as container:
            stream = container.streams.video[self.stream_id]
            frame = next(container.decode(video=self.stream_id))
            try:
                # try more accurate stream counting
                duration = stream.duration * stream.time_base
                count = int(duration * stream.average_rate)
            except TypeError:
                # if stream duration is missing fallback to container meta
                duration = container.duration
                count = int(duration / av.time_base * self.sample_rate)

        return int(count), frame.height, frame.width
