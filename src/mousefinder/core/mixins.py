"""A collection of Mixin classes endowing inheritors with common methods. Mixins
includes

ReprMixin:
    A mixin for string and echo representations of objects.
SaverMixin:
    A mixin for saving attributes or results from method calls of an object.
"""

import inspect
import pickle
import reprlib
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


class ReprMixin:
    """Mixin for pretty echo & str representations.

    This Mixin's representations exclude protected and private attributes.
    """

    def _attributes(self) -> list[str]:
        """Returns a list of 'name: value' strings for each attribute."""

        attrs = {k: v for k, v in vars(self).items() if not k.startswith('_')}
        return [f'{k}: {reprlib.repr(v)}' for k, v in attrs.items()]

    def _properties(self) -> list[str]:
        """Returns a list of 'name: value' strings for each property."""

        def isprop(p):
            return isinstance(p, property)

        props = dict(inspect.getmembers(type(self), isprop))
        props = {k: getattr(self, k) for k in props}
        return [f'{k}: {reprlib.repr(v)}' for k, v in props.items()]

    def _methods(self) -> list[str]:
        """Returns a list of method string names."""

        methods = inspect.getmembers(self, inspect.ismethod)
        return [name for name, _ in methods if not name.startswith('_')]

    def __repr__(self) -> str:
        """Returns the __init__'s signature as the echo representation."""

        # build a signature and get its args and class name
        signature = inspect.signature(self.__init__)  # type: ignore[misc]
        args = str(signature)
        cls_name = type(self).__name__
        return f'{cls_name}{args}'

    def __str__(self) -> str:
        """Returns this instances print representation."""

        # fetch instance name, attrs and methods strings
        cls_name = type(self).__name__
        attrs = self._attributes()
        props = self._properties()
        methods = self._methods()

        # make a help msg
        help_msg = f'Type help({cls_name}) for full documentation'
        # construct print string
        msg = [
            f'{cls_name}',
            '--- Attributes ---',
            '\n'.join(attrs),
            '--- Properties ---',
            '\n'.join(props),
            '--- Methods ---',
            '\n'.join(methods),
            help_msg,
        ]

        return '\n'.join(msg)


#pylint: disable-next=too-few-public-methods
class SavingMixin:
    """A mixin for saving metadata and coordinate data from a detector to
    a pickle.

    This mixin provides inheritors with a save method that enforces the
    following metadata to be saved in addition to the coordinates;
        - path:
            The video file path from which coords were detected.
        - timestamp:
            The acquistion start datetime of the video at path.
        - sample_rate:
            The sample rate of the video at path.
        - scale:
            The pixels per cm scale of the video at path.
        - model:
            The name of the detection model used on the video at path.
        - kwargs:
            Users may further supply additional save metadata via keyword
            arguments.

    Returns:
        None
    """

    def save(
        self,
        destination: str | Path,
        coordinates: npt.NDArray[np.float64],
        *,
        path: Path | str,
        timestamp: datetime,
        sample_rate: float,
        scale: float,
        model: str,
        **kwargs,
    ) -> None:
        """Saves the metadata and coordinate data of a detection model to
        a pickle file at destination."""

        # locals to get dict of all parameters
        data = locals()
        _, _, kwargs = [data.pop(el) for el in ('self', 'destination', 'kwargs')]
        data.update(kwargs)

        t0 = time.perf_counter()
        target = Path(destination).with_suffix('.pkl')
        with open(target, 'wb') as outfile:
            pickle.dump(data, outfile)

        tf = time.perf_counter()
        # shorten destination and print to stdout with timing info
        aRepr = reprlib.Repr(maxother=80)
        print(f'Saved to {aRepr.repr(target)} in {tf - t0} secs.')


#pylint: disable-next=too-few-public-methods
class PrintMixin:
    """A mixin for generic printing of messages to stdout."""

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
