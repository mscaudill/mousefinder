<h1 align="center">
    <img src="https://github.com/mscaudill/mousefinder/raw/master/imgs/logo.png"
    style="width:800px;height:auto;"/>
</h1>

**Machine vision models for mouse position tracking**

[**Key Features**](#key-features)
| [**Usage**](#usage)
| [**Dependencies**](#dependencies)
| [**Installation**](#installation)
| [**Contributing**](#contributing)
| [**Acknowledgments**](acknowledgements)

--------------------

<h1 align="center">
    <img src="https://github.com/mscaudill/mousefinder/blob/master/imgs/5895_R_short_sample.gif?raw=true"
    style="width:600px;height:auto;"/>
</h1>

## Key Features

- **No Training Required**:  
    The machine-vision algorithms in MouseFinder are robust against lighting changes
    and distractors such as EEG wires and connectors. Deep-learning based pose
    estimators like [deeplabcut](
    https://deeplabcut.github.io/DeepLabCut/README.html) require additional training
    to distiguish these objects from the tracking target.  Importantly, mousefinder
    is just center-of-mass tracking for a single target. If you need pose estimation
    or multiple target tracking, deep-learning models are still the best way to go.

- **Fast and Scalable**:  
    MouseFinder has a dedicated `VideoReader` capable of decoding individual
    medium-sized (528 x 960) video frames in less than 400 microseconds. We are
    indebted to [pyav's]( https://github.com/pyav-org/pyav) multithreading support
    for this performance.  MouseFinder's detection algorithms rely soley on
    scipy's `ndimage` and numpy array operations backed by fast C++ code.
    Additionally, MouseFinder's tracking models support multiprocessing. Taken
    together, a mouse in a video of 104000 frames of size 528 x 960 can be tracked
    in 600 seconds using 10 3.4GHz CPUs.
    > Our roadmap for MouseFinder includes further speed improvements using
    > [JAX](
    https://github.com/jax-ml/jax) to target `ndimage` operations to GPUs.
 
- **Extensible**:
    MouseFinder's models are simple callables that can be easily extended to
    address animal tracking under a variety of experimental conditions;
    different chamber geometries, different viewing angles, enrichment
    distractors etc.  
    > Our roadmap aims to address different chamber geometries and viewing
    > angles within the next two or three releases of MouseFinder.

## Usage
```python
from mousefinder.models import PCGTop

>>> help(PCGTop)
```

*Output*
```python
class PCGTop(mixins.ReprMixin, mixins.SavingMixin, mixins.PrintMixin):
    """A model for mouse center-of-mass detection for Pinnacle's circular
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

        self.threshold_: int | None = None
```


