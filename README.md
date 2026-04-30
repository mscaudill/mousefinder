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
    To achieve fast video frame reading, MouseFinder has a dedicated `VideoReader`
    capable of decoding individual video frames in less than 400 microseconds. For
    an hour-long recording and frame size of 528 x 960, the VideoReader will read
    all the frames in 38 seconds on a 3.4 GHz CPU. We are indebted to [pyav's](
    https://github.com/pyav-org/pyav) multithreading support for this performance.
    MouseFinder's detection algorithms rely soley on scipy's `ndimage` and numpy
    array operations backed by fast C++ code. To further enhance speed,
    MouseFinder's tracking models support multiprocessing of video frames. For an
    hour-long recording and frame size of 528 x 960, 10 CPUs will track all target
    positions in 600 seconds.
    > [!NOTE]  
    > Our roadmap for MouseFinder includes further speed improvements using [JAX](
    https://github.com/jax-ml/jax) to target `ndimage` operations to GPUs.
 
- **Extensible**:



