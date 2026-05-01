<h1 align="center">
    <img src="https://github.com/mscaudill/mousefinder/raw/master/imgs/logo.png"
    style="width:800px;height:auto;"/>
</h1>

**Machine vision models for mouse position tracking**

[**Key Features**](#key-features)
| [**Usage**](#usage)
| [**Dependencies**](#dependencies)
| [**Installation**](#installation)
| [**Roadmap**](#roadmap)
| [**Contributing**](#contributing)
| [**Acknowledgments**](acknowledgements)

--------------------

<p float="left">
  <img src="https://github.com/mscaudill/mousefinder/blob/master/imgs/5895_R_short_sample.gif?raw=true"
    style="width:400px;height:auto;"/>
  <img src="https://github.com/mscaudill/mousefinder/blob/master/imgs/6489_L_short_sample.gif?raw=true"
    style="width:400px;height:auto;"/>
</p>

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
    medium-sized (528 x 960) video frames in less than 400 microseconds thanks
    to [pyav's]( https://github.com/pyav-org/pyav) multithreading support.
    MouseFinder's detection algorithms rely soley on scipy's `ndimage` and numpy
    array operations backed by fast C++ code.  Additionally, MouseFinder's tracking
    models support multiprocessing. Taken together, a mouse in a video of 104000
    frames of size 528 x 960 can be tracked in 600 seconds using ten 3.4GHz CPUs.
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
MouseFinder has  5 datatypes; `VideoReader`, `Configuration`, `ROI`, `Model` and
`MPLWriter` that work together to support animal tracking. This usage guide
will explore the first four.

- [Reading Data](#Reading-Data)
- [Chamber Configurations](#Chamber-Configurations)
- [Defining an ROI](#Defining-an-ROI)
- [Tracking with Models](#Tracking-with-Models)

### Reading Data

MouseFinder's VideoReader is an iterable reader of the frames in the video that
yields indexed numpy arrays and stores important metadata like the frame rate.
Below we open a webm video file with this VideoReader.

Let's build a reader...

```python
from mousefinder.readers import VideoReader

# use your video path
path = '/media/matt/compute/PAC_Data/5895_Right.webm'
reader = VideoReader(path)

print(reader)
```

*Output*
```
VideoReader
--- Attributes ---
path: PosixPath('/m...S_video.webm')
stream_id: 0
formatter: <function fro...x7f45daab3420>
--- Properties ---
format: 'yuv420p'
key_spacing: 128
sample_rate: 29.41176470588235
shape: (106683, 528, 960)
--- Methods ---
creation_time
keyseek
Type help(WebmReader) for full documentation
```

```python
# video readers are iterables yielding an index and numpy array of data
for idx, frame in reader:
    print(idx, frame.shape)
    # stop after 5 frames
    if idx > 4:
        break
```

*Output*
```
0, (528, 960)
1, (528, 960)
2, (528, 960)
3, (528, 960)
4, (528, 960)
```

```python
# you can also seek to the prior keyframe(s)
# lets get the keyframes closest to frames 128 and 300 
keyframes = reader.keyseek([128, 300])
print(keyframes)
```

*Output*
```
[(128,
  array([[ 23,  23,  24, ..., 100,  97,  94],
         [ 25,  24,  21, ..., 101,  99,  96],
         [ 31,  29,  25, ..., 101, 101,  99],
         ...,
         [ 95,  96,  97, ...,  62,  52,  37],
         [ 95,  96,  97, ...,  66,  56,  42],
         [ 95,  96,  97, ...,  68,  58,  43]], shape=(528, 960), dtype=uint8)),
 (256,
  array([[ 27,  23,  19, ..., 100, 100, 100],
         [ 30,  27,  22, ..., 100, 100, 100],
         [ 35,  31,  27, ..., 100, 100, 100],
         ...,
         [ 95,  95,  95, ...,  63,  53,  40],
         [ 95,  95,  95, ...,  63,  53,  40],
         [ 95,  95,  95, ...,  63,  53,  40]], shape=(528, 960), dtype=uint8))]
```

Notice the 128th frame was a keyframe but the 300th one was not so keyseek
fetched the 256th frame (i.e. the prior keyframe).

### Chamber Configurations

Configurations are just dataclasses that keep track of metadata about the
recording chamber(s). Most importantly, the contain the dimensions of the
chambers which allows MouseFinder to compute pixel to cm scales. Here is the
Pinnacle Circular Chamber.

```python
from mousefinder.configurations import PCGC

help(PCGC)
```

*Output*
```
class PCGC(Configuration)
 |  PCGC(
 |      name: str = 'Pinnacle Circular Gravel',
 |      manufacturer: str = 'Pinnacle',
 |      material: str = 'plastic',
 |      bottom: str = 'gravel',
 |      shape: str = 'circle',
 |      height: float = 24,
 |      width: float = 24
 |  ) -> None
 |
 |  A representation of Pinnacle's circular gravel bottomed chamber.
 |
 |  Attributes:
 |      name:
 |          The descriptive name of this dataclass.
 |      manufacturer:
 |          The name of this chamber's manufacturer.
 |      material:
 |          The string name of the material used in this chamber.
 |      bottom:
 |          The string name of the material that lines the chamber bottom.
 |      shape:
 |          The shape of the arena within the chamber.
 |      height:
 |          The vertical dimension of the chamber.
 |      width:
 |          The horizontal dimension of the chamber.
```

### Defining an ROI
Regions of Interest help MouseFinder's models look in a specific area of
a video for a mouse. The ROI class supports building these regions of interest
automatically. Lets build an ROI instance.

```python
from mousefinder.rois import ROI

# To construct an ROI for the Pinnacle Circular Gravel chamber, we call the
`from_PCG` class method

help(ROI.from_PCG)
```

*Output*
```
from_PCG(
    reader: mousefinder.readers.VideoReader,
    config: mousefinder.configurations.Configuration,
    frames: collections.abc.Sequence[int] = (0, 1000),
    filt=<function sobel at 0x7f9b2016c860>,
    size: int = 10,
    thresholder: collections.abc.Callable[[numpy.ndarray[tuple[typing.Any, ...], numpy.dtype[~_ScalarT]]], float] = <function threshold_li at 0x7f9b2016f9c0>,
    **kwargs
) -> Self class method of mousefinder.rois.ROI
    Returns the region of interest for a Pinnacle circular gravel
    bottomed chamber.

    Args:
        path:
            Path to a video file with this chamber configuration.
        frames:
            A sequence of frames whose max intensity projection will be
            taken as the background image (i.e. no mouse present). These
            frames should be separated by enough time so that the mouse is
            unlikely to be in the same position in each frame. Defaults to
            the first and 1000-th frames.
        filt:
            An edge detection filter function from ndimage or skimage
            libraries. Defaults to a sobel filter.
        size:
            The size of scipy's maximum_filter kernel in pixels for removing
            the gravel bed texture and possible electrode wires. This size
            should be smaller than the mouse but larger than the variations
            in the gravel bed. The default is 10 pixels.
        thresholder:
            A callable expected to accept an image and return a float
            threshold that segments the chambers bottom. This defaults to
            skimage's threshold_li function.
        kwargs:
            Keyword args are passed to thresholder function.

    Returns:
        An ROI instance.
```

The helper tells us we need to supply a reader and a configuration and this
method will take care of constructing an ROI for us.

```python
from mousefinder.readers import VideoReader
from mousefinder.configurations import PCGC
from mousefinder.rois import ROI

path = '/media/matt/compute/PAC_Data/5895_Right_group B-S_video.webm'
reader = VideoReader(path)
roi = ROI.from_PCG(reader, config=PCGC())

# plot the roi using the first image of the video
idx, img = reader.keyseek(0)[0]
roi.plot(img)
```

<h1 align="center">
    <img src="https://github.com/mscaudill/mousefinder/blob/master/imgs/roi_5895_R.png?raw=true"
    style="width:600px;height:auto;"/>
</h1>

> Our roadmap aims to allow users to custom draw ROIs with a new constructor
> called `from_draw` in the next release of Mousefinder. 


### Tracking with Models
We are now ready to pull everything in this guide together and call our first
model. Lets build a Pinnacle Circular Gravel Chamber model with a top-down
camera view...

```python
from mousefinder.models import PCGTop

# to instantiate the model we need to give its constructor a reader, roi and
# configuration instance we created above

model = PCGTop(reader, roi, config=PCGC())
print(model)
```

*Output*
```
PCGTop
--- Attributes ---
reader: VideoReader(p...3e20>) -> None
roi: <mousefinder....x7f9b133f5940>
configuration: PCGC(name='Pi...=24, width=24)
sigma_: None
threshold_: None
--- Properties ---

--- Methods ---
detect
estimate
printable
save
Type help(PCGTop) for full documentation
```

The two primary methods of the model are the `estimate` and `detect` methods.
`estimate` estimates a threshold for distinguishing the mouse from the background
and `detect` performs the detection of the mouse on each frame. Estimate will need
to be called prior to detection

```python
help(model.estimate)
```

*Output*
```
estimate(
    sigma: int | None = None,
    thresholder: collections.abc.Callable[[numpy.ndarray[tuple[typing.Any, ...], numpy.dtype[~_ScalarT]]], float] = <function threshold_li at 0x7f9b2016f9c0>,
    **kwargs
) -> None method of mousefinder.models.PCGTop instance
    Estimates the float threshold that best distinguishes the mouse
    from the background from an illumniation normalized image.

    The threshold computed by this method is the threshold after adjusting
    for light intensity changes by gaussian blur correction.

    Args:
        sigma:
            The standard deviation of the gaussian for correcting the uneven
            illumination. If None, this value defaults to 1/20 the height of
            the image.
        thresholder:
            A callable thresholding function that accepts an image and
            returns a integer or float value. The default is the
            threshold_li function from the skimage library.
        kwargs:
            All keyword arguments are passed to the thresholder.

    Returns:
        None
```

This function computes a threshold on images that have been normalized to
account for uneven illumination. You can see these light variations in the sample gif
video at the top of this page. Lets call the estimate method using the default
sigma

```python
model.estimate()
print(model.threshold_)
```

*Output*
```
np.float(0.7696747)
```

Now that we have a threshold we are ready to call the detect method
```python
help(model.detect)
```

*Output*
```
detect(
    *,
    minsize: int = 10,
    path: pathlib._local.Path | str | None = None,
    ncores: int | None = None,
    chunksize: int = 100,
    verbose: bool = True,
    saving: bool = True
) -> numpy.ndarray[tuple[typing.Any, ...], numpy.dtype[~_ScalarT]] method of mousefinder.models.PCGTop instance
    Concurrently detects mouse coordinates from each frame of this
    Detector's data.

    Args:
        minsize:
            The minimum size in pixels along rows and columns that objects
            within the images must achieve to be included in the
            segmentation. Objects below minsize x minsize pixels are
            excluded from detection. The default is 10 pixels.
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
```

This is the model's action method. It will use the threshold and locate the
mouse on each frame using as many cores as you allow. Lets call it...

```python
coords = model.detect(ncores=8)
```

At this point, depending on how many frames are in the video, you should get
a cup of coffee or tea. For 100k frames this will take about 10 minutes. When
you come back you'll have your coordinates saved and returned to your current
namespace. If you want to plot the results check out the MPLWriter in the writer
module of MouseFinder.

## Dependencies

MouseFinder's [pyproject.toml](#https://github.com/mscaudill/mousefinder/blob/master/pyproject.toml)
contains all the dependencies for using or developing MouseFinder. These are the
minimum requirements.

| Dependencies  |
|---------------|
| Python >= 3.13|
| numpy         |
| scipy         |
| scikit-image  |
| matplotlib    |
| ffmpeg        |
| av            |
| psutil        |

## Installation

## Roadmap

We've listed a few items throughout that we would like to see done in MouseFinder.
Below is a summary of our roadmap to be completed by the end of 2026.

1. GPU Processing with [JAX](https://github.com/jax-ml/jax):
    - MouseFinder primarily relies on scipy's `ndimage` library that enjoys
      partial support with JAX. We plan on using JAX to dramatically reduce our
      compute time.
2. More Models for more configurations:
    - MouseFinder currently supports only top-down single camera views. We aim
      to support angled camera positions and multicameras in our models for 3D
      tracking support.
3. Custom ROIS:
    - We will build an alternative constructor allowing users to define ROIs of
      their choice. This will allow for multiple models to run tracking
      different mice simultaneously.
4. Testing:
    - MouseFinder was built using test-driven development but it lacks formal
      reproducible pytest at the moment. This is a high-priority item on this
      roadmap.

## Contributing

We're excited you want to contribute! Please check out our
[Contribution](
"https://github.com/mscaudill/mousefinder?tab=contributing-ov-file") guide.


## Acknowledgements

------

**This work was made possible by generous grants and gifts from:**

- The National Institutes of Health NINDS Grant 2R01NS100738-08A1
- Ting Tsung and Wei Fong Chao Foundation at the Jan and Dan Duncan
  Neurological Research Institute at Texas Children's

------
