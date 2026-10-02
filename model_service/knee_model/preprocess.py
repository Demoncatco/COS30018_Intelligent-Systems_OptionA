# This file turns one loaded series into a block that is the same standard (same shape and brightness scale).
# A series comes out of dicom_io.load_series() with any number of slices, any pixel size and any brightness range.
# This file takes those varying slices, pixel size and brightness range and does the below to make a consistent input for the model every time:
# 1. Normalise - clips brightness to the 1st and 99th percentile of the whole series. That scale ranges from 0 - 1
# 2. Sample - Pick a certain amount of slices and spread evenly from the first slice to the last
# 3. Pad - If any slices are not square in shape, add black border to the short side. This is a better alternative to stretching the image (and therefore the knee) to make the shape square.
# 4. Resize - Make the image a size of 224x224, therefore shrink or enlarge image to get one consistent shape size.

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

# 20 Slices are kept per series.
# Most series are above 20 slices, however there are some series that are below 20 slices, in this case, some slices are repeated to make up the required 20 slices.
N_SLICES = 20

# Height and width of every output slice in pixels. This is the consistent standard that is required. If slices are found to be larger or smaller than 224x224 then they are scaled.
IMAGE_SIZE = 224

# Defines the percentile of brightness that will be used to normalise a consistent brightness scale that will be used for outputted slices.
LOW_PERCENTILE = 1.0
HIGH_PERCENTILE = 99.0


@dataclass
class PreprocessedSeries:
    # Everything preprocess_series() produced for one series.

    array: np.ndarray  # The finished block, shape (n_slices, size, size), float32, values in 0 - 1
    slice_indices: np.ndarray  # Which original slice each output slice came from
    n_original_slices: int  # How many slices the series had before sampling
    n_repeated: int  # How many output slices are repeats of another one. 0 unless the series was shorter than n_slices (20)
    brightness_low_value: float  # The brightness that became 0.0 (the 1st percentile)
    brightness_high_value: float  # The brightness that became 1.0 (the 99th percentile)
    warnings: list[str] = field(default_factory=list)  # anything odd that occurs in this run, so it can be logged rather than crashed


def sample_slice_indices(n_available: int, n_wanted: int = N_SLICES) -> np.ndarray:
    # Works out which slices to keep. This is done by spreading evenly from the first slice to the last slice
    # first slice is defined as slice 0 and last slice is defined as number of available slices - 1.

    # When the series has at least n_wanted slices, every position is different and some slices are skipped
    # When the series is shorter than n_wanted slices, some positions do repeat.

    # Ensures that there are at least one slice. A series with exactly one slice passes and that slice is repeated 20 times.
    if n_available < 1:
        raise ValueError("cannot sample slices from an empty series")

    # np.linspace gives n_wanted evenly spaced decimal numbers that range from the first slice to the last slice. These are then rounded to then find the positions as a whole number as for example position 1.78 cannot exist.
    spread = np.linspace(0, n_available - 1, n_wanted)
    rounded = spread.round()
    return rounded.astype(int) # .astype(int) is required as the numbers were still stored as decimals, and the output requires it as an int.


def normalise_volume(volume: np.ndarray, low_percentile: float = LOW_PERCENTILE, high_percentile: float = HIGH_PERCENTILE) -> tuple[np.ndarray, float, float]:
    # Normalising the volume is putting the brightness of the whole series onto a scale that ranges from 0 - 1.
    # This returns the normalised volume, and the two brightness values that become 0 and 1.
    # The percentiles are taken from the entire volume / series as taking it individually from slice to slice would make the brightest part of each slice = 1, when one slice's "1" may be much brighter than another slice's "1".
    # This normalisation is required because an MRI does not have a fixed brightness.
    percentiles = np.percentile(volume, [low_percentile, high_percentile])
    low = float(percentiles[0])
    high = float(percentiles[1])

    # If a series is just one brightness then high is equal to low, returning all zeros so that nothing is divided by zero later. This should never occur though...
    if high <= low:
        return np.zeros(volume.shape, dtype=np.float32), low, high

    # Any outliers then need to be clipped, these are the < 1 percentile and greater than > 99 percentile. Anything outside of the range is converted to either the darkest or the brightest value.
    clipped = np.clip(volume, low, high)

    scaled = (clipped - low) / (high - low)
    return scaled.astype(np.float32), low, high


def pad_to_square(volume: np.ndarray) -> np.ndarray:
    # If a slice is not square, it gets borders added to the short side to then make it square. This is done because a non-square slice when resized to 224x224 will cause the slice to stretch, affecting the knee for analysis.
    n, height, width = volume.shape
    if height == width:
        return volume

    # The new side is the longer of the two, so nothing is cut off.
    side = max(height, width)
    # How many black rows go above the image and how many black columns go to its left. The rest of the gap ends up below and to the right.
    # One of the two is always 0, because only the short side needs a border.
    top = (side - height) // 2
    left = (side - width) // 2

    # Starts with a block that is all black, then copies the original slices into the middle of it.
    padded = np.zeros((n, side, side), dtype=volume.dtype)
    padded[:, top : top + height, left : left + width] = volume
    return padded


def resize_volume(volume: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    # Resizes every slice to the size of 224x224. Once resized, then stacks them back together
    # Pillow's bilinear filter is used here for two main reasons:
    # Firstly, when shrinking, it averages over all original pixels that land in each new pixel, therefore fine detail is blended and not skipped.
    # Every new pixel is a weighted average, so a value can never go below the darkest or above the brightest original pixel, ensuring that it stays in normalised range of 0 - 1 for the brightness scale.
    resized = []
    for one_slice in volume:
        # Pillow needs each slice as its own tidy float32 array. A slice cut out of a bigger block is not always stored that way, therefore the below is done.
        one_slice = np.ascontiguousarray(one_slice, dtype=np.float32)
        image = Image.fromarray(one_slice)  # a float32 array becomes a Pillow image in mode "F"
        image = image.resize((size, size), resample=Image.Resampling.BILINEAR)
        resized.append(np.asarray(image, dtype=np.float32))
    return np.stack(resized)


def _check_volume(volume: np.ndarray) -> None:
    # Stops a volume that cannot become a valid image. Returns nothing: it either raises an error or lets the volume through.
    if volume.ndim != 3: # Checks to see if number of dimensions is not equal to 3 (slices, height, width). If it is not, error is thrown.
        raise ValueError(f"expected a 3D volume (slices, height, width), got shape {volume.shape}")
    # A NaN (Not a Number) or infinity would spread through the percentiles and turn the whole output into NaN without any error.
    # It cannot become a valid image, so this is stopped here instead of being passed on.
    if not np.isfinite(volume).all():
        raise ValueError("volume contains NaN or infinite values")


def _count_repeats(indices: np.ndarray) -> int:
    # Counts the amount of output slices that have repeated, which is only above zero for a series that is shorter than n_slices.
    return int(len(indices) - len(np.unique(indices)))


def preprocess_series(volume: np.ndarray, n_slices: int = N_SLICES, size: int = IMAGE_SIZE) -> PreprocessedSeries:
    # Runs all the functions outlined above in the correct order to achieve the below:
    # Take the volume provided by dicom_io.load_series, shape (slices, height, width), which are already in anatomical order, and return the fixed block of shape (n_slice, size, size) together, also providing a record of what has been done.

    _check_volume(volume) # Checks to make sure volume can become a valid image.

    # Creates a warning array, and also records the original number of slices prior to any changes being made.
    warnings = []
    n_original = volume.shape[0]

    # First part of the preprocess is to normalise, using the whole series.
    normalised, low, high = normalise_volume(volume)
    if high <= low:
        warnings.append(f"series has a single brightness value ({low}); output is all zeros")

    # Second part of the preprocess is to sample and figure out which slices to keep.
    indices = sample_slice_indices(n_original, n_slices)
    sampled = normalised[indices]
    n_repeated = _count_repeats(indices)
    if n_repeated > 0: # If repeated slices occur, this is noted.
        warnings.append(f"series has {n_original} slices, fewer than the {n_slices} wanted; {n_repeated} output slices are repeats")

    # Third part of the preprocess is to pad to square and make the slice square. If a series is already square, then this passes straight through.
    padded = pad_to_square(sampled)

    # Fourth part of the preprocess is to resize every slice to the specified size (224x224 px)
    resized = resize_volume(padded, size)
    # The resize cannot leave brightness range of 0 - 1 by design. The clip is this guard of breaching the range.
    array = np.clip(resized, 0.0, 1.0).astype(np.float32)

    return PreprocessedSeries(
        array=array,
        slice_indices=indices,
        n_original_slices=n_original,
        n_repeated=n_repeated,
        brightness_low_value=low,
        brightness_high_value=high,
        warnings=warnings,
    )
