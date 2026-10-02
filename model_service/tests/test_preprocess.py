# Tests for the preprocessing, checking that sampling, normalising, padding and resizing each do what they are meant to.
# No Kaggle data is needed. Every test builds a small volume of numbers by hand, where the right answer is known in advance.

import numpy as np
import pytest

from model_service.knee_model.preprocess import (
    IMAGE_SIZE,
    N_SLICES,
    normalise_volume,
    pad_to_square,
    preprocess_series,
    resize_volume,
    sample_slice_indices,
)


# Builds a fake volume where every slice is filled with its own slice number (slice 0 is all 0s, slice 1 is all 1s, and so on).
# After any processing, the brightness of a slice says which original slice it was, so order can be checked.
def _numbered_volume(n: int, height: int = 32, width: int = 32) -> np.ndarray:
    return np.stack([np.full((height, width), i, dtype=np.float32) for i in range(n)])


# A series with more slices than wanted. Must give exactly 20 positions, all different, going upwards, starting at the first slice and ending at the last.
def test_long_series_gives_unique_evenly_spread_indices() -> None:
    indices = sample_slice_indices(41, 20)
    assert len(indices) == 20
    assert len(set(indices)) == 20
    assert indices[0] == 0 and indices[-1] == 40
    assert np.all(np.diff(indices) > 0)


# A series with exactly as many slices as wanted. Every slice must be used once, in order, with none skipped.
def test_series_of_exactly_n_slices_uses_every_slice_once() -> None:
    assert sample_slice_indices(20, 20).tolist() == list(range(20))


# Checks every series length from 20 up to 300. None of them may produce a repeated slice, because repeats are only allowed for short series.
def test_no_repeats_for_any_series_at_least_n_long() -> None:
    for n in range(20, 301):
        indices = sample_slice_indices(n, 20)
        assert len(set(indices)) == 20, f"repeated slice for a series of {n}"


# A series shorter than wanted. Still 20 positions, every one of the 12 real slices is used, the order never goes backwards, and both ends are kept.
def test_short_series_repeats_slices_and_keeps_order() -> None:
    indices = sample_slice_indices(12, 20)
    assert len(indices) == 20
    assert set(indices) == set(range(12))
    assert np.all(np.diff(indices) >= 0)
    assert indices[0] == 0 and indices[-1] == 11


# The most extreme short series, a single slice. Every position must point at that one slice.
def test_single_slice_series_repeats_that_slice() -> None:
    assert sample_slice_indices(1, 20).tolist() == [0] * 20


# There is nothing to sample from zero slices, so it must raise an error.
def test_empty_series_raises() -> None:
    with pytest.raises(ValueError):
        sample_slice_indices(0, 20)


# Values 0 to 999. After normalising, everything must sit between 0 and 1 as float32, and the darkest and brightest must reach exactly 0 and 1.
def test_normalise_scales_into_zero_to_one() -> None:
    volume = np.arange(1000, dtype=np.float32).reshape(10, 10, 10)
    out, low, high = normalise_volume(volume)
    assert out.dtype == np.float32
    assert out.min() == 0.0 and out.max() == 1.0
    assert low < high


# A knee with brightness 0 to 100, plus one stray pixel at 100000. Without clipping, that one pixel would become 1.0 and the
# whole knee would be squashed below 0.001. With percentile clipping the knee must still use most of the 0 - 1 range.
def test_one_very_bright_pixel_does_not_squash_everything_else() -> None:
    rng = np.random.default_rng(0)
    volume = rng.uniform(0, 100, size=(10, 32, 32)).astype(np.float32)
    volume[0, 0, 0] = 100000.0
    out, _, high = normalise_volume(volume)
    assert high < 200
    assert np.median(out) > 0.4
    assert out[0, 0, 0] == 1.0


# One dim slice (values up to 50) and one bright slice (values up to 100) in the same volume. The dim slice must stay
# dimmer after normalising. If normalising were done slice by slice, both would end up with the same maximum of 1.0.
def test_normalise_is_per_volume_so_slices_keep_their_relative_brightness() -> None:
    ramp = np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32)
    volume = np.stack([ramp * 50, ramp * 100])
    out, _, _ = normalise_volume(volume)
    assert out[0].max() < 0.6
    assert out[1].max() == 1.0


# The same volume multiplied by 2.55507 (the Philips RescaleSlope seen in Step 1) must normalise to the same result.
# This is the claim from the worklog that a constant slope is undone by per-volume normalisation.
def test_normalise_cancels_a_constant_scale_factor() -> None:
    rng = np.random.default_rng(1)
    volume = rng.uniform(0, 889, size=(6, 16, 16)).astype(np.float32)
    plain, _, _ = normalise_volume(volume)
    scaled, _, _ = normalise_volume(volume * 2.55507)
    assert np.allclose(plain, scaled, atol=1e-5)


# A blank series where every pixel is 7. Must come back as zeros with no NaN (dividing by zero is the danger here), and with a warning.
def test_blank_series_becomes_zeros_with_a_warning_not_nan() -> None:
    result = preprocess_series(np.full((5, 16, 16), 7.0, dtype=np.float32))
    assert not np.isnan(result.array).any()
    assert result.array.max() == 0.0
    assert any("single brightness" in w for w in result.warnings)


# Shrinking (512 down to 224) and growing (160 up to 224) must both give exactly 224 x 224, with values still inside 0 - 1.
def test_resize_gives_the_exact_size_and_stays_in_range() -> None:
    rng = np.random.default_rng(2)
    for original in (512, 160):
        volume = rng.uniform(0, 1, size=(2, original, original)).astype(np.float32)
        out = resize_volume(volume, 224)
        assert out.shape == (2, 224, 224)
        assert out.dtype == np.float32
        assert out.min() >= 0.0 and out.max() <= 1.0


# A slice that is 20 high and 40 wide must become 40 x 40, with the original in the middle rows and black rows above and below.
def test_pad_to_square_adds_black_borders_to_the_short_side() -> None:
    volume = np.ones((3, 20, 40), dtype=np.float32)
    out = pad_to_square(volume)
    assert out.shape == (3, 40, 40)
    assert np.all(out[:, 10:30, :] == 1.0)
    assert np.all(out[:, :10, :] == 0.0) and np.all(out[:, 30:, :] == 0.0)


# A slice that is already square must be handed back untouched.
def test_pad_to_square_leaves_square_slices_alone() -> None:
    volume = np.ones((3, 16, 16), dtype=np.float32)
    assert pad_to_square(volume) is volume


# A wide slice holding a bright square exactly in the middle. After the whole pipeline the bright patch must still be as tall
# as it is wide. If the slice had been stretched to fit, the patch would come out taller than wide.
def test_non_square_slices_are_not_stretched() -> None:
    volume = np.zeros((20, 100, 200), dtype=np.float32)
    volume[:, 30:70, 80:120] = 1.0
    result = preprocess_series(volume, n_slices=20, size=224)
    bright = result.array[0] > 0.5
    rows_tall = int(bright.any(axis=1).sum())
    columns_wide = int(bright.any(axis=0).sum())
    assert abs(rows_tall - columns_wide) <= 1


# The main contract check. A series of 30 slices at 64 x 48 must come out as (20, 224, 224), float32, inside 0 - 1, with no warnings.
def test_preprocess_series_returns_the_contract_shape() -> None:
    rng = np.random.default_rng(3)
    volume = rng.uniform(0, 900, size=(30, 64, 48)).astype(np.float32)
    result = preprocess_series(volume)
    assert result.array.shape == (N_SLICES, IMAGE_SIZE, IMAGE_SIZE) == (20, 224, 224)
    assert result.array.dtype == np.float32
    assert result.array.min() >= 0.0 and result.array.max() <= 1.0
    assert result.n_original_slices == 30
    assert result.n_repeated == 0
    assert result.warnings == []


# Slices numbered 0 to 29 by brightness. After preprocessing, brightness must still rise from the first output slice to the last,
# and each output slice must be the original slice that slice_indices says it is. Proves sampling does not scramble the order.
def test_preprocess_series_keeps_slices_in_anatomical_order() -> None:
    result = preprocess_series(_numbered_volume(30))
    brightness = result.array[:, 100, 100]
    assert np.all(np.diff(brightness) > 0)
    assert result.slice_indices[0] == 0 and result.slice_indices[-1] == 29


# A series of 12 slices. The shape must still be (20, 224, 224), and the repeats must be counted and reported, not hidden.
def test_short_series_keeps_the_shape_and_is_flagged() -> None:
    result = preprocess_series(_numbered_volume(12))
    assert result.array.shape == (20, 224, 224)
    assert result.n_original_slices == 12
    assert result.n_repeated == 8
    assert any("fewer than the 20 wanted" in w for w in result.warnings)


# N is not fixed at 20 yet. Asking for 16 slices at 128 pixels must give exactly that, proving nothing in the code assumes 20 or 224.
def test_slice_count_and_size_can_be_changed() -> None:
    result = preprocess_series(_numbered_volume(30), n_slices=16, size=128)
    assert result.array.shape == (16, 128, 128)


# A volume containing a NaN cannot become a valid image. It must raise an error and not quietly return an array full of NaN.
def test_nan_in_the_volume_raises() -> None:
    volume = _numbered_volume(5)
    volume[2, 3, 3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        preprocess_series(volume)
