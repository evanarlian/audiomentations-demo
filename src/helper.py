from abc import ABC
from enum import Enum
from typing import Generator

import numpy as np
import audiomentations as A
import streamlit as st


class Widget(Enum):
    BOOL = 1  # checkbox
    CHOICE = 2  # selectbox (dropdown)
    RANGE1 = 3  # slider with one value
    RANGE2 = 4  # slider with left and right values


class BaseHelper(ABC):
    """
    Helper class to encapsulate all data required
    for a single audio transform. This absctract class
    must be subclassed to be useful. User must pass a
    key generator from the main script to preserve input.
    Key generator can be an incrementing counter e.g.
    itertools.count

    The instance must render the streamlit widget
    before the real transform instance available
    for use.
    """

    def __init__(self, aug_class, widgets_kwargs: list, keygen: Generator) -> None:
        super().__init__()
        self.aug_class = aug_class
        self.widgets_kwargs = widgets_kwargs
        self.keygen = keygen
        self.proba = 1.0  # proba is always 100% for demo purposes
        self.captured = {}
        self.aug_instance = None

    def docstring(self) -> str:
        return self.aug_class.__doc__

    def init_docstring(self) -> str:
        return self.aug_class.__init__.__doc__

    def render(self) -> None:
        """Show streamlit widgets and store the selected values."""
        for wk in self.widgets_kwargs:
            widget_enum = wk.pop("_widget_enum")
            aug_param = wk["label"]
            if widget_enum == Widget.BOOL:
                self.captured[aug_param] = st.sidebar.checkbox(
                    **wk, key=next(self.keygen)
                )
            elif widget_enum == Widget.CHOICE:
                self.captured[aug_param] = st.sidebar.selectbox(
                    **wk, key=next(self.keygen)
                )
            elif widget_enum == Widget.RANGE1:
                self.captured[aug_param] = st.sidebar.slider(
                    **wk, key=next(self.keygen)
                )
            elif widget_enum == Widget.RANGE2:
                min_val, max_val = st.sidebar.slider(**wk, key=next(self.keygen))
                self.captured[f"min_{aug_param}"] = min_val
                self.captured[f"max_{aug_param}"] = max_val
            else:
                raise NotImplementedError(f"{widget_enum} is not implemented!")
        self.aug_instance = self.aug_class(**self.captured, p=self.proba)


class AddBackgroundNoise(BaseHelper):
    def __init__(self, keygen: Generator) -> None:
        super().__init__(
            keygen=keygen,
            aug_class=A.AddBackgroundNoise,
            widgets_kwargs=[
                {
                    "_widget_enum": Widget.UNKNOWN,  # FIXME subclass the render method
                    "label": "sounds_path",
                },
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "snr_in_db",  # match the param name
                    "value": (3, 30),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.CHOICE,  # FIXME change options and index by digging the docs
                    "label": "noise_rms",  # match the param name
                    "options": ["red", "green", "blue"],  # match allowed values
                    "index": 0,  # match default value
                },
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "absolute_rms_in_db",  # match the param name
                    "value": (-45, -15),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.UNKNOWN,  # FIXME subclass the render method
                    "label": "noise_transform",
                },
                {
                    "_widget_enum": Widget.RANGE1,  # FIXME add min_value, max_value, step
                    "label": "lru_cache_size",  # match the param name
                    "value": 2,  # match default values
                },
            ],
        )


class AddGaussianNoise(BaseHelper):
    def __init__(self, keygen: Generator) -> None:
        super().__init__(
            keygen=keygen,
            aug_class=A.AddGaussianNoise,
            widgets_kwargs=[
                {
                    "_widget_enum": Widget.RANGE2,
                    "label": "amplitude",
                    "value": (0.001, 0.015),
                    "min_value": 0.001,
                    "max_value": 0.030,
                    "step": 0.001,
                    "format": "%.3f",
                },
            ],
        )


class AddGaussianSNR(BaseHelper):
    def __init__(self, keygen: Generator) -> None:
        super().__init__(
            keygen=keygen,
            aug_class=A.AddGaussianSNR,
            widgets_kwargs=[
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "snr_in_db",  # match the param name
                    "value": (5, 40.0),  # match default values (min and max tuple)
                },
            ],
        )


class AddShortNoises(BaseHelper):
    def __init__(self, keygen: Generator) -> None:
        super().__init__(
            keygen=keygen,
            aug_class=A.AddShortNoises,
            widgets_kwargs=[
                {
                    "_widget_enum": Widget.UNKNOWN,  # FIXME subclass the render method
                    "label": "sounds_path",
                },
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "snr_in_db",  # match the param name
                    "value": (0.0, 24.0),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "time_between_sounds",  # match the param name
                    "value": (4.0, 16.0),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.CHOICE,  # FIXME change options and index by digging the docs
                    "label": "noise_rms",  # match the param name
                    "options": ["red", "green", "blue"],  # match allowed values
                    "index": 0,  # match default value
                },
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "absolute_noise_rms_db",  # match the param name
                    "value": (-50.0, -20),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.RANGE1,  # FIXME add min_value, max_value, step
                    "label": "add_all_noises_with_same_level",  # match the param name
                    "value": False,  # match default values
                },
                {
                    "_widget_enum": Widget.RANGE1,  # FIXME add min_value, max_value, step
                    "label": "include_silence_in_noise_rms_estimation",  # match the param name
                    "value": True,  # match default values
                },
                {
                    "_widget_enum": Widget.RANGE1,  # FIXME add min_value, max_value, step
                    "label": "burst_probability",  # match the param name
                    "value": 0.22,  # match default values
                },
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "pause_factor_during_burst",  # match the param name
                    "value": (0.1, 1.1),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "fade_in_time",  # match the param name
                    "value": (0.005, 0.08),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "fade_out_time",  # match the param name
                    "value": (0.01, 0.1),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.RANGE1,  # FIXME add min_value, max_value, step
                    "label": "signal_gain_in_db_during_noise",  # match the param name
                    "value": 0.0,  # match default values
                },
                {
                    "_widget_enum": Widget.UNKNOWN,  # FIXME subclass the render method
                    "label": "noise_transform",
                },
                {
                    "_widget_enum": Widget.RANGE1,  # FIXME add min_value, max_value, step
                    "label": "lru_cache_size",  # match the param name
                    "value": 64,  # match default values
                },
            ],
        )


class AirAbsorption(BaseHelper):
    def __init__(self, keygen: Generator) -> None:
        super().__init__(
            keygen=keygen,
            aug_class=A.AirAbsorption,
            widgets_kwargs=[
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "temperature",  # match the param name
                    "value": (10.0, 20.0),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "humidity",  # match the param name
                    "value": (30.0, 90.0),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "distance",  # match the param name
                    "value": (10.0, 100.0),  # match default values (min and max tuple)
                },
            ],
        )


class ApplyImpulseResponse(BaseHelper):
    def __init__(self, keygen: Generator) -> None:
        super().__init__(
            keygen=keygen,
            aug_class=A.ApplyImpulseResponse,
            widgets_kwargs=[
                {
                    "_widget_enum": Widget.UNKNOWN,  # FIXME subclass the render method
                    "label": "ir_path",
                },
                {
                    "_widget_enum": Widget.RANGE1,  # FIXME add min_value, max_value, step
                    "label": "lru_cache_size",  # match the param name
                    "value": 128,  # match default values
                },
                {
                    "_widget_enum": Widget.UNKNOWN,  # FIXME subclass the render method
                    "label": "leave_length_unchanged",
                },
            ],
        )


class BandPassFilter(BaseHelper):
    def __init__(self, keygen: Generator) -> None:
        super().__init__(
            keygen=keygen,
            aug_class=A.BandPassFilter,
            widgets_kwargs=[
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "center_freq",  # match the param name
                    "value": (200.0, 4000.0),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "bandwidth_fraction",  # match the param name
                    "value": (0.5, 1.99),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "rolloff",  # match the param name
                    "value": (12, 24),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.RANGE1,  # FIXME add min_value, max_value, step
                    "label": "zero_phase",  # match the param name
                    "value": False,  # match default values
                },
            ],
        )


class BandStopFilter(BaseHelper):
    def __init__(self, keygen: Generator) -> None:
        super().__init__(
            keygen=keygen,
            aug_class=A.BandStopFilter,
            widgets_kwargs=[
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "center_freq",  # match the param name
                    "value": (200.0, 4000.0),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "bandwidth_fraction",  # match the param name
                    "value": (0.5, 1.99),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "rolloff",  # match the param name
                    "value": (12, 24),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.RANGE1,  # FIXME add min_value, max_value, step
                    "label": "zero_phase",  # match the param name
                    "value": False,  # match default values
                },
            ],
        )


class Clip(BaseHelper):
    def __init__(self, keygen: Generator) -> None:
        super().__init__(
            keygen=keygen,
            aug_class=A.Clip,
            widgets_kwargs=[
                {
                    "_widget_enum": Widget.RANGE1,  # FIXME add min_value, max_value, step
                    "label": "a_min",  # match the param name
                    "value": -1.0,  # match default values
                },
                {
                    "_widget_enum": Widget.RANGE1,  # FIXME add min_value, max_value, step
                    "label": "a_max",  # match the param name
                    "value": 1.0,  # match default values
                },
            ],
        )


class ClippingDistortion(BaseHelper):
    def __init__(self, keygen: Generator) -> None:
        super().__init__(
            keygen=keygen,
            aug_class=A.ClippingDistortion,
            widgets_kwargs=[
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "percentile_threshold",  # match the param name
                    "value": (0, 40),  # match default values (min and max tuple)
                },
            ],
        )


class FrequencyMask(BaseHelper):
    def __init__(self, keygen: Generator) -> None:
        super().__init__(
            keygen=keygen,
            aug_class=A.FrequencyMask,
            widgets_kwargs=[
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "frequency_band",  # match the param name
                    "value": (0.0, 0.5),  # match default values (min and max tuple)
                },
            ],
        )


class Gain(BaseHelper):
    def __init__(self, keygen: Generator) -> None:
        super().__init__(
            keygen=keygen,
            aug_class=A.Gain,
            widgets_kwargs=[
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "gain_in_db",  # match the param name
                    "value": (-12, 12),  # match default values (min and max tuple)
                },
            ],
        )


class GainTransition(BaseHelper):
    def __init__(self, keygen: Generator) -> None:
        super().__init__(
            keygen=keygen,
            aug_class=A.GainTransition,
            widgets_kwargs=[
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "gain_in_db",  # match the param name
                    "value": (-24.0, 6.0),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "duration",  # match the param name
                    "value": (0.2, 6.0),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.CHOICE,  # FIXME change options and index by digging the docs
                    "label": "duration_unit",  # match the param name
                    "options": ["red", "green", "blue"],  # match allowed values
                    "index": 0,  # match default value
                },
            ],
        )


class HighPassFilter(BaseHelper):
    def __init__(self, keygen: Generator) -> None:
        super().__init__(
            keygen=keygen,
            aug_class=A.HighPassFilter,
            widgets_kwargs=[
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "cutoff_freq",  # match the param name
                    "value": (20, 2400),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "rolloff",  # match the param name
                    "value": (12, 24),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.RANGE1,  # FIXME add min_value, max_value, step
                    "label": "zero_phase",  # match the param name
                    "value": False,  # match default values
                },
            ],
        )


class HighShelfFilter(BaseHelper):
    def __init__(self, keygen: Generator) -> None:
        super().__init__(
            keygen=keygen,
            aug_class=A.HighShelfFilter,
            widgets_kwargs=[
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "center_freq",  # match the param name
                    "value": (300.0, 7500.0),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "gain_db",  # match the param name
                    "value": (-18.0, 18.0),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "q",  # match the param name
                    "value": (0.1, 0.999),  # match default values (min and max tuple)
                },
            ],
        )


class Limiter(BaseHelper):
    def __init__(self, keygen: Generator) -> None:
        super().__init__(
            keygen=keygen,
            aug_class=A.Limiter,
            widgets_kwargs=[
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "threshold_db",  # match the param name
                    "value": (-24, -2),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "attack",  # match the param name
                    "value": (0.0005, 0.025),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "release",  # match the param name
                    "value": (0.05, 0.7),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.CHOICE,  # FIXME change options and index by digging the docs
                    "label": "threshold_mode",  # match the param name
                    "options": ["red", "green", "blue"],  # match allowed values
                    "index": 0,  # match default value
                },
            ],
        )


class LoudnessNormalization(BaseHelper):
    def __init__(self, keygen: Generator) -> None:
        super().__init__(
            keygen=keygen,
            aug_class=A.LoudnessNormalization,
            widgets_kwargs=[
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "lufs_in_db",  # match the param name
                    "value": (-31, -13),  # match default values (min and max tuple)
                },
            ],
        )


class LowPassFilter(BaseHelper):
    def __init__(self, keygen: Generator) -> None:
        super().__init__(
            keygen=keygen,
            aug_class=A.LowPassFilter,
            widgets_kwargs=[
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "cutoff_freq",  # match the param name
                    "value": (150, 7500),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "rolloff",  # match the param name
                    "value": (12, 24),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.RANGE1,  # FIXME add min_value, max_value, step
                    "label": "zero_phase",  # match the param name
                    "value": False,  # match default values
                },
            ],
        )


class LowShelfFilter(BaseHelper):
    def __init__(self, keygen: Generator) -> None:
        super().__init__(
            keygen=keygen,
            aug_class=A.LowShelfFilter,
            widgets_kwargs=[
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "center_freq",  # match the param name
                    "value": (50.0, 4000.0),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "gain_db",  # match the param name
                    "value": (-18.0, 18.0),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "q",  # match the param name
                    "value": (0.1, 0.999),  # match default values (min and max tuple)
                },
            ],
        )


class Mp3Compression(BaseHelper):
    def __init__(self, keygen: Generator) -> None:
        super().__init__(
            keygen=keygen,
            aug_class=A.Mp3Compression,
            widgets_kwargs=[
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "bitrate",  # match the param name
                    "value": (8, 64),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.CHOICE,  # FIXME change options and index by digging the docs
                    "label": "backend",  # match the param name
                    "options": ["red", "green", "blue"],  # match allowed values
                    "index": 0,  # match default value
                },
            ],
        )


class Normalize(BaseHelper):
    def __init__(self, keygen: Generator) -> None:
        super().__init__(
            keygen=keygen,
            aug_class=A.Normalize,
            widgets_kwargs=[

            ],
        )


class Padding(BaseHelper):
    def __init__(self, keygen: Generator) -> None:
        super().__init__(
            keygen=keygen,
            aug_class=A.Padding,
            widgets_kwargs=[
                {
                    "_widget_enum": Widget.CHOICE,  # FIXME change options and index by digging the docs
                    "label": "mode",  # match the param name
                    "options": ["red", "green", "blue"],  # match allowed values
                    "index": 0,  # match default value
                },
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "fraction",  # match the param name
                    "value": (0.01, 0.7),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.CHOICE,  # FIXME change options and index by digging the docs
                    "label": "pad_section",  # match the param name
                    "options": ["red", "green", "blue"],  # match allowed values
                    "index": 0,  # match default value
                },
            ],
        )


class PeakingFilter(BaseHelper):
    def __init__(self, keygen: Generator) -> None:
        super().__init__(
            keygen=keygen,
            aug_class=A.PeakingFilter,
            widgets_kwargs=[
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "center_freq",  # match the param name
                    "value": (50.0, 7500.0),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "gain_db",  # match the param name
                    "value": (-24, 24),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "q",  # match the param name
                    "value": (0.5, 5.0),  # match default values (min and max tuple)
                },
            ],
        )


class PitchShift(BaseHelper):
    def __init__(self, keygen: Generator) -> None:
        super().__init__(
            keygen=keygen,
            aug_class=A.PitchShift,
            widgets_kwargs=[
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "semitones",  # match the param name
                    "value": (-4, 4),  # match default values (min and max tuple)
                },
            ],
        )


class PolarityInversion(BaseHelper):
    def __init__(self, keygen: Generator) -> None:
        super().__init__(
            keygen=keygen,
            aug_class=A.PolarityInversion,
            widgets_kwargs=[

            ],
        )


class Resample(BaseHelper):
    def __init__(self, keygen: Generator) -> None:
        super().__init__(
            keygen=keygen,
            aug_class=A.Resample,
            widgets_kwargs=[
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "sample_rate",  # match the param name
                    "value": (8000, 44100),  # match default values (min and max tuple)
                },
            ],
        )


class Reverse(BaseHelper):
    def __init__(self, keygen: Generator) -> None:
        super().__init__(
            keygen=keygen,
            aug_class=A.Reverse,
            widgets_kwargs=[

            ],
        )


class RoomSimulator(BaseHelper):
    def __init__(self, keygen: Generator) -> None:
        super().__init__(
            keygen=keygen,
            aug_class=A.RoomSimulator,
            widgets_kwargs=[
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "size_x",  # match the param name
                    "value": (3.6, 5.6),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "size_y",  # match the param name
                    "value": (3.6, 3.9),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "size_z",  # match the param name
                    "value": (2.4, 3.0),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "absorption_value",  # match the param name
                    "value": (0.075, 0.4),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "target_rt60",  # match the param name
                    "value": (0.15, 0.8),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "source_x",  # match the param name
                    "value": (0.1, 3.5),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "source_y",  # match the param name
                    "value": (0.1, 2.7),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "source_z",  # match the param name
                    "value": (1.0, 2.1),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "mic_distance",  # match the param name
                    "value": (0.15, 0.35),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "mic_azimuth",  # match the param name
                    "value": (-3.141592653589793, 3.141592653589793),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "mic_elevation",  # match the param name
                    "value": (-3.141592653589793, 3.141592653589793),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.CHOICE,  # FIXME change options and index by digging the docs
                    "label": "calculation_mode",  # match the param name
                    "options": ["red", "green", "blue"],  # match allowed values
                    "index": 0,  # match default value
                },
                {
                    "_widget_enum": Widget.RANGE1,  # FIXME add min_value, max_value, step
                    "label": "use_ray_tracing",  # match the param name
                    "value": True,  # match default values
                },
                {
                    "_widget_enum": Widget.UNKNOWN,  # FIXME subclass the render method
                    "label": "max_order",
                },
                {
                    "_widget_enum": Widget.UNKNOWN,  # FIXME subclass the render method
                    "label": "leave_length_unchanged",
                },
                {
                    "_widget_enum": Widget.RANGE1,  # FIXME add min_value, max_value, step
                    "label": "padding",  # match the param name
                    "value": 0.1,  # match default values
                },
                {
                    "_widget_enum": Widget.UNKNOWN,  # FIXME subclass the render method
                    "label": "ray_tracing_options",
                },
            ],
        )


class SevenBandParametricEQ(BaseHelper):
    def __init__(self, keygen: Generator) -> None:
        super().__init__(
            keygen=keygen,
            aug_class=A.SevenBandParametricEQ,
            widgets_kwargs=[
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "gain_db",  # match the param name
                    "value": (-12.0, 12.0),  # match default values (min and max tuple)
                },
            ],
        )


class Shift(BaseHelper):
    def __init__(self, keygen: Generator) -> None:
        super().__init__(
            keygen=keygen,
            aug_class=A.Shift,
            widgets_kwargs=[
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "fraction",  # match the param name
                    "value": (-0.5, 0.5),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.RANGE1,  # FIXME add min_value, max_value, step
                    "label": "rollover",  # match the param name
                    "value": True,  # match default values
                },
                {
                    "_widget_enum": Widget.RANGE1,  # FIXME add min_value, max_value, step
                    "label": "fade",  # match the param name
                    "value": False,  # match default values
                },
                {
                    "_widget_enum": Widget.RANGE1,  # FIXME add min_value, max_value, step
                    "label": "fade_duration",  # match the param name
                    "value": 0.01,  # match default values
                },
            ],
        )


class TanhDistortion(BaseHelper):
    def __init__(self, keygen: Generator) -> None:
        super().__init__(
            keygen=keygen,
            aug_class=A.TanhDistortion,
            widgets_kwargs=[
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "distortion",  # match the param name
                    "value": (0.01, 0.7),  # match default values (min and max tuple)
                },
            ],
        )


class TimeMask(BaseHelper):
    def __init__(self, keygen: Generator) -> None:
        super().__init__(
            keygen=keygen,
            aug_class=A.TimeMask,
            widgets_kwargs=[
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "band_part",  # match the param name
                    "value": (0.0, 0.5),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.RANGE1,  # FIXME add min_value, max_value, step
                    "label": "fade",  # match the param name
                    "value": False,  # match default values
                },
            ],
        )


class TimeStretch(BaseHelper):
    def __init__(self, keygen: Generator) -> None:
        super().__init__(
            keygen=keygen,
            aug_class=A.TimeStretch,
            widgets_kwargs=[
                {
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step
                    "label": "rate",  # match the param name
                    "value": (0.8, 1.25),  # match default values (min and max tuple)
                },
                {
                    "_widget_enum": Widget.RANGE1,  # FIXME add min_value, max_value, step
                    "label": "leave_length_unchanged",  # match the param name
                    "value": True,  # match default values
                },
            ],
        )


class Trim(BaseHelper):
    def __init__(self, keygen: Generator) -> None:
        super().__init__(
            keygen=keygen,
            aug_class=A.Trim,
            widgets_kwargs=[
                {
                    "_widget_enum": Widget.UNKNOWN,  # FIXME subclass the render method
                    "label": "top_db",
                },
            ],
        )


class DummyTransform:
    def __init__(
        self,
        is_alive: bool,
        color: str,
        fav_number: int,
        min_age: int,
        max_age: int,
        p: float,
    ) -> None:
        self.is_alive = is_alive
        self.color = color
        self.fav_number = fav_number
        self.min_age = min_age
        self.max_age = max_age
        self.p = p

    def __call__(self, samples: np.ndarray, sample_rate: int):
        # fake identity call
        # used for debugging purposes
        print("np audio arr:", samples.shape)
        print("sample rate:", sample_rate)
        print("is_alive:", self.is_alive)
        print("color:", self.color)
        print("fav_number:", self.fav_number)
        print("min_age:", self.min_age)
        print("max_age:", self.max_age)
        print()
        return samples


class Dummy(BaseHelper):
    def __init__(self, keygen: Generator) -> None:
        super().__init__(
            keygen=keygen,
            aug_class=DummyTransform,
            widgets_kwargs=[
                {
                    "_widget_enum": Widget.BOOL,
                    "label": "is_alive",
                    "value": True,
                },
                {
                    "_widget_enum": Widget.CHOICE,
                    "label": "color",
                    "options": ["red", "green", "blue"],
                    "index": 0,
                },
                {
                    "_widget_enum": Widget.RANGE1,
                    "label": "fav_number",
                    "min_value": 0,
                    "max_value": 100,
                    "value": 20,
                    "step": 1,
                },
                {
                    "_widget_enum": Widget.RANGE2,
                    "label": "age",
                    "min_value": 12,
                    "max_value": 76,
                    "value": (22, 50),
                    "step": 1,
                },
            ],
        )


# TODO transforms that require folder need to be frozen (disabled in the streamlit widget, but still shows text)
# TODO add ALL classes with the same name
helper_classes = {
    # "AddBackgroundNoise": AddBackgroundNoise,
    "AddGaussianNoise": AddGaussianNoise,
    "Dummy": Dummy,
}
