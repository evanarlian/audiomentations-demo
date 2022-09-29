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

    def _render(self, widget: Widget, widget_kwargs: dict) -> None:
        """Determine how to render different widgets."""

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


# class AddBackgroundNoise(BaseHelper):
#     pass


# class AddGaussianNoise(BaseHelper):
#     def __init__(self) -> None:
#         super().__init__(
#             aug_class=A.AddGaussianNoise,
#             input_list=[
#                 {
#                     "name": "amplitude",
#                     "widget": Widget.,
#                     "widget_kwargs": {
#                         "label": "amplitude",
#                         "min_value": 1,
#                         "max_value": 10,
#                         "value": (3, 5),
#                     }
#                 },
#             ]
#         )

# TODO transforms that require folder need to be frozen (disabled in the streamlit widget, but still shows text)
# TODO add ALL classes with the same name
helper_classes = {
    "Dummy": Dummy,
    # "AddGaussianNoise": AddGaussianNoise,
}
