from abc import ABC
from enum import Enum
from typing import Generator

import audiomentations as A
import streamlit as st


class Widget(Enum):
    BOOL = 1  # checkbox
    CHOICE = 2  # selectbox (dropdown)
    RANGE1 = 3  # slider with one values
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
        self.proba = 1.0  # proba is always 100% for testing purposes
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
                # TODO try this if false
                raise NotImplementedError(f"Widget {widget_enum} is not implemented!")
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

    def __call__(self, x):
        # fake identity call
        # used for debug instead
        print("is_alive", self.is_alive)
        print("color", self.color)
        print("fav_number", self.fav_number)
        print("min_age", self.min_age)
        print("max_age", self.max_age)
        return x


class Dummy(BaseHelper):
    def __init__(self, keygen: Generator) -> None:
        super().__init__(
            keygen=keygen,
            aug_class=DummyTransform,
            widgets_kwargs=[
                {
                    "_widget_enum": Widget.BOOL,
                    "label": "is_alive",  # match the param name
                    "value": True,  # match default value
                },
                {
                    "_widget_enum": Widget.CHOICE,
                    "label": "color",  # match the param name
                    "options": ["red", "green", "blue"],  # match allowed values
                    "index": 0,  # match default value
                },
                {
                    "_widget_enum": Widget.RANGE1,
                    "label": "fav_number",  # match the param name
                    "min_value": 0,  # experiment
                    "max_value": 100,  # experiment
                    "value": 20,  # match default values
                    "step": 1,  # match data type and experiment
                },
                {
                    "_widget_enum": Widget.RANGE2,
                    "label": "age",  # match the param name
                    "min_value": 12,  # experiment
                    "max_value": 76,  # experiment
                    "value": (22, 50),  # match default values (min and max)
                    "step": 1,  # match data type and experiment
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


# TODO add ALL classes with the same name
helper_classes = {
    "Dummy": Dummy,
    # "AddGaussianNoise": AddGaussianNoise,
}
