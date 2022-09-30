import inspect
import audiomentations as A


# widget dicts should be joined with "\n"
TEMPLATE = """
class {class_name}(BaseHelper):
    def __init__(self, keygen: Generator) -> None:
        super().__init__(
            keygen=keygen,
            aug_class=A.{class_name},
            widgets_kwargs=[
{widget_dicts}
            ],
        )
""".strip(
    "\n"
)

WIDGET_BOOL = """
                {{
                    "_widget_enum": Widget.BOOL,  # FIXME just check
                    "label": "{param_name}",  # match the param name
                    "value": {default},  # match default value
                }},
""".strip(
    "\n"
)

WIDGET_CHOICE = """
                {{
                    "_widget_enum": Widget.CHOICE,  # FIXME change options and index by digging the docs
                    "label": "{param_name}",  # match the param name
                    "options": ["red", "green", "blue"],  # match allowed values
                    "index": 0,  # match default value
                }},
""".strip(
    "\n"
)

WIDGET_RANGE1 = """
                {{
                    "_widget_enum": Widget.RANGE1,  # FIXME add min_value, max_value, step, format
                    "label": "{param_name}",  # match the param name
                    "value": {default},  # match default values
                }},
""".strip(
    "\n"
)

WIDGET_RANGE2 = """
                {{
                    "_widget_enum": Widget.RANGE2,  # FIXME add min_value, max_value, step, format
                    "label": "{param_name}",  # match the param name
                    "value": {default},  # match default values (min and max tuple)
                }},
""".strip(
    "\n"
)

WIDGET_UNKNOWN = """
                {{
                    "_widget_enum": Widget.UNKNOWN,  # FIXME subclass the render method
                    "label": "{param_name}",
                }},
""".strip(
    "\n"
)


def get_classes(ignore=set()) -> list:
    classes = []
    for content in dir(A):
        content = getattr(A, content)
        if not inspect.isclass(content):
            continue
        if content.__name__ not in ignore:
            classes.append(content)
    return classes


def get_class_init_param(cls, ignore=set()) -> dict:
    init_params = {}
    result = inspect.signature(cls.__init__)
    for v in result.parameters.values():
        if v.name in ignore:
            continue
        init_params[v.name] = v.default
    return init_params


def main():

    # last 2 is for spectrogram, first 5 is control flow
    ignore_classes = {
        "Compose",
        "Lambda",
        "OneOf",
        "SomeOf",
        "SpecCompose",
        "SpecChannelShuffle",
        "SpecFrequencyMask",
        # "RoomSimulator",
    }
    classes = get_classes(ignore=ignore_classes)

    # self and p is useless for now
    ignore_params = {"self", "p"}
    for cls in classes:
        widget_dicts = []
        widget_orders = []

        # original init params (no self and no p)
        init_params = get_class_init_param(cls, ignore_params)

        # used for sorting the widget dicts
        init_params_rank = {k: v for v, k in enumerate(init_params.keys())}

        # separate min max from the group
        minmax_without_minmax = {}
        for name, default in list(init_params.items()):

            # if min exists, do not search for max, and vice versa
            shortname = name[4:]
            if shortname in minmax_without_minmax:
                continue

            # pop 2 from original, add 1 no new
            if name.startswith("min_"):
                max_name = "max_" + shortname
                if max_name in init_params:
                    max_val = init_params.pop(max_name)
                    init_params.pop(name)
                    minmax_without_minmax[shortname] = (default, max_val)
            elif name.startswith("max_"):
                min_name = "min_" + shortname
                # print(min_name, cls, init_params)
                if min_name in init_params:
                    min_val = init_params.pop(min_name)
                    init_params.pop(name)
                    minmax_without_minmax[shortname] = (default, min_val)

        # handle non minmax (bool, choice, range1)
        for name, default in init_params.items():
            # possible types: numeric (int, float), bool, None, no-default, string
            if isinstance(default, (int, float)):
                widget_dicts.append(
                    WIDGET_RANGE1.format(param_name=name, default=default)
                )
            elif isinstance(default, bool):
                widget_dicts.append(
                    WIDGET_BOOL.format(param_name=name, default=default)
                )
            elif isinstance(default, str):
                widget_dicts.append(WIDGET_CHOICE.format(param_name=name))
            elif default is None:
                widget_dicts.append(WIDGET_UNKNOWN.format(param_name=name))
            elif default is inspect._empty:
                widget_dicts.append(WIDGET_UNKNOWN.format(param_name=name))
            else:
                print(default, type(default))
                raise RuntimeError("Default type is not handled!")
            widget_orders.append(init_params_rank[name])

        # handle minmax
        for name, default in minmax_without_minmax.items():
            widget_dicts.append(WIDGET_RANGE2.format(param_name=name, default=default))
            widget_orders.append(
                min(init_params_rank[f"min_{name}"], init_params_rank[f"max_{name}"])
            )

        # sort widget dicts by using original value
        widget_dicts = [
            val
            for order, val in sorted(
                zip(widget_orders, widget_dicts), key=lambda x: x[0]
            )
        ]

        # make a complete class
        complete_class = TEMPLATE.format(
            class_name=cls.__name__, widget_dicts="\n".join(widget_dicts)
        )
        print(complete_class)
        print("\n")


if __name__ == "__main__":
    main()
