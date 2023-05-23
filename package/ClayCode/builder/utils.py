from __future__ import annotations

from functools import singledispatch, wraps
from typing import Any, Dict, List, Optional, Set, Tuple, Union


def make_manual_setup_choice(select_function: str):
    @wraps(select_function)
    def wrapper(instance_or_manual_setup, *args, **kwargs):
        manual_setup = get_manual_setup_option(instance_or_manual_setup)
        if manual_setup:
            result = select_function(*args, **kwargs)
        else:
            try:
                result = kwargs["result_map"][""]
            except TypeError:
                raise AttributeError("No default option specified")
            except KeyError:
                result = select_function(*args, **kwargs)
        return result

    return wrapper


@singledispatch
def get_manual_setup_option(instance_or_setup_option) -> str:
    assert instance_or_setup_option.__class__.__name__ in [
        "TargetClayComposition",
        "MatchClayComposition",
    ], f"Invalid usage of {__name__} with {instance_or_setup_option.__class__.__name__}!"
    return instance_or_setup_option.manual_setup


@get_manual_setup_option.register(str)
def _(instance_or_setup_option) -> str:
    return instance_or_setup_option


@make_manual_setup_choice
def select_input_option(
    query: str,
    options: Union[List[str], Tuple[str], Set[str]],
    result: Optional[str] = None,
    result_map: Optional[Dict[str, Any]] = None,
) -> str:
    while result not in options:
        result = input(query).lower()
    if result_map is not None:
        result = result_map[result]
    return result
