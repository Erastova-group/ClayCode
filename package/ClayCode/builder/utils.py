from __future__ import annotations

import logging
import re
import sys
from functools import singledispatch, wraps
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

logger = logging.getLogger(__name__)


def make_manual_setup_choice(select_function):
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


@get_manual_setup_option.register(bool)
def _(instance_or_setup_option) -> bool:
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


def get_checked_input(
    query: str,
    result_type: Type,
    check_value: Optional[Any] = None,
    result: Optional[Any] = None,
    re_flags=0,
    exit_val: str = "e",
    *result_init_args,
    **result_init_kwargs,
) -> Any:
    while not isinstance(result, result_type):
        result_input = input(f"{query} (or exit with {exit_val!r})\n")
        if result_input == exit_val:
            logger.info(f"Selected {exit_val!r}, exiting.")
            sys.exit(0)
        try:
            result_match = re.match(
                check_value, result_input, flags=re_flags
            ).group(0)
            if result_match != result_input:
                raise AttributeError
        except AttributeError:
            print(f"\tInvalid input: {result_input!r}")
        else:
            result = result_type(
                result_input, *result_init_args, **result_init_kwargs
            )
    return result
