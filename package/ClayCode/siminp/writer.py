import shutil
from functools import wraps
from typing import Dict, List, Literal, Union

from ClayCode.core.classes import File, MDPFile
from ClayCode.core.consts import MDP, USER_MDP
from ClayCode.core.lib import set_mdp_freeze_clay, set_mdp_parameter


class MDPWriter:
    def __init__(
        self,
        template=None,
        run_type: Union[
            Literal["EM"], Literal["D_SPACE"], Literal["EQ"], Literal["P"]
        ] = None,
    ):
        template = File(template)
        if template.is_file() and not template.is_relative_to(MDP):
            template_file = USER_MDP / template
            shutil.copy(template, template_file)
        else:
            template_file = MDP / template
        self.template_file = MDPFile(template_file)
        if not template_file.is_file():
            if template.is_file():
                mdp_options = te

        with open(self.template_file, "r") as template_file:
            mdp_str = template_file.read()
        self._string = self.__string = mdp_str

    def freeze_groups(
        self,
        resnames: List[str],
        freeze_dims: List[Union[Literal["Y"], Literal["N"]]] = ["Y", "Y", "Y"],
    ):
        self.string = set_mdp_freeze_clay(
            uc_names=resnames, em_template=self.string, freeze_dims=freeze_dims
        )

    @property
    def string(self):
        return self._string

    @string.setter
    def string(self, string: str):
        self._string = string

    def modify_mdp_str(f):
        @wraps(f)
        def wrapper(self, **run_prms):
            mdp_str = self.string
            for parameter, value in run_prms:
                mdp_str = f(parameter=parameter, value=value, mdp_str=mdp_str)
            self.string = mdp_str

        return wrapper

    def reset(self):
        self._string = self.__string

    @modify_mdp_str
    def set(self, parameter: str, value: str, mdp_str):
        return set_mdp_parameter(
            parameter=parameter, value=value, mdp_str=mdp_str
        )

    @modify_mdp_str
    def add(self, parameter: str, value: str, mdp_str):
        return set_mdp_parameter(
            parameter=parameter, value=value, mdp_str=mdp_str
        )
