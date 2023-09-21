from ClayCode.core.classes import YAMLFile
from ClayCode.core.consts import CLAYFF_AT_CHARGES, CLAYFF_AT_TYPES, FF

CLAY_FF = FF / "ClayFF_Fe"
CLAY_ATYPES = YAMLFile(CLAYFF_AT_TYPES)
CLAY_ACHARGES = YAMLFile(CLAYFF_AT_CHARGES)
