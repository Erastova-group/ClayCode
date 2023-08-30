import logging
import sys
import warnings

from Bio import BiopythonDeprecationWarning
from ClayCode.core.log import ClayCodeLogger

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

logging.setLoggerClass(ClayCodeLogger)
