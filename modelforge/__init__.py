"""Infrastructure to implement and train NNPs"""

from pkgutil import extend_path

# Allow optional subpackages (for example, modelforge-ase) to extend modelforge.
__path__ = extend_path(__path__, __name__)

# Add imports here
from .modelforge import *


from ._version import __version__
