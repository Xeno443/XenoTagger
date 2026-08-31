"""Vendored inference code for RedRocket's Hydra 3.5 e621 tag classifier
(https://huggingface.co/RedRocket/Hydra), used by core.hydra_classifier
as the in-process second-stage tagger. Copied verbatim from upstream's
own `hydra/hydra/` package (their internal implementation, not their
gui.pyw/inference.py/service.py entry points, which this app doesn't
use) - only image.py/model.py/classification.py/label.py plus their
siglip2/pool/head/glu/cufork internals are needed for a synchronous
single-image classify() call; upstream's utils/ (an async multi-worker
dataloader for their own CLI/service) is not vendored.

Renamed from upstream's own top-level package name `hydra` to `rr_hydra`
to avoid colliding with PyPI's unrelated (and far more commonly
installed) `hydra-core`, which also imports as `hydra`. All of this
package's internal imports are relative, so the rename is safe.
"""

from .classification import IMPLICATION_MODES, simple_slider
from .label import Label, Rewriter
from .model import Hydra, Extension, open_image, load_image, load_model
from .image import Image, patchify, stack, varlen, stack_to_varlen, unfold, unfold_varlen
