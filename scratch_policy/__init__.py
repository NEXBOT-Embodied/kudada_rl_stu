"""Independent locomotion policy trained from random initialization.

No original ONNX model is needed to train or export this policy.
"""

from .policy import CHECKPOINT_FORMAT, ScratchActor, load_actor

__all__ = ["CHECKPOINT_FORMAT", "ScratchActor", "load_actor"]
