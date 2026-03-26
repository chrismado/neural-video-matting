from .composite import composite_foreground, composite_video, suppress_spill
from .inference import MattingInference, load_video_frames, save_alpha_sequence

__all__ = [
    "MattingInference",
    "load_video_frames",
    "save_alpha_sequence",
    "composite_foreground",
    "composite_video",
    "suppress_spill",
]
