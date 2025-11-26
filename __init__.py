from .nodes import NODE_CLASS_MAPPINGS as LyraNodeClasses
from .nodes import NODE_DISPLAY_NAME_MAPPINGS as LyraNodeDisplayNames

NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(LyraNodeClasses)

NODE_DISPLAY_NAMES = {}
NODE_DISPLAY_NAMES.update(LyraNodeDisplayNames)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAMES"]
