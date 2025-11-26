from .nodes import NODE_CLASS_MAPPINGS as LyraNodeClasses
from .nodes import NODE_DISPLAY_NAME_MAPPINGS as LyraNodeDisplayNames

NODE_CLASSES = {}
NODE_CLASSES.update(LyraNodeClasses)

NODE_DISPLAY_NAME = {}
NODE_DISPLAY_NAME.update(LyraNodeDisplayNames)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]