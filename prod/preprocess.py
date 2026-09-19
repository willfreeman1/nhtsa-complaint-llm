"""Shared text/prompt helpers.

Training (`scripts/train_deberta.py`, `scripts/train_llm_lora.py`) and serving
must use the same formatting. This module re-exports the research functions
rather than copying them. A later CI test will import this module from both
sides.
"""
import sys

from prod.paths import SCRIPTS

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from teacher_prompt import (  # noqa: E402
    ALL_VALID_COMPONENTS,
    CLASS_DEFINITIONS,
    RARE_CATEGORIES,
    build_user_message,
    bucket_component,
)
from train_deberta import build_text  # noqa: E402

LABEL_DEFINITIONS = {**CLASS_DEFINITIONS, **RARE_CATEGORIES}

assert build_text("n", "Make", "Model", "2019") == build_user_message(
    "n", "Make", "Model", "2019"
), "encoder text format and teacher user message must stay identical"

__all__ = [
    "ALL_VALID_COMPONENTS",
    "CLASS_DEFINITIONS",
    "LABEL_DEFINITIONS",
    "RARE_CATEGORIES",
    "build_text",
    "build_user_message",
    "bucket_component",
]
