"""Talker–Worker dual-runtime shopping engine (Phase 1 skeletal)."""

from .bus import EventBus
from .session import SessionStore

__all__ = ["EventBus", "SessionStore"]
