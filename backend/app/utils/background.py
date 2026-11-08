from __future__ import annotations

import threading
from collections.abc import Callable


def spawn(fn: Callable[..., None], *args) -> None:
    thread = threading.Thread(target=fn, args=args, daemon=True)
    thread.start()
