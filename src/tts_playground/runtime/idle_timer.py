import threading


class DebouncedIdle:
    """Call `on_idle` after `idle_secs` without use; resets on each `reset()`.
    Minimal overhead: single timer thread.
    """
    def __init__(self, on_idle, idle_secs: int = 180):
        self.on_idle = on_idle
        self.idle_secs = idle_secs
        self._lock = threading.Lock()
        self._t: threading.Timer | None = None

    def reset(self):
        with self._lock:
            if self._t is not None:
                self._t.cancel()
            self._t = threading.Timer(self.idle_secs, self._fire)
            self._t.daemon = True
            self._t.start()

    def _fire(self):
        try:
            self.on_idle()
        finally:
            with self._lock:
                self._t = None