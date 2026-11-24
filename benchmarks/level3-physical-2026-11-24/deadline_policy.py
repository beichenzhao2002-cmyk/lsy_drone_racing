"""Zero-order hold schedule for measured controller deadline overruns."""
from math import ceil

def missed_periods(compute_ms, period_ms=20.0):
    """Extra physical periods before a late command can be delivered."""
    if compute_ms < 0 or period_ms <= 0:
        raise ValueError("Nonnegative compute time and positive period required")
    return max(0, ceil(compute_ms / period_ms - 1e-12) - 1)

class CommandQueue:
    """Deliver due radio commands even while the control process is busy."""
    def __init__(self, initial):
        self.current = initial.copy()
        self.pending = {}
    def submit(self, tick, command, delayed):
        self.pending[tick + int(delayed)] = command.copy()
    def advance(self, tick):
        for due in sorted(list(self.pending)):
            if due <= tick:
                self.current = self.pending.pop(due)
        return self.current.copy()
