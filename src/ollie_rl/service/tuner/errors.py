"""Exception types raised by :class:`~ollie_rl.service.tuner.TunerService`."""

from typing import Optional


class TunerNotFoundError(Exception):
    pass


class InvalidRunCursorError(Exception):
    """Raised when a runs pagination cursor cannot be decoded."""

    pass


class RunNotFoundError(Exception):
    pass


class ChatCompletionNotFoundError(Exception):
    pass


class RunExpiredError(Exception):
    pass


class RewardAlreadySetError(Exception):
    pass


class EmptyRunError(Exception):
    """Raised when a reward is submitted for a run with no chat completions.

    A run that produced zero completions carries no training signal, so
    rewarding it is meaningless. We reject the reward outright; the run's
    lease simply expires and the datum is re-dispensed for a fresh attempt.
    """

    pass


class ContentFilterSampleError(Exception):
    def __init__(self, message: str, raw_content: Optional[str] = None):
        super().__init__(message)
        self.raw_content = raw_content


class LengthSampleError(Exception):
    def __init__(self, message: str, raw_content: Optional[str] = None):
        super().__init__(message)
        self.raw_content = raw_content
