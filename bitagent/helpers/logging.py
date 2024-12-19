import bittensor as bt
from contextlib import contextmanager
@contextmanager
def temporary_logging_state(new_state):
    """
    A context manager to temporarily set Bittensor's logging state.
    """
    # Cache the current logging state
    current_state = bt.logging.current_state
    bt.logging.info(f"OFFLINE: Caching current logging state: {current_state}")

    # Set the new logging state
    if new_state == 'Debug':
        bt.logging.set_debug()
    elif new_state == 'Trace':
        bt.logging.set_trace()
    elif new_state == 'Warning':
        bt.logging.set_warning()
    elif new_state == 'Info':
        bt.logging.set_info()
    else:
        bt.logging.set_default()

    try:
        yield
    finally:
        # Restore the original logging state
        if current_state.value == 'Debug':
            bt.logging.set_debug()
        elif current_state.value == 'Trace':
            bt.logging.set_trace()
        elif current_state.value == 'Warning':
            bt.logging.set_warning()
        elif current_state.value == 'Info':
            bt.logging.set_info()
        else:
            bt.logging.set_default()
        bt.logging.info(f"OFFLINE: Restored logging state to: {current_state}")
