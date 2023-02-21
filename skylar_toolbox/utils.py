# =============================================================================
# Load libraries
# =============================================================================

import datetime as de
import time

# =============================================================================
# time_method
# =============================================================================

def time_method(method):
    def wrap_method(*pargs, **kwargs):
        start_ft = time.perf_counter()
        result = method(*pargs, **kwargs)
        end_ft = time.perf_counter()
        print(f'{method.__qualname__} - {de.timedelta(seconds=end_ft - start_ft)}')
        return result
    return wrap_method
