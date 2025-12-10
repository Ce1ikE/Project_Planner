from .global_const import *

def time_decorator(func):
    import time
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(
            f"{GlobalConst.TERMINAL_COLORS['GREEN']}"
            f"Function {func.__name__} took {end_time - start_time:.4f} seconds"
            f"{GlobalConst.TERMINAL_COLORS['RESET_COLOR']}"
        )
        return result
    return wrapper