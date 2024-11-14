import time
from functools import wraps
from typing import Any, Sequence, Optional
from math import floor

import numpy as np
import pyarrow as pa

import pyarrow_demo as pyad



def emit_result(action: str, time_in_ms: float):
    print(f"{time_in_ms:7.1f}ms to run {action}")


def time_it(func):
    import time
    @wraps(func)
    def wrapper(*args,**kwargs):
        start = time.time()
        result = func(*args,**kwargs)
        end = time.time()
        emit_result(func.__name__, 1000*(end-start))
        return result
    
    return wrapper



def index_left(list_input: Sequence[Any], value: Any, left_count: Optional[int] = 0):
    list_length = len(list_input)

    if list_length == 1:
        raise ValueError("`index_left` designed for intervals. Cannot index list of length 1.")

    if list_length == 2:
        return left_count

    split = floor((list_length - 1) / 2)
    if list_length == 3 and value == list_input[split]:
        return left_count

    if value <= list_input[split]:
        return index_left(list_input[: split + 1], value, left_count)
    else:
        return index_left(list_input[split:], value, left_count + split)


def native_python(search_space: Sequence[float], targets: list[int]):
    return [index_left(search_space, i) for i in targets]


def rust_vector(search_space: Sequence[float], targets: list[int]):
    return [pyad.index_left_from_rust(search_space, i) for i in targets]


@time_it
def native_python_on_list(search_space: np.ndarray, targets: list[int]):
    search_space_as_list = list(search_space)   
    return native_python(search_space_as_list, targets)


@time_it
def native_python_on_ndarray(search_space: np.ndarray, targets: list[int]):
    return native_python(search_space, targets)


@time_it
def rust_list_to_vec(search_space: np.ndarray, targets: list[int]):
    search_space_as_list = list(search_space)   
    return rust_vector(search_space_as_list, targets)


@time_it
def rust_ndarray_to_vec(search_space: np.ndarray, targets: list[int]):
    return rust_vector(search_space, targets)


@time_it
def rust_ndarray_to_array_dyn(search_space: np.ndarray, targets: list[int]):
    return [pyad.index_left_np(search_space, i) for i in targets]


@time_it
def rust_pyarrow(search_space: pa.lib.DoubleArray, targets: pa.lib.Int32Array):
    return pyad.index_left_arrow_vec(search_space, targets)


@time_it
def run_rust_pyarrow(search_space: np.ndarray, targets: list[int]):
    search_space_as_arrow = pa.array(search_space)
    targets_as_arrow = pa.array(targets)

    return rust_pyarrow(search_space_as_arrow, targets_as_arrow)


#index_left_arrow_vec


def _run():
    sample_size = 1000000
    repeats = 1000

    s=np.random.uniform(size=sample_size)
    ascending = np.cumsum(s)
    search = np.random.randint(0, sample_size/2, repeats)


    #native_python_on_list(ascending, search)
    native_python_on_ndarray(ascending, search)
    #rust_list_to_vec(ascending, search)
    #rust_ndarray_to_vec(ascending, search)
    #rust_ndarray_to_array_dyn(ascending, search)
    run_rust_pyarrow(ascending, search)


if __name__ == "__main__":
    _run()
