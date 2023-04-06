# =============================================================================
# Load libraries
# =============================================================================

import datetime as de
import numpy as np
import pandas as pd
import time
import tqdm

# =============================================================================
# convert_bytes
# =============================================================================

def convert_bytes(
        input_bytes_it: int, 
        unit_sr: str):
    '''
    Converts bytes to specified unit

    Parameters
    ----------
    input_bytes_it : int
        Bytes.
    unit_sr : str
        Unit.

    Raises
    ------
    NotImplementedError
        Implemented values of unit_sr are {implemented_units_lt}.

    Returns
    -------
    output_ft : float
        Output float.
    output_sr : TYPE
        Output string.

    '''
    divisors_dt = {
        'KB': 1e3,
        'MB': 1e6,
        'GB': 1e9}
    implemented_units_lt = list(divisors_dt.keys())
    if unit_sr not in implemented_units_lt:
        raise NotImplementedError(f'Implemented values of unit_sr are {implemented_units_lt}')
    output_ft = input_bytes_it / divisors_dt[unit_sr]
    output_sr = f'{output_ft:0.1f} {unit_sr}'
    return output_ft, output_sr

# =============================================================================
# convert_types
# =============================================================================

def convert_types(df: pd.DataFrame):
    '''
    Converts types to those that are more appropriate (e.g., bool) or require less memory

    Parameters
    ----------
    df : pd.DataFrame
        Data frame.

    Returns
    -------
    df : pd.DataFrame
        Data frame.

    '''
    for column_sr in tqdm.tqdm(iterable=df.columns):
        if set(df[column_sr].unique()) == {0, 1}:
            df[column_sr] = df[column_sr].astype(dtype=bool)
        elif df[column_sr].dtype == 'object':
            df[column_sr] = df[column_sr].astype(dtype='category')
        elif df[column_sr].dtype == float:
            df[column_sr] = df[column_sr].astype(dtype=np.float32)
        elif df[column_sr].dtype == int:
            df[column_sr] = df[column_sr].astype(dtype=np.int32)
    return df

# =============================================================================
# print_sequence
# =============================================================================

def print_sequence(
        name_sr: str,
        sequence):
    '''
    Prints sequence as numbered list

    Parameters
    ----------
    name_sr : str
        Sequence name.
    sequence : TYPE
        Sequence.

    Returns
    -------
    None.

    '''
    len_it = len(sequence)
    len_len_it = len(str(len_it))
    sequence_sr = '\n'.join(
        f'{index_it:0{len_len_it}d}. {element}' 
        for index_it, element in enumerate(iterable=sequence))
    print(f'{name_sr} ({len_it}):\n{sequence_sr}')

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
