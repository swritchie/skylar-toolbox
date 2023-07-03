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

def convert_types(
        df: pd.DataFrame, 
        unit_sr: str = 'MB'):
    '''
    Converts types to those that require less memory

    Parameters
    ----------
    df : pd.DataFrame
        Data frame.
    unit_sr : str, optional
        Unit of memory. The default is 'MB'.

    Returns
    -------
    df : pd.DataFrame
        Data frame.

    '''
    # Get memory before
    print('Memory: {}'.format(convert_bytes(
        input_bytes_it=df.memory_usage().sum().sum(), 
        unit_sr=unit_sr)[1]))
    
    # List types to convert
    dtypes_lt = ['float32', 'float64', 'int16', 'int32', 'int64']
    
    # Loop through columns converting types
    for column_sr in tqdm.tqdm(iterable=df.columns):
        # Get type
        dtype_sr = str(df[column_sr].dtypes)
        
        # If it is one of listed types...
        if dtype_sr in dtypes_lt:
            # Get min and max
            column_min = df[column_sr].min()
            column_max = df[column_sr].max()
        
            # Apply lossless compression
            if dtype_sr[:5] == 'float':
                if column_min > np.finfo(dtype='float16').min and column_max < np.finfo(dtype='float16').max:
                    df[column_sr] = df[column_sr].astype(dtype='float16')
                elif column_min > np.finfo(dtype='float32').min and column_max < np.finfo(dtype='float32').max:
                    df[column_sr] = df[column_sr].astype(dtype='float32')
            else:
                if column_min > np.iinfo(int_type='int8').min and column_max < np.iinfo(int_type='int8').max:
                    df[column_sr] = df[column_sr].astype(dtype='int8')
                elif column_min > np.iinfo(int_type='int16').min and column_max < np.iinfo(int_type='int16').max:
                    df[column_sr] = df[column_sr].astype(dtype='int16')
                elif column_min > np.iinfo(int_type='int32').min and column_max < np.iinfo(int_type='int32').max:
                    df[column_sr] = df[column_sr].astype(dtype='int32')
            
    # Get memory after
    print('Memory: {}'.format(convert_bytes(
        input_bytes_it=df.memory_usage().sum().sum(),
        unit_sr=unit_sr)[1]))
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
# return_memory
# =============================================================================
    
def return_memory(
        df: pd.DataFrame, 
        unit_sr: str):
    '''
    Returns memory for data frame

    Parameters
    ----------
    df : pd.DataFrame
        Data frame.
    unit_sr : str
        Unit.

    Returns
    -------
    memory_ft : float
        Memory float.
    memory_sr : str
        Memory string.

    '''
    memory_ft, memory_sr = convert_bytes(input_bytes_it=df.memory_usage().sum(), unit_sr=unit_sr)
    return memory_ft, memory_sr

# =============================================================================
# time_callable
# =============================================================================

def time_callable(callable_):
    def wrap_callable(*pargs, **kwargs):
        start_ft = time.perf_counter()
        result = callable_(*pargs, **kwargs)
        end_ft = time.perf_counter()
        print(f'{callable_.__qualname__} - {de.timedelta(seconds=end_ft - start_ft)}')
        return result
    return wrap_callable
