# =============================================================================
# Load libraries
# =============================================================================

import datetime
import pandas as pd

# =============================================================================
# filter_dir
# =============================================================================

def filter_dir(item, under_flag_bl=False, module_flag_bl=False): return (
    pd.Series(data=dir(item))
    .to_frame(name='object')
    .assign(**{
        'under_flag': lambda x: x['object'].str.startswith(pat='_'),
        'type': lambda x: x['object'].apply(func=lambda y: type(getattr(item, y)).__name__),
        'module_flag': lambda x: x['type'].eq(other='module')})
    .query(expr=f'under_flag.eq(other={under_flag_bl})')
    .query(expr=f'module_flag.eq(other={module_flag_bl})'))

# =============================================================================
# print_sequence
# =============================================================================

def print_sequence(sequence, name_sr='Sequence'):
    len_it = len(sequence)
    len_len_it = len(str(len_it))
    sequence_sr = '\n'.join(map(lambda x: f'{x[0]:0{len_len_it}d}. {x[1]}', enumerate(iterable=sequence)))
    print(f'{name_sr} ({len_it}):\n{sequence_sr}')
    
# =============================================================================
# print_shapes
# =============================================================================

def print_shapes(sequence, **print_kwargs): print(
    *map(lambda x: x.shape if hasattr(x, 'shape') else len(x), sequence), 
    **print_kwargs)

# =============================================================================
# time_callable
# =============================================================================

def time_callable(callable_):
    def wrap_callable(*pargs, **kwargs):
        now_dt = datetime.datetime.now()
        result = callable_(*pargs, **kwargs)
        print(f'{callable_.__qualname__} - {str(datetime.datetime.now() - now_dt)}')
        return result
    return wrap_callable

# =============================================================================
# write_readme
# =============================================================================

def write_readme(outputs_directory_ph):
    ph = outputs_directory_ph.joinpath('README.md')
    paths_lt = sorted(outputs_directory_ph.glob(pattern='*.png'))
    names_lt = list(map(lambda x: x.name, paths_lt))
    stems_lt = list(map(lambda x: x.stem, paths_lt))
    data_sr = '\n'.join(map(lambda x, y: '# %s\n![](%s)' % (x, y), stems_lt, names_lt))
    ph.write_text(data=data_sr)
