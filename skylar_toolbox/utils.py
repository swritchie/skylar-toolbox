# =============================================================================
# Load libraries
# =============================================================================

import datetime
import numpy as np
import pandas as pd

# =============================================================================
# DocFilter
# =============================================================================

class DocFilter:
    def __init__(self, item): self.doc_sr = item.__doc__
    def fit(self):
        self.doc_ss = pd.Series(data=self.doc_sr.splitlines())
        self.sections_df = (
            self.doc_ss
            .pipe(func=lambda x: x.loc[x.str.contains(pat='---').shift(periods=-1).ffill()])
            .pipe(func=lambda x: pd.Series(data=x.index, index=x))
            .rename(index=str.strip)
            .to_frame(name='starts')
            .assign(**{'stops': lambda x: x['starts'].shift(periods=-1)})
            .fillna(value={'stops': self.doc_ss.index.max().__add__(1)})
            .astype(dtype={'stops': int})
            .pipe(func=lambda x: pd.concat(objs=[pd.DataFrame(data=[[0, x.iloc[0, 0]]], index=['Intro'], columns=x.columns), x])))
        return self
    def print(self, section_sr='Intro'):
        if not section_sr: print(self.doc_sr)
        else: print(self.doc_ss.iloc[slice(*self.sections_df.loc[section_sr, :])].str.cat(sep='\n'))

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
# get_shape 
# =============================================================================

def get_shape(item): 
    if hasattr(item, 'shape'): return item.shape
    elif hasattr(item, '__len__'): return len(item)
    else: return np.nan

# =============================================================================
# get_type_and_shape 
# =============================================================================

def get_type_and_shape(item): return type(item), get_shape(item=item)

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
