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
    def __init__(self, x): self.doc_sr = x.__doc__
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
# describe_structure
# =============================================================================

def describe_structure(x, indent_it=0, max_indent_it=2):
    prefix_sr = '  ' * indent_it
    if indent_it > max_indent_it:
        print(f'{prefix_sr}(Depth limit reached)')
        return
    if isinstance(x, dict):
        print(f'{prefix_sr}{type(x)} with {len(x)} keys')
        for key, value in x.items():
            print(f'{prefix_sr}- key: {key}')
            describe_structure(x=value, indent_it=indent_it + 1, max_indent_it=max_indent_it)
    elif isinstance(x, (list, set, tuple)):
        print(f'{prefix_sr}{type(x)} with {len(x)} elements')
        for i_it, element in enumerate(iterable=x):
            print(f'{prefix_sr}- element: {i_it}')
            describe_structure(x=element, indent_it=indent_it + 1, max_indent_it=max_indent_it)
    else: print(f'{prefix_sr}{get_type_and_shape(x=x)}')

# =============================================================================
# filter_dir
# =============================================================================

def filter_dir(x, under_flag_bl=False, module_flag_bl=False): return (
    pd.Series(data=dir(x))
    .to_frame(name='object')
    .assign(**{
        'under_flag': lambda y: y['object'].str.startswith(pat='_'),
        'type': lambda y: y['object'].apply(func=lambda z: type(getattr(x, z)).__name__),
        'module_flag': lambda y: y['type'].eq(other='module')})
    .query(expr=f'under_flag.eq(other={under_flag_bl})')
    .query(expr=f'module_flag.eq(other={module_flag_bl})'))

# =============================================================================
# get_shape 
# =============================================================================

def get_shape(x): 
    if hasattr(x, 'shape'): return x.shape
    elif hasattr(x, 'size'): return x.size
    elif hasattr(x, '__len__'): return len(x)
    else: return np.nan

# =============================================================================
# get_type_and_shape 
# =============================================================================

def get_type_and_shape(x): return type(x), get_shape(x=x)

# =============================================================================
# print_doc
# =============================================================================

def print_doc(x, sections_lt=['Intro']):
    df = DocFilter(x=x).fit()
    print(df.sections_df.index.tolist())
    for section_sr in sections_lt:
        try: df.print(section_sr=section_sr)
        except: pass
    return df

# =============================================================================
# print_sequence
# =============================================================================

def print_sequence(x, name_sr='Sequence'):
    len_it = len(x)
    len_len_it = len(str(len_it))
    sequence_sr = '\n'.join(map(lambda x: f'{x[0]:0{len_len_it}d}. {x[1]}', enumerate(iterable=x)))
    print(f'{name_sr} ({len_it}):\n{sequence_sr}')
    
# =============================================================================
# print_shapes
# =============================================================================

def print_shapes(x, types_bl=False, **kwargs): 
    print(*map(get_type_and_shape if types_bl else get_shape, x), **kwargs)

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
