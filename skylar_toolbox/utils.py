# =============================================================================
# Load libraries
# =============================================================================

import datetime

# =============================================================================
# print_sequence
# =============================================================================

def print_sequence(name_sr, sequence):
    len_it = len(sequence)
    len_len_it = len(str(len_it))
    sequence_sr = '\n'.join(f'{index_it:0{len_len_it}d}. {element}' for index_it, element in enumerate(iterable=sequence))
    print(f'{name_sr} ({len_it}):\n{sequence_sr}')
    
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
