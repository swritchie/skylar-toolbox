# =============================================================================
# Load libraries
# =============================================================================

import os
import pathlib

# =============================================================================
# get_processor_path
# =============================================================================

def get_processor_path(
        in_path_sr: str, 
        input_bl: bool):
    '''
    Gets input or output path within *Processor container 

    Parameters
    ----------
    in_path_sr : str
        Input path.
    input_bl : bool
        Flag for whether path is input or output

    Returns
    -------
    out_path_sr : str
        Output path.

    '''
    # Declare constants
    base_directory_sr = '/opt/ml/processing'
    input_directory_sr = f'{base_directory_sr}/input'
    output_directory_sr = f'{base_directory_sr}/output'
    
    # Convert to pathlib
    in_path_ph = pathlib.Path(in_path_sr)
    
    # Join
    out_path_sr = os.path.join(
        input_directory_sr if input_bl else output_directory_sr, 
        in_path_ph.stem, 
        in_path_ph.name)
    return out_path_sr
