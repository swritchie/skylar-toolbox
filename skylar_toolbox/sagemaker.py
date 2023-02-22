# =============================================================================
# Load libraries
# =============================================================================

import sagemaker
from sagemaker import sklearn as srsn

# =============================================================================
# CustomHyperparameterTuner
# =============================================================================

class CustomHyperparameterTuner:
    def __init__(
            self, 
            estimator: sagemaker.EstimatorBase, 
            hyperparameter_ranges_dt: dict, 
            objective_type_sr: str, 
            source_directory_sr: str, 
            init_dt: dict = dict()):
        '''
        Wraps sagemaker.HyperparameterTuner to provide defaults

        Parameters
        ----------
        estimator : sagemaker.EstimatorBase
            Estimator.
        hyperparameter_ranges_dt : dict
            Hyperparameter ranges.
        objective_type_sr : str
            Objective type.
        source_directory_sr : str
            Directory with source code (and optionally requirements file).
        init_dt : dict, optional
            Additional arguments passed to sagemaker.HyperparameterTuner(). The default is dict().

        Returns
        -------
        None.

        '''
        defaults_dt = {
            'estimator': estimator,
            'objective_metric_name': 'Score',
            'hyperparameter_ranges': hyperparameter_ranges_dt,
            'metric_definitions': [{'Name': 'Score', 'Regex': 'Score: ([-]?[0-9\\.]+)'}],
            'strategy': 'Bayesian',
            'objective_type': objective_type_sr,
            'max_jobs': 10,
            'max_parallel_jobs': 1,
            'base_tuning_job_name': source_directory_sr.replace('_', '-')[:30],
            'random_seed': 0,
            }
        defaults_dt.update(init_dt)
        self.init_dt = defaults_dt
        self.ht = sagemaker.HyperparameterTuner(**self.init_dt)
        
    def fit(
            self, 
            inputs_dt: dict, 
            fit_dt: dict = dict()):
        '''
        Fits tuner

        Parameters
        ----------
        inputs_dt : dict
            Inputs.
        fit_dt : dict, optional
            Additional arguments passed to sagemaker.HyperparameterTuner.fit(). The default is dict().

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        self.ht.fit(inputs=inputs_dt, **fit_dt)
        return self

# =============================================================================
# CustomSKLearn
# =============================================================================

class CustomSKLearn:
    def __init__(
            self, 
            source_directory_sr: str, 
            hyperparameters_dt: dict, 
            sagemaker_sn: sagemaker.Session, 
            dependencies_lt: list, 
            instance_type_sr: str, 
            use_spot_instances_bl: bool, 
            init_dt: dict = dict()):
        '''
        Wraps srsn.SKLearn to provide defaults

        Parameters
        ----------
        source_directory_sr : str
            Directory with source code (and optionally requirements file).
        hyperparameters_dt : dict
            Hyperpararmeters.
        sagemaker_sn : sagemaker.Session
            SageMaker session.
        dependencies_lt : list
            Paths to dependencies.
        instance_type_sr : str
            Instance type.
        use_spot_instances_bl : bool
            Flag for whether to use spot instances.
        init_dt : dict, optional
            Additional arguments passed to srsn.SKLearn(). The default is dict().

        Returns
        -------
        None.

        '''
        default_bucket_sr = sagemaker_sn.default_bucket()
        base_job_name_sr = source_directory_sr.replace('_', '-')[:30]
        local_mode_bl = instance_type_sr == 'local'
        defaults_dt = {
            # SKLearn
            'entry_point': 'script.py',
            'framework_version': '0.23-1',
            'source_dir': source_directory_sr,
            'hyperparameters': hyperparameters_dt,
            # Framework
            'code_location': f's3://{default_bucket_sr}/',
            'dependencies': dependencies_lt,
            # EstimatorBase
            'role': sagemaker.get_execution_role(sagemaker_session=sagemaker_sn),
            'instance_count': 1,
            'instance_type': instance_type_sr,            
            'output_path': f's3://{default_bucket_sr}/{base_job_name_sr}',
            'base_job_name': base_job_name_sr,
            'sagemaker_session': None if local_mode_bl else sagemaker_sn,
            'metric_definitions': [{'Name': 'Score', 'Regex': 'Score: ([-]?[0-9\\.]+)'}]}
        if not local_mode_bl:
            defaults_dt['disable_profiler'] = True
            defaults_dt['use_spot_instances'] = use_spot_instances_bl
        if defaults_dt['use_spot_instances']:
            defaults_dt['max_wait'] = 1 * 60 * 60 # Time in seconds: hours x minutes/hour x seconds/minute
        defaults_dt.update(init_dt)
        self.init_dt = defaults_dt
        self.skl = srsn.SKLearn(**self.init_dt)
    
    def fit(
            self, 
            inputs_dt: dict, 
            fit_dt: dict = dict()):
        '''
        Fits estimator

        Parameters
        ----------
        inputs_dt : dict
            Inputs.
        fit_dt : dict, optional
            Additional arguments passed to srsn.SKLearn.fit(). The default is dict().

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        self.skl.fit(inputs=inputs_dt, **fit_dt)
        return self