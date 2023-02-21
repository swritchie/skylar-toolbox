# =============================================================================
# Load libraries
# =============================================================================

# =============================================================================
# EarlyStoppingCallback
# =============================================================================

class EarlyStoppingCallback:
    def __init__(
            self, 
            wait_it: int):
        '''
        Stops study early after specified wait

        Parameters
        ----------
        wait_it : int
            Number of trials to wait before terminating study.

        Returns
        -------
        None.

        '''
        self.wait_it = wait_it
        
    def __call__(self, study, trial):
        # Get best trial
        best_trial_it = study.best_trial.number
        
        # Get last trial
        last_trial_it = study.trials[-1].number
        
        # Stop study if difference exceeds wait
        if last_trial_it - best_trial_it > self.wait_it:
            study.stop()