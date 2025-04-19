class PacingScheduler:
    def __init__(self, method, alpha0, step, lambda_=None, r=None):
        """
        Initialize the pacing scheduler.

        Parameters:
        - method (str): Either 'linear' or 'exponential'.
        - alpha0 (float): Starting fraction of the data (e.g., 0.1 for 10%).
        - step (int): Number of epochs for which the fraction remains constant.
        - lambda_ (float): Increment factor for linear pacing.
                           (Used only if method == 'linear')
        - r (float): Multiplicative growth factor for exponential pacing.
                     (Used only if method == 'exponential')
        """
        self.method = method.lower()
        self.alpha0 = alpha0
        self.step = step
        self.lambda_ = lambda_
        self.r = r

        self.current_epoch = 0
        # Set the initial fraction based on alpha0
        self.current_fraction = alpha0

    def should_update(self):
        """
        Check if the dataset fraction should be updated.
        A new step is reached when the current epoch is a multiple of step.
        """
        return self.current_epoch % self.step == 0

    def _update_fraction(self):
        """
        Update the current fraction using the chosen pacing strategy.
        """
        step_count = self.current_epoch // self.step

        if self.method == 'linear':
            new_fraction = min(1, self.alpha0 + self.lambda_ * step_count)
        elif self.method == 'exponential':
            new_fraction = min(1, self.alpha0 * (self.r ** step_count))
        else:
            raise ValueError("Unknown pacing method. Use 'linear' or 'exponential'.")
        
        self.current_fraction = new_fraction

    def next_epoch(self):
        """
        Advance one epoch: update the fraction if needed and increment the epoch counter.
        """
        if self.should_update():
            self._update_fraction()
        fraction = self.current_fraction
        print(f"Epoch {self.current_epoch}: Using {fraction*100:.1f}% of data.")
        self.current_epoch += 1
        return fraction
