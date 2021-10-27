import sklearn


class Pipeline:
    def __init__(self, steps):
        self.steps = steps
        self._validate_steps()

    def _validate_steps(self):
        names, processing_steps = zip(*self.steps)
        self._validate_names(names)

    def _validate_names(self, names):
        if len(set(names)) != len(names):
            raise ValueError('Names provided are not unique: '
                             '{0!r}'.format(list(names)))

        invalid_names = [name for name in names if '__' in name]
        if invalid_names:
            raise ValueError('Estimator names must not contain __: got '
                             '{0!r}'.format(invalid_names))

    def __len__(self):
        """
        Returns the length of the Pipeline
        """
        return len(self.steps)

    @property
    def named_steps(self):
        """Map the steps to a dictionary"""
        return dict(self.steps)


    def _iter(self):
        for idx, (step_name, step_trans) in enumerate(self.steps):
            yield idx, step_name, step_trans


    def fit(self, y, X, **fit_kwargs):
        yt = y
        Xt = X

        for idx, step_name, step_trans in self._iter():
            clon
