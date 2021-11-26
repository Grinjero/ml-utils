import inspect


def _validate_transformers(steps):
    for name, transformer in steps:
        has_transform = hasattr(transformer, 'transform')
        has_inverse = hasattr(transformer, 'inverse_transform')

        if (has_transform is False) and (has_inverse is False):
            raise ValueError(f'Transformer {name} has neither a \'transform\' nor \'inverse_transform\' function.'
                             f'For this to be a valid transformer either one of those must be present')


class TransformerPipeline:
    """
    Pipeline for transforming pandas DataFrames
    """

    def __init__(self, steps):
        _validate_transformers(steps)
        self.steps = steps

    def fit(self, y, X=None):
        for step_idx, name, transformer in self._iter_transformers():
            y = transformer.fit_transform(y, X)
            self.steps[step_idx] = (name, transformer)

    def update(self, y, X=None):
        for step_idx, name, transformer in self._iter_transformers():
            if hasattr(transformer, "update"):
                transformer.update(y, X)
                y = transformer.transform(y, X)
                self.steps[step_idx] = (name, transformer)

    def _iter_transformers(self, reverse=False):
        steps = self.steps
        if reverse:
            steps = reversed(steps)

        for idx, (name, transformer) in enumerate(steps):
            yield idx, name, transformer

    def transform(self, y, X=None):
        for step_idx, name, transformer in self._iter_transformers():
            if hasattr(transformer, 'transform') is False:
                continue

            y = transformer.transform(y, X)

        return y

    def inverse_transform(self, y, X=None, is_future=False):
        for step_idx, name, transformer in self._iter_transformers(reverse=True):
            if hasattr(transformer, 'inverse_transform') is False:
                continue
            inverse_transform_signature = inspect.signature(transformer.inverse_transform)
            if 'is_future' in inverse_transform_signature.parameters:
                y = transformer.inverse_transform(y, X, is_future=is_future)
            else:
                y = transformer.inverse_transform(y, X)

        return y