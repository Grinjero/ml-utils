

class PandasTransformerPipeline:
    """
    Pipeline for transforming pandas DataFrames
    """
    def fit(self, y, X):
        for step_idx, name, transformer in self._iter_transformers():
            y = transformer.fit_transform(y, X)
            self.steps_[step_idx] = (name, transformer)

    def update(self, y, X):
        for step_idx, name, transformer in self._iter_transformers():
            if hasattr(transformer, "update"):
                transformer.update(X)
                self.steps_[step_idx] = (name, transformer)