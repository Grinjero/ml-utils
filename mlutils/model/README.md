## Models
Contains components of each model and their implementations in corresponding folders.

TODO:
- Every model should have optional components:
    - `updatable`, `requires_fh_in_fit`
- Interface should follow `sklearn` eg. 
    - `fit` 
    - `predict`
    - `update(y, X, update_params=True/False)`- more or less extends the model
    
- largest choice is if it's better to adopt `sktime` or write something own
  