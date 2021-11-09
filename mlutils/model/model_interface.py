"""
Module for tracking created models. This module is a silly way of ensuring
that every model with same hyperparameters returns the same ID for easier comparison
of models.
"""

import uuid

_tracked_models = {}


class Model:
    def __init__(self):
        Model._model_created(self)
        self._shared_id = None

    @staticmethod
    def _model_created(model):
        """
        :param model: newly created model
        """
        new_id = uuid.uuid4()
        _tracked_models[new_id] = model

    @staticmethod
    def _get_meta_id(model):
        """
        Get the id shared by all models with same hyperparameters or architectures

        :param model:
        :return: shared id
        """

        for old_id, old_model in _tracked_models.items():
            if (old_model.hyperparameters() == model.hyperparameters()) and (old_model.name == model.name):
                return old_id

        raise ValueError("An ID should always be available")

    @property
    def name(self):
        """
        :return: Name of the model
        """
        return self.__class__

    def hyperparameters(self):
        """
        :return: dict describing model hyperparameters, by default returns an empty dict
        """
        return {}

    @property
    def shared_id(self):
        """
        :return: unique identifier of the model that should be the same for models with same hyperparameters
        """
        if self._shared_id is None:
            self._shared_id = Model._get_meta_id(self)
        return self._shared_id

    def __str__(self):
        return self.name
