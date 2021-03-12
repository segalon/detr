import torch

class ActiveLearner:
    def __init__(self, model, al_method):
        self.model = model
        self.al_method = al_method
        self.history = {}


