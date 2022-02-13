from collections import UserDict

class ModelDict(UserDict):
    """Helper class to pack multiple models into a compact dictionary and delegate methods to
        underlying classes.
    """

    def train(self, mode=True):
        for model in self.values():
            model.train(mode)

    def eval(self):
        self.train(False)

    def state_dict(self):
        return {key: value.state_dict() for key, value in self.items()}

    def load_state_dict(self, state_dict):
        for key, value in state_dict.items():
            self[key].load_state_dict(value)

    def is_distributed(self):
        for value in self.values():
            if value is not None and hasattr(value, 'module'):
                return True
        return False

    def get_submodules(self):
        if self.is_distributed():
            return [subm.module for subm in self.values()]

        return [subm for subm in self.values()]
