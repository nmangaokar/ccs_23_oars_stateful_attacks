from abc import abstractmethod
from utils.transforms import transform


class AttackError(Exception):
    pass


class BudgetExhaustionError(AttackError):
    pass


class AttackUnableToCompleteError(AttackError):
    pass


class Attack:
    @abstractmethod
    def __init__(self, model, model_config, attack_config):
        self._model = model
        self.model_config = model_config
        self.attack_config = attack_config

    @abstractmethod
    def attack_targeted(self, x, y):
        pass

    @abstractmethod
    def attack_untargeted(self, x):
        pass

    def get_cache_hits(self):
        return self._model.cache_hits

    def get_total_queries(self):
        return self._model.total

    def reset(self):
        self._model.reset()

    def _check_budget(self, budget):
        if self.get_total_queries() > budget:
            raise BudgetExhaustionError(
                f'Attack budget exhausted: {self.get_total_queries()} > {budget}')

    def model(self, x):
        if self.attack_config["adaptive"]["query_blinding_transform"] is not None:
            x = transform(x, **self.attack_config["adaptive"]["query_blinding_transform"])
        out = self._model(x)
        self._check_budget(self.attack_config['budget'])
        return out

    def end(self, reason):
        raise AttackUnableToCompleteError(reason)
