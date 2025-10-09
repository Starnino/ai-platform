from sklearn.ensemble import RandomForestRegressor
from src.core.models import SklearnModel

class ExampleModel(SklearnModel):
    """
    Simulation model for generating synthetic data.
    Uses RandomForestRegressor to predict future values based on past data.
    """
    def __init__(self, n_estimators, random_state, **kwargs):
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.random_state = random_state
        
    def model(self):
        return RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state)