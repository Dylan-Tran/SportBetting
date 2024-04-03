from typing import List
from Models.Model_interface import InputFeatures, ModelInterface


import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import clone  # Import clone here


class GBR_Tree_Model(ModelInterface):
    def __init__(self):
        self.FUT_pipline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                (
                    "regressor",
                    GradientBoostingRegressor(
                        learning_rate=0.1, max_depth=3, n_estimators=300
                    ),
                ),
            ]
        )

        self.model = None

    def train(self, data: List[InputFeatures]):
        X_train = [
            [e.OR_home, e.OR_away, e.DR_home, e.DR_away, e.rest_home - e.rest_away]
            for e in data
        ]
        Y_train = [e.pts_home - e.pts_away for e in data]

        self.model = clone(self.FUT_pipline)
        self.model.fit(X_train, Y_train)

    def predict(self, data: List[InputFeatures]):
        X_test = [
            [e.OR_home, e.OR_away, e.DR_home, e.DR_away, e.rest_home - e.rest_away]
            for e in data
        ]
        Y_test = [e.pts_home - e.pts_away for e in data]

        Y_pred_fold = self.model.predict(X_test)
        return Y_pred_fold

    def clear(self):
        self.model = None
