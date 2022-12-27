import json
import pandas as pd
import pytest
import os
from unittest import TestCase
from model import (
    PATH_TO_MODELS,
    model
)

from werkzeug.exceptions import NotFound


class TestGetModelsRest(TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ['TEST'] = '1'

    def test_get(self):
        gmr = PATH_TO_MODELS()

        with self.assertRaises(NotFound):
            gmr.get('wrong_type')

        self.assertDictEqual(gmr.get('classification'), {  'model_names': ['LogisticRegression', 'KNeighborsClassifier']})


class TestGetHyperParamsRest(TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ['TEST'] = '1'

    def test_get(self):
        ghpr = PATH_TO_MODELS()

        self.assertIn('hyper_params', ghpr.get('LogisticRegression'))

        with self.assertRaises(NotFound):
            ghpr.get('LogisticRegressionWrong')