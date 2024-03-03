from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from evaluator.data import Data
# from data import Data

class Model:
  def __init__(self, train_data, test_data, features, pred_column, models):
    self.train_data = train_data
    self.test_data = test_data
    self.features = features
    self.pred_column = pred_column
    self.models = models


  def train_model(self):
    X = self.train_data[self.features].copy()
    y = self.train_data[self.pred_column]
    X_train,X_valid, y_train , y_valid = train_test_split(X,y, train_size=0.8, test_size=0.2, random_state=0)
    X_train_data = Data(X_train)
    X_train_data.skim_columns()

    for model in self.models:
      model.fit(X_train, y_train)
      preds = model.predict(X_valid)
      score = mean_absolute_error(y_valid, preds)
      print(score)

