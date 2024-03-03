from evaluator import data, model
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

training_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")

training_data = data.Data(training_df)
training_data.transform(columns=training_data.get_string_columns())
test_data = data.Data(test_df)
test_data.transform(columns=test_data.get_string_columns())


features = ["HomePlanet", "CryoSleep", "Cabin", "Destination", "Age", "VIP", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck" ]


#Random forest
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]

model = model.Model(train_data=training_data.data, test_data=test_data.data, features=features, pred_column="Transported", models=models)
model.train_model()