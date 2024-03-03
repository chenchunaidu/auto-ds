from evaluator import data


training_data = data.Data("./data/train.csv")
training_data.skim_data()
training_data.skim_columns()
training_data.encode_string_columns(columns=training_data.get_string_columns())
training_data.skim_data()