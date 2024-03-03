import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class Data:
  def __init__(self, path):
    self.data = pd.read_csv(path)

  def skim_data(self):
    print("columns\n", self.data.columns)
    print("head data\n", self.data.head())

  def skim_columns(self):
    for columnName in self.data.columns:
      column = self.data[columnName]
      column_dtype = column.dtype
      print("\n")
      print(f"------skimming column {columnName}-------")
      print("data type", column_dtype)
      if(column_dtype=="object"):
        print("Unique column data", column.unique(), len(column.unique()))
      if(column_dtype in ["float64", "float32"]):
        print("float min value", column.min())
        print("float max value", column.max())
      print(f"------skimming column {columnName}-------")


  def get_string_columns(self):
    string_columns = []
    for column in self.data.columns:
      if(self.data[column].dtype=="object"):
        string_columns.append(column)
    return string_columns
      

  def encode_string_columns(self, columns=[], encoder=LabelEncoder):
    le = encoder()
    for column in columns:
      self.data[column] = le.fit_transform(self.data[column])


