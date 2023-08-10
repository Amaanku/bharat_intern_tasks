import pandas as pd

url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
titanic_data = pd.read_csv(url)
titanic_data.to_csv("titanic_dataset.csv", index=False)
print("Titanic dataset downloaded")