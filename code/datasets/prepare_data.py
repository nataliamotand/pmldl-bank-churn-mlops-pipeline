import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/raw/churn.csv")

# remove identificadores sem valor preditivo
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

#print(df.shape)
#print(df.head())

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_df.to_csv("data/processed/train.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)