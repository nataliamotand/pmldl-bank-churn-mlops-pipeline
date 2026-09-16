import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/raw/churn.csv")

# remove identificadores sem valor preditivo
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

# remove outliers
def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

numeric_columns = ["CreditScore", "Age", "Balance", "EstimatedSalary"]
df = remove_outliers_iqr(df, numeric_columns)
#print(df.shape)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_df.to_csv("data/processed/train.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)