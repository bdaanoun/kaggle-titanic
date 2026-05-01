from data_loader import load_data
from preprocessing import preprocess
from model import train_model
from predict import make_submission

# load data
train_df, test_df = load_data()

# preprocess
train_df, test_df = preprocess(train_df, test_df)

# split features
X = train_df.drop("Survived", axis=1)
y = train_df["Survived"]

# train (prints train/test accuracy internally, then retrains on all data)
model = train_model(X, y)

# predict + submission
make_submission(model, test_df)