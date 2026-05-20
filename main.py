from src.loading import load_data
from src.training import train_model
from src.evaluation import evaluate_model

df = load_data('data\Housing.csv')

model, x_test, y_test = train_model(df)

evaluate_model(model, x_test, y_test)