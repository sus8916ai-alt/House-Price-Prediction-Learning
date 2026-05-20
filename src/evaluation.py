from sklearn.metrics import mean_absolute_error, r2_score

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("MAE", mae)
    print("R2 Score", r2)
