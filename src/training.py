from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

def train_model(df):
    x= df.drop(['price', 'mainroad', 'guestroom', 'basement','hotwaterheating','airconditioning','prefarea','furnishingstatus' ], axis=1)
    y = df['price']
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, 
        test_size=0.20, 
        random_state=80
        )
    model = LinearRegression()
    model.fit(x_train, y_train)

    with open('model/Housing_Price_Model.pkl', 'wb') as f:
        pickle.dump(model, f)
    return model, x_test, y_test