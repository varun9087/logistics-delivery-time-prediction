from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Sklearn Model
    model_sklearn = LinearRegression()
    model_sklearn.fit(X_train, y_train)

    joblib.dump(model_sklearn, "models/linear_regression_model.pkl")

    return X_train, X_test, y_train, y_test, scaler, model_sklearn