import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_model_selection import train_test_split


def model_hazirlik_ve_egitim(df):

    X = df.drop(columns=["Fiyat_TL"])
    y = df["Fiyat_TL"]


    X_encoded = pd.get_dummies(X, drop_first=True)
    model_sutunlari = X_encoded.columns

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=12)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)

    metrikler = {
        "RF_R2": r2_score(y_test, rf_preds),
        "RF_RMSE": root_mean_squared_error(y_test, rf_preds),
        "LR_R2": r2_score(y_test, lr_preds),
        "LR_RMSE": root_mean_squared_error(y_test, lr_preds),
    }

    final_rf_model = RandomForestRegressor(
        n_estimators=100, random_state=42, max_depth=12
    )
    final_rf_model.fit(X_encoded, y)

    return final_rf_model, model_sutunlari, metrikler