import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def load_and_train_model():

    df = pd.read_csv("data/preprocessed_data.csv")
    df.columns = df.columns.str.strip()

    df = df.select_dtypes(include=['number'])
    df = df.fillna(df.mean())

    # Target
    sped_thresh = df['SPED'].quantile(0.75)
    vsbk_thresh = df['VSBK'].quantile(0.25)

    df['Storm_actual'] = (
        (df['SPED'] > sped_thresh) &
        (df['VSBK'] < vsbk_thresh)
    ).astype(int)

    X = df.drop(['Storm_actual', 'SPED', 'VSBK'], axis=1)
    y = df['Storm_actual']

    # Split (for accuracy comparison)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    return model, X.columns, df, train_acc, test_acc
   