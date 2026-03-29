from train_model import load_and_train_model
from ui import render_ui

model, feature_cols, df, train_acc, test_acc = load_and_train_model()

render_ui(model, feature_cols, df, train_acc, test_acc)