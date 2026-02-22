from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class ModelClassifier:
    
    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        self.clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.feature_names = None
        self.classes_ = None
    def train(self, X: pd.DataFrame, y: pd.Series, feature_names: list):
        self.feature_names = feature_names
        self.clf.fit(X, y)
        self.classes_ = self.clf.classes_
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, labels=None) -> dict:
        y_pred = self.clf.predict(X_test)
        print("--- Classification Report ---")
        print(classification_report(y_test, y_pred, labels=labels))
        
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels or self.classes_)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(cmap='Blues', ax=ax, xticks_rotation=45)
        plt.title("Confusion Matrix: Model Identification")
        plt.tight_layout()
        plt.show()
        
        return y_pred
    def plot_feature_importance(self, top_k: int = 20):
        importances = pd.Series(self.clf.feature_importances_, index=self.feature_names).nlargest(top_k)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances.values, y=importances.index, palette="viridis")
        plt.title(f"Top {top_k} Most Important Stylometric Features")
        plt.xlabel("Gini Importance")
        plt.tight_layout()
        plt.show()