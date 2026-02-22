from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd

def test_robustness(classifier, X_style_modified: pd.DataFrame, y_true: pd.Series):
    y_pred = classifier.clf.predict(X_style_modified)
    
    print("--- Robustness Test: Style-Modified Prompts ---")
    report = classification_report(y_true, y_pred, zero_division=0)
    print(report)
    
    cm = confusion_matrix(y_true, y_pred, labels=classifier.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap='Reds', ax=ax, xticks_rotation=45) #Using Reds to differentiate from the standard test
    plt.title("Robustness Test: Confusion Matrix on Unseen Genre")
    plt.tight_layout()
    plt.show()
    
    return report