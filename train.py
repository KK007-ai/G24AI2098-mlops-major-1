# train.py
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

def main():
    print("Loading Olivetti faces...")
    data = fetch_olivetti_faces(shuffle=True, random_state=42)
    X = data.data
    y = data.target

    print("Splitting: 70% train, 30% test ...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

    print("Training DecisionTreeClassifier ...")
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    print("Evaluating on test set ...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")

    print("Saving model and testset to savedmodel.pth ...")
    joblib.dump({'model': clf, 'X_test': X_test, 'y_test': y_test}, "savedmodel.pth")
    print("Done: savedmodel.pth created.")

if __name__ == "__main__":
    main()
