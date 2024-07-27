{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2f3ed82a",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn import svm\n",
    "from sklearn.ensemble import AdaBoostClassifier\n",
    "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n",
    "\n",
    "def classifyWithKNN(train_data, train_labels, test_data, test_labels, n_neighbors):\n",
    "    knn = KNeighborsClassifier(n_neighbors=n_neighbors)\n",
    "    knn.fit(train_data, train_labels)\n",
    "    predictions = knn.predict(test_data)\n",
    "    accuracy = accuracy_score(test_labels, predictions)\n",
    "    print(f\"KNN Accuracy: {accuracy}\")\n",
    "    print(confusion_matrix(test_labels, predictions))\n",
    "    print(classification_report(test_labels, predictions))\n",
    "\n",
    "def classifyWithSVM(train_data, train_labels, test_data, test_labels, C):\n",
    "    model = svm.SVC(C=C, kernel='linear')\n",
    "    model.fit(train_data, train_labels)\n",
    "    predictions = model.predict(test_data)\n",
    "    accuracy = accuracy_score(test_labels, predictions)\n",
    "    print(f\"SVM Accuracy: {accuracy}\")\n",
    "    print(confusion_matrix(test_labels, predictions))\n",
    "    print(classification_report(test_labels, predictions))\n",
    "\n",
    "def classifyWithAdaBoost(train_data, train_labels, test_data, test_labels, n_estimators):\n",
    "    adb = AdaBoostClassifier(n_estimators=n_estimators)\n",
    "    adb.fit(train_data, train_labels)\n",
    "    predictions = adb.predict(test_data)\n",
    "    accuracy = accuracy_score(test_labels, predictions)\n",
    "    print(f\"AdaBoost Accuracy: {accuracy}\")\n",
    "    print(confusion_matrix(test_labels, predictions))\n",
    "    print(classification_report(test_labels, predictions))\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
