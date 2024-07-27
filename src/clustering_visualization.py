{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "89a0b488",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import cv2 as cv\n",
    "from sklearn.cluster import KMeans\n",
    "from sklearn.metrics import silhouette_score\n",
    "from matplotlib import pyplot as plt\n",
    "from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer, InterclusterDistance\n",
    "\n",
    "def determineKWithSilhouetteScore(data, k_range):\n",
    "    print(\"\\nAverage Silhouette Coefficient (SC) Scores for All Cluster Sizes:\")\n",
    "    print(\" K     SC Score\")\n",
    "    sc_scores = []\n",
    "    for k in k_range:\n",
    "        k_means = KMeans(n_clusters=k, n_init=10)\n",
    "        predicted_labels = k_means.fit_predict(data)\n",
    "        sc_score = silhouette_score(data, predicted_labels)\n",
    "        sc_scores.append(sc_score)\n",
    "        print(f\"{str(k).rjust(2)}     {round(sc_score, 6)}\")\n",
    "\n",
    "    optimal_k = k_range[np.argmax(sc_scores)]\n",
    "    print(f\"\\nOptimal cluster size according to the average silhouette coefficient score is {optimal_k}\")\n",
    "\n",
    "    # Plot SC Scores\n",
    "    plt.figure()\n",
    "    plt.plot(k_range, sc_scores, marker='o')\n",
    "    plt.title('Silhouette Scores for Different Values of K')\n",
    "    plt.xlabel('Number of Clusters (K)')\n",
    "    plt.ylabel('Silhouette Score')\n",
    "    plt.show()\n",
    "\n",
    "def exploreOptimalK(data, k_range):\n",
    "    print(\"Visualising KMeans with various K for the Elbow method\")\n",
    "    visualizer = KElbowVisualizer(KMeans(), k=k_range)\n",
    "    visualizer.fit(data)\n",
    "    visualizer.show()\n"
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
