{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "eebc0c48",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import cv2 as cv\n",
    "from sklearn.cluster import KMeans\n",
    "\n",
    "class BoDModel(object):\n",
    "    def __init__(self, model_name, num_descriptors, class_names, img_file_paths, descriptor_labels):\n",
    "        self.name = model_name\n",
    "        self.img_filenames = img_file_paths\n",
    "        self.descriptor_labels = descriptor_labels\n",
    "        self.descriptor_names = class_names\n",
    "        self.num_descriptors = num_descriptors  # the number of descriptors\n",
    "        self.training_data = []  # this is the training data required by the K-Means algorithm\n",
    "        self.descriptors = []  # list of descriptors, which are the centroids of clusters\n",
    "        self.training_histogram_of_descriptors = []  # list of descriptor histograms of all training images\n",
    "\n",
    "    def learn(self):\n",
    "        sift = cv.SIFT_create()\n",
    "        num_keypoints = []  # this is used to store the number of keypoints in each image\n",
    "        \n",
    "        # load training images and compute SIFT descriptors\n",
    "        for filename in self.img_filenames:\n",
    "            img = cv.imread(filename)\n",
    "            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)\n",
    "            list_des = sift.detectAndCompute(img_gray, None)[1]\n",
    "            if list_des is None:\n",
    "                num_keypoints.append(0)\n",
    "            else:\n",
    "                num_keypoints.append(len(list_des))\n",
    "                for des in list_des:\n",
    "                    self.training_data.append(des)\n",
    "\n",
    "        # cluster SIFT descriptors using K-means algorithm\n",
    "        kmeans = KMeans(self.num_descriptors, n_init='auto')\n",
    "        kmeans.fit(self.training_data)\n",
    "        self.descriptors = kmeans.cluster_centers_\n",
    "        \n",
    "        # create descriptor histograms for training images\n",
    "        index = 0\n",
    "        for i in range(len(self.img_filenames)):\n",
    "            histogram = np.zeros(self.num_descriptors, np.float32)\n",
    "            if num_keypoints[i] > 0:\n",
    "                for j in range(num_keypoints[i]):\n",
    "                    histogram[kmeans.labels_[j + index]] += 1\n",
    "                index += num_keypoints[i]\n",
    "                histogram /= num_keypoints[i]\n",
    "                self.training_histogram_of_descriptors.append(histogram)\n",
    "\n",
    "    def histogramOfDescriptors(self, img_filenames):\n",
    "        sift = cv.SIFT_create()\n",
    "        histograms = []\n",
    "\n",
    "        for filename in img_filenames:\n",
    "            img = cv.imread(filename)\n",
    "            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)\n",
    "            descriptors = sift.detectAndCompute(img_gray, None)[1]\n",
    "            histogram = np.zeros(self.num_descriptors, np.float32)\n",
    "            if descriptors is not None:\n",
    "                for des in descriptors:\n",
    "                    min_distance = np.inf\n",
    "                    matching_descriptor_id = -1\n",
    "                    for i in range(self.num_descriptors):\n",
    "                        distance = np.linalg.norm(des - self.descriptors[i])\n",
    "                        if distance < min_distance:\n",
    "                            min_distance = distance\n",
    "                            matching_descriptor_id = i\n",
    "                    histogram[matching_descriptor_id] += 1\n",
    "                histogram /= len(descriptors)\n",
    "            histograms.append(histogram)\n",
    "        return histograms\n"
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
