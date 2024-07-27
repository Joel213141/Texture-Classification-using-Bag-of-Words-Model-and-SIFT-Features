import os
import glob
import pickle
import argparse
import numpy as np
import cv2 as cv
import random
import torchvision
from typing import Dict, List, Tuple
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import silhouette_score, confusion_matrix, accuracy_score, classification_report
from matplotlib import pyplot as plt
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer, InterclusterDistance


USE_PETS_DATA = False

# the following is a list of image classes that have corrupt images according to opencv (which craps out on them)
# don't forget to use spaces in the class names and make sure to get the capitalisation right
BORKED_IMAGE_CLASSES = ['Egyptian Mau', 'Pomeranian', 'Abyssinian', 'Chihuahua']

# if the following list is not None, the classes specified will be used instead of a random selection
FIXED_CLASS_NAMES = None  # just use random_see 6 on the comand line instead


def contentsOfDir(
    dir_path: str,
    search_terms: List[str],
    search_extension_only: bool = True
) -> Tuple[str, List[Tuple[str, str]]]:
    """ return the base directory path and list of [file_name, file_extension] tuples """
    all_files_found = []
    if os.path.isdir(dir_path):
        base_dir = dir_path
        for search_term in search_terms:
            glob_search_term = '*' + search_term
            if not search_extension_only:
                glob_search_term += '*'
            files_found = glob.glob(os.path.join(dir_path, glob_search_term))
            if len(files_found) > 0:
                all_files_found.extend(files_found)
    else:
        # presume it's actually a single file path
        base_dir = os.path.dirname(dir_path)
        all_files_found = [dir_path]

    files = []
    for file_path in all_files_found:
        file_name, file_extension = os.path.splitext(os.path.basename(file_path))
        files.append((file_name, file_extension))
    return base_dir, files


def random_split(elements: List, split_ratios: List[float]) -> List[List]:
    """ Splits the input list into in the ratios specified.
        Ratios must be > 0 & < 1 and each split must produce at least 1 element
    """
    elements = elements.copy()
    split_ratios = split_ratios.copy()

    ratio_sum = np.sum(split_ratios)
    ratios_positive = np.min(split_ratios) > 0
    if ratio_sum > 1.0 or not ratios_positive:
        raise RuntimeError("split_ratios must be within > 0 &  < 1.0 and split at least 1 element per split")

    split_ratios.pop(-1)  # last split is implied by previous splits
    # get the numbers for the splits from the ratios
    split_numbers = []
    num_elements = len(elements)
    for split_ratio in split_ratios:
        split_numbers.append(round(num_elements * split_ratio))

    splits = []
    for num_elements_to_split in split_numbers:
        num_elements = len(elements)
        random_elements = random.sample(population=range(0, num_elements), k=num_elements_to_split)
        split_elements = []
        for element_id in random_elements:
            # add a random element to a new list
            split_elements.append(elements[element_id])
            # mark the random element for removal
            elements[element_id] = ''
        # remove all elements marked for removal
        for _ in range(len(split_elements)):
            elements.remove('')
        # add the new list of random elements to split lists to return
        splits.append(split_elements)
    # add the remaining elements to the split lists to return
    splits.append(elements)

    return splits


def write_split_files(file_paths: List[str], data_split_name: str):
    new_file_name = './' + data_split_name + '_data_split_file_paths.txt'
    with open(new_file_name, 'w') as fp:
        for item in file_paths:
            fp.write("%s\n" % item)


def imageData(
    data_dir_path: str = 'data',
    num_classes_to_use: int = 4,
    num_images_per_class: int = None,
    shuffle_seed: int = None
):
    """ Downloads the data; if it doesn't already exist, to the specified data dir path,
        creates a dictionary with the class names as well as a dictionary that contains
        image file paths and their labels, split into train, test, validation sets.
    """
    # download the data if it's not already and get the class names/idx
    if USE_PETS_DATA:
        pt_data = torchvision.datasets.OxfordIIITPet(root=data_dir_path, download=True)
        dir_path_to_search = os.path.join(data_dir_path, 'oxford-iiit-pet', 'images')
    else:
        pt_data = torchvision.datasets.DTD(root=data_dir_path, download=True)
        dir_path_to_search = os.path.join(data_dir_path, 'dtd', 'images')
    # NOTE: This code presumes a directory structure like OxfordIIIPet's.
    #       To use the DTD data set, run this script once and it will download all the data.
    #       It's very slow, like an hour ish. Once the images are downloaded,
    #       you'll get an error because the images are in a different dir structure than expected
    #       (the expectation is that all the images have their class name in the file name and are in a single dir).
    #       Now you need to move or copy all the image files from their individual class dirs,
    #       into a single dir called images, under the 1st dir with the data set name
    #       i.e. for using DTD:
    #       > mkdir /path/to/dtd/images/
    #       > cp -r /path/to/dtd/dtd/images/*/*.jpg /path/to/dtd/images/
    #       The next time you execute, all the images will be found.
    #       I could write the code to deal with it, but honestly, this is already so much work for a P task.
    #       It's just faster and easier to copy them into a single dir.

    class_names = pt_data.classes
    if shuffle_seed is not None:
        print(f"Setting Random Seed to: {shuffle_seed}")
        random.seed(shuffle_seed)
    random.shuffle(class_names)

    # just set specific classes that seem to separate well
    if FIXED_CLASS_NAMES is not None:
        class_names = FIXED_CLASS_NAMES

    # construct a dictionary with class names & labels, and don't use any classes with data that is borked
    # (no idea why some images are corrupt, but they are according to opencv)
    class_label_idx = 0
    class_names_to_remove = []
    class_info = {}
    for class_name in class_names:
        if class_name in BORKED_IMAGE_CLASSES:
            print(f"Ignoring class: {class_name}, because it has images that are borked")
            class_names_to_remove.append(class_name)
            continue
        class_info[class_name] = class_label_idx
        class_label_idx += 1
    for class_name in class_names_to_remove:
        class_names.remove(class_name)

    # collect all the image file paths
    supported_file_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    base_dir, files_to_analyze = contentsOfDir(dir_path=dir_path_to_search, search_terms=supported_file_extensions)
    if len(files_to_analyze) < 1:
        raise RuntimeError(f"No files found in the {data_dir_path}")

    # construct a dictionary with all the file paths for each class (and a class label id)
    image_data = {}
    for file_name, file_extension in files_to_analyze:
        # extract the class name from the file name
        file_name_parts = file_name.split(sep='_')
        if USE_PETS_DATA:
            file_name_parts = [part.capitalize() for part in file_name_parts[:-1]]  # ignore the last part (an int)
        else:  # presume the file names are like for dtd i.e. not capitalized
            file_name_parts = file_name_parts[:-1]  # ignore the last part which is an int
        class_name = ' '.join(file_name_parts)
        file_path = os.path.join(base_dir, file_name + file_extension)
        if class_name not in class_names:
            continue  # must be from a class that has borked images
        if class_name not in image_data:
            image_data[class_name] = {
                'label': class_info[class_name],
                'file_paths': []
            }
        image_data[class_name]['file_paths'].append(file_path)

    train_data = {'file_paths': [], 'labels': []}
    test_data = {'file_paths': [], 'labels': []}
    val_data = {'file_paths': [], 'labels': []}
    split_ratios = [0.4, 0.3, 0.3]  # train / test / val
    num_classes_used = 0
    classes_used = []
    for class_name, class_label in class_info.items():
        if num_classes_used >= num_classes_to_use:
            break
        num_classes_used += 1

        class_data = image_data[class_name]
        classes_used.append(class_name)
        file_paths = class_data['file_paths']
        if len(file_paths) < num_images_per_class:
            continue

        # restrict the number of files to be used for each class
        num_files = len(file_paths)
        if num_images_per_class is not None:
            num_files_to_use = min(num_files, num_images_per_class)
            file_paths = file_paths[:num_files_to_use]

        # split the file paths into the specified ratios
        path_splits = random_split(elements=file_paths, split_ratios=split_ratios)

        label = class_data['label']
        train_paths = path_splits[0]
        train_data['file_paths'].extend(train_paths)
        train_labels = [label] * len(train_paths)
        train_data['labels'].extend(train_labels)

        test_paths = path_splits[1]
        test_data['file_paths'].extend(test_paths)
        test_labels = [label] * len(test_paths)
        test_data['labels'].extend(test_labels)

        val_paths = path_splits[2]
        val_data['file_paths'].extend(val_paths)
        val_labels = [label] * len(val_paths)
        val_data['labels'].extend(val_labels)

    # write out the paths used for each data set split
    write_split_files(train_data['file_paths'], 'train')
    write_split_files(test_data['file_paths'], 'test')
    write_split_files(val_data['file_paths'], 'val')

    print(f"Classes selected: {', '.join(classes_used)}")
    return {
        'train': train_data,
        'test': test_data,
        'val': val_data,
        'class_names': classes_used
    }


class BoDModel(object):
    def __init__(self, model_name, num_descriptors, class_names, img_file_paths, descriptor_labels):
        self.name = model_name
        self.img_filenames = img_file_paths
        self.descriptor_labels = descriptor_labels
        self.descriptor_names = class_names
        self.num_descriptors = num_descriptors  # the number of descriptors
        self.training_data = []  # this is the training data required by the K-Means algorithm
        self.descriptors = []  # list of descriptors, which are the centroids of clusters
        self.training_histogram_of_descriptors = []  # list of descriptor histograms of all training images

    def learn(self):
        """ Computes SIFT descriptors from images in the training set
            and clusters them into a fixed number of representative 
            descriptors (cluster centers) which forms a dictionary of descriptors.
            This dictionary of descriptors is used to create a "bag of descriptors" (BoD)
            for each image. The BoD is a histogram where each bin represents the frequency 
            with which each descriptor in the dictionary appears in the image.
        """
        sift = cv.SIFT_create()
        num_keypoints = []  # this is used to store the number of keypoints in each image
        
        # load training images and compute SIFT descriptors
        self.training_data = []
        for filename in self.img_filenames:
            img = cv.imread(filename)
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            list_des = sift.detectAndCompute(img_gray, None)[1]
            if list_des is None:
                num_keypoints.append(0)
            else:
                num_keypoints.append(len(list_des))
                for des in list_des:
                    self.training_data.append(des)

        # cluster SIFT descriptors using K-means algorithm
        kmeans = KMeans(self.num_descriptors, n_init='auto')
        print("Running KMeans to train a BoD model ...")
        kmeans.fit(self.training_data)
        print("...KMeans training of a BoD model complete")
        self.descriptors = kmeans.cluster_centers_
        
        # create descriptor histograms for training images
        print("Creating Descriptor Histograms For Training Data...")
        self.training_histogram_of_descriptors = []
        index = 0
        for i in range(0, len(self.img_filenames)):
            # for each file, create a histogram
            histogram = np.zeros(self.num_descriptors, np.float32)
            # if some keypoints exist
            if num_keypoints[i] > 0:
                for j in range(0, num_keypoints[i]):
                    histogram[kmeans.labels_[j + index]] += 1
                index += num_keypoints[i]
                histogram /= num_keypoints[i]
                self.training_histogram_of_descriptors.append(histogram)
        print("...Descriptor Histograms For Training Data Created")

    def histogramOfDescriptors(self, img_filenames):
        """ converts each image to a histogram of descriptors.
            each SIFT descriptor found in the image is matched 
            to the (Frobenius norm) closest fixed descriptor centers
            found in from the learn method.
            Each one of the fixed descriptors found from the learn method
            is a representative descriptor of a cluster center and not
            necesseraly a descriptor found in any of the images.
        """
        sift = cv.SIFT_create()
        histograms = []

        for filename in img_filenames:
            img = cv.imread(filename)
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            descriptors = sift.detectAndCompute(img_gray, None)[1]
            histogram = np.zeros(self.num_descriptors, np.float32)  # descriptor histogram for the input image
            if descriptors is not None:
                for des in descriptors:
                    # find the best matching descriptor
                    min_distance = np.inf
                    matching_descriptor_id = -1  # initial matching_descriptor_id=-1 means no matching

                    for i in range(0, self.num_descriptors):  # search for the best matching descriptor
                        distance = np.linalg.norm(des - self.descriptors[i])
                        if distance < min_distance:
                            min_distance = distance
                            matching_descriptor_id = i

                    histogram[matching_descriptor_id] += 1

                histogram /= len(descriptors)  # normalise histogram to frequencies

            histograms.append(histogram)

        return histograms


def classifyWithKNN(
    training_histogram_of_descriptors,
    training_descriptor_labels,
    test_histogram_of_descriptors,
    known_class_labels: [str],
    class_names: str,
    neighbour_params: List[int]
):
    best_n = None
    best_accuracy = -1.0
    for num_nearest_neighbours in neighbour_params:
        knn_classifier = KNeighborsClassifier(n_neighbors=num_nearest_neighbours)
        knn_classifier.fit(
            training_histogram_of_descriptors,
            training_descriptor_labels,
        )
        predicted_class_labels = knn_classifier.predict(test_histogram_of_descriptors)
        knn_accuracy = classificationResults(
            num_nearest_neighbours,
            known_class_labels,
            predicted_class_labels,
            class_names,
            classifier_name='KNN',            
        )
        training_score = knn_classifier.score(training_histogram_of_descriptors, training_descriptor_labels)
        print(f"Training score: {round(training_score*100, 2)}")

        if knn_accuracy > best_accuracy:
            best_n = num_nearest_neighbours
            best_accuracy = knn_accuracy

    print(f"Best KNN Accuracy: {round(best_accuracy, 2)} found for K = {best_n}")
    return best_n


def classifyWithSVM(
    training_histogram_of_descriptors,
    training_descriptor_labels,
    test_histogram_of_descriptors,
    known_class_labels: [str],
    class_names: str,
    regularization_params: List[int]
):
    best_c = None
    best_accuracy = -1.0
    for svm_regularization_parameter in regularization_params:
        svm_classifier = svm.SVC(
            C=svm_regularization_parameter,
            kernel='linear'
        )
        svm_classifier.fit(
            training_histogram_of_descriptors,
            training_descriptor_labels,
        )
        predicted_class_labels = svm_classifier.predict(test_histogram_of_descriptors)
        svm_accuracy = classificationResults(
            svm_regularization_parameter,
            known_class_labels,
            predicted_class_labels,
            class_names,
            classifier_name='SVM',
        )
        training_score = svm_classifier.score(training_histogram_of_descriptors, training_descriptor_labels)
        print(f"Training score: {round(training_score*100, 2)}")

        if svm_accuracy > best_accuracy:
            best_c = svm_regularization_parameter
            best_accuracy = svm_accuracy

    print(f"Best SVM Accuracy: {round(best_accuracy, 2)} found for C = {best_c}")

    return best_c


def classifyWithAdaBoost(
    training_histogram_of_descriptors,
    training_descriptor_labels,
    test_histogram_of_descriptors,
    known_class_labels: [str],
    class_names: str,
    ada_boost_params: List[int]
):

    best_num_estimators = None
    best_accuracy = -1.0
    for num_estimators in ada_boost_params:
        adb_classifier = AdaBoostClassifier(
            n_estimators=num_estimators,  # weak classifiers
            random_state=0
        )
        adb_classifier.fit(
            training_histogram_of_descriptors,
            training_descriptor_labels,
        )
        predicted_class_labels = adb_classifier.predict(test_histogram_of_descriptors)
        adb_accuracy = classificationResults(
            num_estimators,
            known_class_labels,
            predicted_class_labels,
            class_names,
            classifier_name='AdaBoost',            
        )
        training_score = adb_classifier.score(training_histogram_of_descriptors, training_descriptor_labels)
        print(f"Training score: {round(training_score*100, 2)}\n")

        if adb_accuracy > best_accuracy:
            best_num_estimators = num_estimators
            best_accuracy = adb_accuracy

    print(f"Best AdaBoost Accuracy: {round(best_accuracy, 2)} found for num_estimators = {best_num_estimators}")

    return best_num_estimators


def classificationResults(
    method_variable,
    known_class_labels,
    predicted_class_labels,
    class_labels,
    classifier_name,
    show_confusion_matrix: bool = True
):
    print(f"\nClassification Performance with {classifier_name}({method_variable})")
    accuracy = accuracy_score(known_class_labels, predicted_class_labels)*100.0
    print(f"Accuracy: {round(accuracy, 2)}%")
    if show_confusion_matrix:
        confusion_table = confusion_matrix(known_class_labels, predicted_class_labels)
        print('Confusion Matrix:')
        print(f"{confusion_table}")
        # print(classification_report(known_class_labels, predicted_class_labels, target_names=class_labels))
    return accuracy


def determineKWithSilhouetteScore(
    data_dir_path: str,
    num_classes_to_use: int,
    num_images_per_class: int = None,
    show_plot: bool = True,
    shuffle_seed: int = None
):

    print("Determining Optimal K for KMeans using the Silhouette method")
    # get the data
    image_data = imageData(
        data_dir_path=data_dir_path,
        num_classes_to_use=num_classes_to_use,
        num_images_per_class=num_images_per_class,
        shuffle_seed=shuffle_seed
    )

    sift = cv.SIFT_create()
    num_keypoints = []  # this is used to store the number of keypoints in each image

    # load training images and compute SIFT descriptors
    train_data = []
    print("Constructing the descriptors for determination of K for KMmeans")
    for filename in image_data['train']['file_paths']:
        img = cv.imread(filename)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        list_des = sift.detectAndCompute(img_gray, None)[1]
        if list_des is None:
            num_keypoints.append(0)
        else:
            num_keypoints.append(len(list_des))
            for des in list_des:
                train_data.append(des)

    print("\nAverage Silhouette Coefficient (SC) Scores for All Cluster Sizes:")
    print(" K     SC Score")
    sc_sizes = []
    sc_scores = []
    k_means_to_try = [5, 15, 25, 35, 45, 55, 65, 75, 85]
    for cluster_size in k_means_to_try:
        print(f"Fitting with K = {cluster_size} ...")
        k_means = KMeans(n_clusters=cluster_size, n_init=10)
        predicted_labels = k_means.fit_predict(train_data)
        print(f"Computing the score...")
        sc_score = silhouette_score(train_data, predicted_labels)
        sc_scores.append(sc_score)
        sc_sizes.append(cluster_size)
        print(f"{str(cluster_size).rjust(2)}     {round(sc_score, 6)}")

    optimal_cluster_size_idx = np.argmax(sc_scores)
    optimal_cluster_size = sc_sizes[optimal_cluster_size_idx]
    print(f"\nOptimal cluster size according to the average silhouette coefficient score is {optimal_cluster_size}")

    if show_plot:
        # plot sc scores for each cluster size
        plt.title("Average SC Scores for Various Cluster Sizes")
        plt.xlabel("Cluster Size")
        plt.ylabel("Average SC Score")
        plt.plot(k_means_to_try, sc_scores, marker='o', linestyle='dashed', linewidth=2, markersize=12)
        plt.show()


def exploreOptimalK(
    data_dir_path: str,
    num_classes_to_use: int,
    num_images_per_class: int = None,
    shuffle_seed: int = None,
    method: str = None
):

    print("Visualising KMeans with various K for the Elbow method")
    # get the data
    image_data = imageData(
        data_dir_path=data_dir_path,
        num_classes_to_use=num_classes_to_use,
        num_images_per_class=num_images_per_class,
        shuffle_seed=shuffle_seed
    )

    sift = cv.SIFT_create()
    num_keypoints = []  # this is used to store the number of keypoints in each image

    # load training images and compute SIFT descriptors
    train_data = []
    print("Constructing the descriptors for determination of K for KMmeans")
    for filename in image_data['train']['file_paths']:
        img = cv.imread(filename)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        list_des = sift.detectAndCompute(img_gray, None)[1]
        if list_des is None:
            num_keypoints.append(0)
        else:
            num_keypoints.append(len(list_des))
            for des in list_des:
                train_data.append(des)
    train_data = np.asarray(train_data)

    if method is None or method == 'silhouette':
        k_means_to_try = [5, 25, 50, 75]
        for k_means_k in k_means_to_try:
            k_means = KMeans(n_clusters=k_means_k, n_init=10)
            # show silhouette analysis
            visualizer = SilhouetteVisualizer(k_means, colors='yellowbrick')
            visualizer.fit(train_data)
            visualizer.show()
            # show inter cluster distance map
            visualizer = InterclusterDistance(k_means)
            visualizer.fit(train_data)
            visualizer.show()
    elif method == 'elbow':
        k_means_to_try = [5, 15, 25, 35, 45, 55, 65, 75, 85]
        k_means = KMeans(n_init=10)
        visualizer = KElbowVisualizer(k_means, k=k_means_to_try)
        visualizer.fit(train_data)
        visualizer.show()
    else:
        raise RuntimeError(
            "Valid parameter values that can be passed for exploring KMeans optimal K are 'silhouette' or 'elbow'"
        )


def main(
    data_dir_path: str,
    num_classes_to_use: int,
    optimal_k: int,
    num_images_per_class: int = None,
    shuffle_seed: int = None
):

    # get the data
    image_data = imageData(
        data_dir_path=data_dir_path,
        num_classes_to_use=num_classes_to_use,
        num_images_per_class=num_images_per_class,
        shuffle_seed=shuffle_seed
    )

    # build the model
    bod_model_path = 'model.pkl'
    # if os.path.isfile(bod_model_path):
    if False:  # remove this once we've ironed out all the bugs
        print("Existing model found in current dir. Loading model for use.")
        with open(bod_model_path, 'rb') as file_to_load:
            bod_model = pickle.load(file_to_load)
    else:
        print("No existing model found in current dir. Training model for use.")
        bod_model = BoDModel(
            model_name='pet_model',
            num_descriptors=optimal_k,
            class_names=image_data['class_names'],
            img_file_paths=image_data['train']['file_paths'],
            descriptor_labels=image_data['train']['labels']
        )
        bod_model.learn()
        with open(bod_model_path, 'wb') as file_to_write:
            pickle.dump(bod_model, file_to_write)

    # test various classification methods to find optimal their hyper-parameters
    class_names = bod_model.descriptor_names  # class labels/names
    training_descriptor_labels = bod_model.descriptor_labels
    training_histogram_of_descriptors = bod_model.training_histogram_of_descriptors

    print("Creating Histogram of Descriptors for the Validation Data...")
    val_histogram_of_descriptors = bod_model.histogramOfDescriptors(image_data['val']['file_paths'])
    val_descriptor_labels = image_data['val']['labels']

    print("Determining Optimal Hyper-Parameters for Classifiers")
    knn_params = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]
    knn_best_param = classifyWithKNN(
        training_histogram_of_descriptors,
        training_descriptor_labels,
        val_histogram_of_descriptors,
        val_descriptor_labels,
        class_names,
        knn_params
    )

    svm_params = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    svm_best_param = classifyWithSVM(
        training_histogram_of_descriptors, 
        training_descriptor_labels,
        val_histogram_of_descriptors,
        val_descriptor_labels,
        class_names,
        svm_params
    )

    ada_boost_params = [10, 50, 100, 150, 200, 250]
    ada_boost_best_param = classifyWithAdaBoost(
        training_histogram_of_descriptors, 
        training_descriptor_labels,
        val_histogram_of_descriptors,
        val_descriptor_labels,
        class_names,
        ada_boost_params
    )

    # test various classifiers with the best hyper-params previously found
    print("Creating Histogram of Descriptors for the Test Data...")
    test_histogram_of_descriptors = bod_model.histogramOfDescriptors(image_data['test']['file_paths'])
    test_descriptor_labels = image_data['test']['labels']

    print("Testing Classifiers with optimal Hyper-Parameters")
    classifyWithKNN(
        training_histogram_of_descriptors,
        training_descriptor_labels,
        test_histogram_of_descriptors,
        test_descriptor_labels,
        class_names,
        [knn_best_param]
    )

    classifyWithSVM(
        training_histogram_of_descriptors,
        training_descriptor_labels,
        test_histogram_of_descriptors,
        test_descriptor_labels,
        class_names,
        [svm_best_param]
    )

    classifyWithAdaBoost(
        training_histogram_of_descriptors,
        training_descriptor_labels,
        test_histogram_of_descriptors,
        test_descriptor_labels,
        class_names,
        [ada_boost_best_param]
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Perform image classification using a BoD model',
    )
    parser.add_argument(
        '--data_dir_path',
        type=str,
        default='data',
        help='Path to store image data.',
    )
    parser.add_argument(
        '--k',
        type=int,
        default=None,
        help='Value to use when training KMeans',
    )
    parser.add_argument(
        '--num_images_per_class',
        type=int,
        default=10,
        help='maximum number of images to use from each class',
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=4,
        help='number of classes to use (randomly selected)',
    )
    parser.add_argument(
        '--shuffle_seed',
        type=int,
        default=None,
        help='seed for randomisation',
    )
    parser.add_argument(
        '-find_k_with_silhouette',
        action='store_true',
        help='Determine the optimal K for KMeans using the silhouette method',
    )
    parser.add_argument(
        '-explore_k_with_elbow',
        action='store_true',
        help='Explore K values for KMeans using the elbow method',
    )
    parser.add_argument(
        '-explore_k_with_silhouette',
        action='store_true',
        help='Explore K values for KMeans using the silhouette method',
    )
    raw_args = parser.parse_args()

    if raw_args.k is None:
        raw_args.k = raw_args.num_classes

    if raw_args.find_k_with_silhouette:
        determineKWithSilhouetteScore(
            data_dir_path=raw_args.data_dir_path,
            num_classes_to_use=raw_args.num_classes,
            num_images_per_class=raw_args.num_images_per_class,
            shuffle_seed=raw_args.shuffle_seed
        )
    elif raw_args.explore_k_with_elbow:
        exploreOptimalK(
            data_dir_path=raw_args.data_dir_path,
            num_classes_to_use=raw_args.num_classes,
            num_images_per_class=raw_args.num_images_per_class,
            shuffle_seed=raw_args.shuffle_seed,
            method='elbow'
        )
    elif raw_args.explore_k_with_elbow:
        exploreOptimalK(
            data_dir_path=raw_args.data_dir_path,
            num_classes_to_use=raw_args.num_classes,
            num_images_per_class=raw_args.num_images_per_class,
            shuffle_seed=raw_args.shuffle_seed,
            method='elbow'
        )
    elif raw_args.explore_k_with_silhouette:
        exploreOptimalK(
            data_dir_path=raw_args.data_dir_path,
            num_classes_to_use=raw_args.num_classes,
            num_images_per_class=raw_args.num_images_per_class,
            shuffle_seed=raw_args.shuffle_seed,
            method='silhouette'
        )
    else:
        main(
            data_dir_path=raw_args.data_dir_path,
            num_classes_to_use=raw_args.num_classes,
            optimal_k=raw_args.k,
            num_images_per_class=raw_args.num_images_per_class,
            shuffle_seed=raw_args.shuffle_seed
        )
