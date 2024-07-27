{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a1fd27d8",
   "metadata": {},
   "outputs": [],
   "source": [
    "import argparse\n",
    "from data_handling import imageData\n",
    "from feature_extraction import BoDModel\n",
    "from classification import classifyWithKNN, classifyWithSVM, classifyWithAdaBoost\n",
    "\n",
    "def main():\n",
    "    parser = argparse.ArgumentParser(description='Image Classification System')\n",
    "    parser.add_argument('--data_dir', type=str, default='./data', help='Directory for storing image data')\n",
    "    args = parser.parse_args()\n",
    "\n",
    "    # Load and preprocess data\n",
    "    data = imageData(args.data_dir)\n",
    "    \n",
    "    # Example classifier usage\n",
    "    classifyWithKNN(data['train']['features'], data['train']['labels'], data['test']['features'], data['test']['labels'], n_neighbors=5)\n",
    "    classifyWithSVM(data['train']['features'], data['train']['labels'], data['test']['features'], data['test']['labels'], C=1.0)\n",
    "    classifyWithAdaBoost(data['train']['features'], data['train']['labels'], data['test']['features'], data['test']['labels'], n_estimators=50)\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    main()\n"
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
