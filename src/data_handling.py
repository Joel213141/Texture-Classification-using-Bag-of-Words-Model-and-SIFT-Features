{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3e81507d",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import glob\n",
    "import random\n",
    "import numpy as np\n",
    "from typing import List, Tuple\n",
    "\n",
    "def contentsOfDir(dir_path: str, search_terms: List[str], search_extension_only: bool = True) -> Tuple[str, List[Tuple[str, str]]]:\n",
    "    \"\"\" return the base directory path and list of [file_name, file_extension] tuples \"\"\"\n",
    "    all_files_found = []\n",
    "    if os.path.isdir(dir_path):\n",
    "        base_dir = dir_path\n",
    "        for search_term in search_terms:\n",
    "            glob_search_term = '*' + search_term\n",
    "            if not search_extension_only:\n",
    "                glob_search_term += '*'\n",
    "            files_found = glob.glob(os.path.join(dir_path, glob_search_term))\n",
    "            if len(files_found) > 0:\n",
    "                all_files_found.extend(files_found)\n",
    "    else:\n",
    "        base_dir = os.path.dirname(dir_path)\n",
    "        all_files_found = [dir_path]\n",
    "\n",
    "    files = []\n",
    "    for file_path in all_files_found:\n",
    "        file_name, file_extension = os.path.splitext(os.path.basename(file_path))\n",
    "        files.append((file_name, file_extension))\n",
    "    return base_dir, files\n",
    "\n",
    "def random_split(elements: List, split_ratios: List[float]) -> List[List]:\n",
    "    # ... (same as previously defined)\n",
    "    return splits\n",
    "\n",
    "def write_split_files(file_paths: List[str], data_split_name: str):\n",
    "    # ... (same as previously defined)\n",
    "    pass\n",
    "\n",
    "def imageData(data_dir_path: str = 'data', num_classes_to_use: int = 4, num_images_per_class: int = None, shuffle_seed: int = None):\n",
    "    # ... (same as previously defined)\n",
    "    return {\n",
    "        'train': train_data,\n",
    "        'test': test_data,\n",
    "        'val': val_data,\n",
    "        'class_names': classes_used\n",
    "    }"
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
