import os
from tqdm import tqdm
import shutil

source_dir = "assets/unzipped/vtc_endface"
unzip_dir = "assets/unzipped"

os.makedirs(source_dir)

for file_name in tqdm(os.listdir(unzip_dir)):
    shutil.move(os.path.join(unzip_dir, file_name), source_dir)

# shutil.rmtree(source_dir)


# from featureExtraction import extract_features_paths

# unzip_dir = "assets/unzipped"

# features, img_paths = extract_features_paths(unzip_dir.split("/")[0])

# print(features[0])
# print(img_paths[0])
