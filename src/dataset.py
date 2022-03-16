try:
    from cuml.manifold.Tumap import UMAP

    print("[INFO]: Using cuml UMAP")
except:
    from umap import UMAP

    print("[INFO]: CUML not available; using umap-learn UMAP")

import os
import io
import faiss
import torch
import pickle
import base64
import shutil
import hdbscan
import numpy as np
import pandas as pd
from PIL import Image
from zipfile import ZipFile

config = {
    "debug": True,
    "assets": "assets",
    "unzipped_dir": "assets/unzipped",
    "resources_dir": "assets/resources",
    "clusters_dir": "assets/clusters",
}

from torchvision import transforms, models
from featureExtraction import extract_features_paths


class Dataset:
    def __init__(self):
        self.name = None
        self.directory = None
        self.img_paths = None
        self.features = None
        self.embeddings = None
        self.labels = None
        self.test_image = None
        self.index = None

    ################################################################################
    """Functions for Dataset Processing"""
    ################################################################################

    def decode_and_extract_zip(self, content_string: str) -> ZipFile:
        # Decode the base64 string
        content_decoded = base64.b64decode(content_string)
        # Use BytesIO to handle the decoded content
        zip_str = io.BytesIO(content_decoded)
        # Now you can use ZipFile to take the BytesIO output
        zip_obj = ZipFile(zip_str, "r")
        # if dataset directory not empty, delete
        dest_dir = config["unzipped_dir"]
        # remove existing dir
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        # Extract image files from zip folder and save directory data located
        zip_obj.extractall(dest_dir)
        self.directory = self.get_unzip_dir()
        return

    def load_features_paths(self) -> None:
        # create names for the uploaded files' potential resources
        img_paths_pickle = f"{self.name}_img_paths.pickle"
        features_pickle = f"{self.name}_features.pickle"
        # create paths to these resource files
        path_to_img_paths_pickle = os.path.join(
            config["resources_dir"], img_paths_pickle
        )
        path_to_features_pickle = os.path.join(config["resources_dir"], features_pickle)
        # If img_paths/features for this dataset are available use them, else save after generating
        # so, list files currently in resources folder
        if not os.path.exists(config["resources_dir"]):
            os.mkdir(config["resources_dir"])
        resources = os.listdir(config["resources_dir"])
        if img_paths_pickle and features_pickle in resources:
            with open(path_to_img_paths_pickle, "rb") as f:
                self.img_paths = pickle.load(f)
            with open(path_to_features_pickle, "rb") as f:
                self.features = pickle.load(f)
        else:
            # Extract using model and save features+img_paths
            self.features, self.img_paths = extract_features_paths(self.directory)
            pickle.dump(self.features, open(path_to_features_pickle, "wb"))
            pickle.dump(self.img_paths, open(path_to_img_paths_pickle, "wb"))

        print(f"[INFO][STARTED] Adding features to FAISS index")
        self.index = faiss.IndexFlatL2(2048)
        self.index.add(self.features)
        print(f"[INFO][DONE] Adding features to FAISS index")

        if config["debug"]:
            print(
                "[DEBUG]: Size of dataset['features']: "
                + str(self.features.__sizeof__())
                + "bytes"
            )
            print(
                "[DEBUG]: Size of dataset['img_paths']: "
                + str(self.img_paths.__sizeof__())
                + "bytes"
            )

        return

    def find_similar_imgs(self, test_image_path: str) -> list:
        transform = transforms.Compose(
            [
                transforms.Resize(size=[224, 224], interpolation=2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print("[INFO] Using Device:", DEVICE)
        model = models.resnet101(pretrained=True, progress=True)
        model.to(DEVICE)

        def pooling_output(x):
            for layer_name, layer in model._modules.items():
                x = layer(x)
                if layer_name == "avgpool":
                    break
            return x

        PIL_img = Image.open(test_image_path).convert("RGB")
        input_tensor = transform(PIL_img)
        input_tensor = input_tensor.view(1, *input_tensor.shape)

        with torch.no_grad():
            query_descriptors = pooling_output(input_tensor.to(DEVICE)).cpu().numpy()
            distance, indices = self.index.search(
                query_descriptors.reshape(1, 2048), 12
            )

        return indices

    def generate_clusters(self, n_neighbors, min_dist, min_cluster_size, min_samples):
        """Clusters the input feature vectors and returns embeddings + labels.
        - [INPUTS]:
            - features: List of vertically stacked feature vectors with shape [[n,],].
            - n_neighbors: UMAP Parameter
        - [OUTPUTS]:
            - clusteredFeatures: Dictionary containing the following:
                - embeddings: coordinates of each vector (image), shape can be
                    [[x, y, z],...] or [[x, y],...] depending on UMAP params.
                - labels: list of the cluster each image belongs to with shape [n,].
        - [ALGORITHMS]:
            - [UMAP]: To reduce feature dimensionality to create embeddings.
            - [HDBSCAN]: To cluster feature embeddings."""
        self.embeddings = self.setup_umap(n_neighbors, min_dist)
        print(f"[INFO][STARTED]: Clustering with HDBSCAN.")
        self.labels = hdbscan.HDBSCAN(
            min_samples=min_samples,
            min_cluster_size=min_cluster_size,
        ).fit_predict(self.embeddings)
        print(f"[INFO][DONE]: Clustering with HDBSCAN.")
        return

    def setup_umap(self, n_neighbors, min_dist, n_components=3):
        """Returns UMAP feature embeddings with specified number of components
        Outputs:
            - embeddings: coordinates of each vector (image), shape can be
                    [[x, y, z],...] or [[x, y],...] depending on n_components.
        Inputs:
            - features: List of vertically stacked feature vectors with shape [[n,],].
            - n_neighbors: UMAP Parameter
            - n_components: Number of dimensions to reduce to, default is 3."""

        # create names for the uploaded files' potential resources
        if isinstance(min_dist, int):
            min_dist_str = str(min_dist)
        else:
            min_dist_str = str(min_dist).split(".")[1]
        embeddings_pickle = f"{self.name}_umap_{n_components}D_embeddings_{n_neighbors}_{min_dist_str}.pickle"
        # create paths to these resource files
        path_to_embeddings_pickle = os.path.join(
            config["resources_dir"], embeddings_pickle
        )
        # If img_paths/features for this dataset are available, use them, else save after generating
        resources = os.listdir(config["resources_dir"])
        if embeddings_pickle in resources:
            with open(path_to_embeddings_pickle, "rb") as f:
                embeddings = pickle.load(f)
        else:
            print(
                f"[INFO][STARTED]: Dimensionality reduction... this could take a few minutes."
            )
            print("self.features before umapping", self.features)
            embeddings = UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=n_components,
                metric="correlation",
                random_state=24,
                transform_seed=24,
                verbose=True,
            ).fit_transform(self.features)
            pickle.dump(embeddings, open(path_to_embeddings_pickle, "wb"))
            print(f"[INFO][DONE]: Dimensionality reduction.")

        return embeddings

    def calculate_percent_clustered(self):
        """input:
            - labels: List of labels for all images
            - numPoints: number of total points (images)
        output: percent_clustered as a float"""
        num_clustered = self.labels >= 0
        percent_clustered = round(100 * np.sum(num_clustered) / len(self.img_paths), 2)
        return percent_clustered

    def create_clusters_zip(self):
        # Export clusters to folders
        imgs_with_clusters = [
            [self.labels[i], self.img_paths[i]] for i in range(len(self.labels))
        ]

        df = pd.DataFrame(
            imgs_with_clusters,
            columns=["cluster", "imgPath"],
            index=range(len(self.labels)),
        )

        # First add the noise cluster to the list then the remaining clusters as lists
        clusters_dict = {}

        cluster_noise = df[df["cluster"] == -1]["imgPath"].tolist()
        clusters_dict["cluster_noise"] = cluster_noise
        # use set() to iterate over the number of unique classes minus the noise class
        for i in range(len(set(self.labels)) - 1):
            subFrame = df[df["cluster"] == i]
            clusters_dict["cluster_%s" % i] = subFrame["imgPath"].tolist()

        # [TODO]: Extract assets/clusters to config file
        cluster_dir = config["clusters_dir"]

        # Remove pre-existing cluster directory
        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)
        else:
            shutil.rmtree(cluster_dir)  # Removes all cluster subdirectories!
            # If a zip download file already exists, delete it
            if os.path.exists(os.path.join(config["assets"], "clusters.zip")):
                os.remove(os.path.join(config["assets"], "clusters.zip"))
            os.makedirs(cluster_dir)

        for cluster in clusters_dict:
            # create a folder for the current cluster
            current_output_dir = os.path.join(cluster_dir, str(cluster))
            os.makedirs(current_output_dir)
            # copy images from original location to new folder
            for path in clusters_dict[str(cluster)]:
                shutil.copy(path, current_output_dir)
        Dataset.make_archive(cluster_dir, cluster_dir + ".zip")
        return

    @classmethod
    def make_archive(cls, source, destination):
        # used to take a folder and create a zip with it placed in the containing dir
        base = os.path.basename(destination)
        name = base.split(".")[0]
        format = base.split(".")[1]
        archive_from = os.path.dirname(source)
        archive_to = os.path.basename(source.strip(os.sep))
        shutil.make_archive(name, format, archive_from, archive_to)
        shutil.move("%s.%s" % (name, format), destination)
        if os.path.exists(source):
            shutil.rmtree(source)
        return

    @classmethod
    def del_folder_contents(cls, folder: str) -> None:
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print("Failed to delete %s. Reason: %s" % (file_path, e))

    def prepare_preview_download(self, selected_img_idxs):
        preview_files_dir = os.path.join(config["assets"], "preview_2D")
        preview_zip_path = os.path.join(config["assets"], "preview_2D.zip")
        # remove previous file
        if os.path.exists(preview_files_dir):
            shutil.rmtree(preview_files_dir)
        if not os.path.exists(preview_files_dir):
            os.makedirs(preview_files_dir)
        if os.path.exists(preview_zip_path):
            os.remove(preview_zip_path)
        # create directory with the preview images copied from original dataset
        for idx in selected_img_idxs:
            img_path = self.img_paths[idx]
            shutil.copy(img_path, preview_files_dir)
        Dataset.make_archive(preview_files_dir, preview_files_dir + ".zip")
        return

    def reset_dataset(self):
        new_dataset_obj = Dataset()
        # if the
        if not os.path.exists(config["unzipped_dir"]):
            os.makedirs(config["unzipped_dir"])
        else:
            shutil.rmtree(config["unzipped_dir"])
            os.makedirs(config["unzipped_dir"])
        return new_dataset_obj

    def gen_img_uri(self, img_index) -> str:
        """genImgURI Open image file at provided path with PIL and encode to
        img_uri string.

        Args:
            image_path (str): Path to image file.

        Returns:
            img_uri (str): str containing image bytes viewable by html.Img
        """
        im = Image.open(self.img_paths[img_index])
        # dump it to base64
        buffer = io.BytesIO()
        im.save(buffer, format="jpeg")
        encoded_image = base64.b64encode(buffer.getvalue()).decode()
        im_url = "data:image/jpeg;base64, " + encoded_image

        return im_url

    def get_unzip_dir(self):
        """The unzipped files get stored into assets folder as files or a directory containing files.
        Returns directory as string"""
        # A folder was created when they uploaded
        filenames = os.listdir(config["unzipped_dir"])
        if len(filenames) == 0:
            print("[DEBUGLOG]: Nothing was uploaded")
        else:
            if len(filenames) == 1:
                print("[INFO]: Uploaded zip file stored as a folder.", filenames)
                dataDir = config["unzipped_dir"]
            else:
                print("[INFO]: Uploaded zip file stored as individual files.")
                dataDir = config["unzipped_dir"].split("/")[0]

            print("[DEBUGLOG]: DataDir =", dataDir)

        return dataDir
