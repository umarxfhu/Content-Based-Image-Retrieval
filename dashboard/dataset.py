try:
    from cuml.manifold.Tumap import UMAP

    print("[INFO]: Using cuml UMAP")
except:
    from umap import UMAP

    print("[INFO]: CUML not available; using umap-learn UMAP")

import os
import io
import pickle
import base64
import shutil
import hdbscan
import numpy as np
import pandas as pd
from PIL import Image
from zipfile import ZipFile

from config import config
from featureExtraction import extract_features_paths


class Dataset:
    def __init__(self):
        self.name = None
        self.directory = None
        self.img_paths = None
        self.features = None
        self.embeddings = None
        self.labels = None

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

    def generate_clusters(self, n_neighbors):
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

        print(
            f"[INFO][STARTED]: Dimensionality reduction... this could take a few minutes."
        )
        self.embeddings = self.setup_umap(n_neighbors)
        print(f"[INFO][DONE]: Dimensionality reduction.")
        print(f"[INFO][STARTED]: Clustering with HDBSCAN.")
        self.labels = hdbscan.HDBSCAN(
            min_samples=10,
            min_cluster_size=250,
        ).fit_predict(self.embeddings)
        print(f"[INFO][DONE]: Clustering with HDBSCAN.")
        return

    def setup_umap(self, n_neighbors, n_components=3):
        """Returns UMAP feature embeddings with specified number of components
        Outputs:
            - embeddings: coordinates of each vector (image), shape can be
                    [[x, y, z],...] or [[x, y],...] depending on n_components.
        Inputs:
            - features: List of vertically stacked feature vectors with shape [[n,],].
            - n_neighbors: UMAP Parameter
            - n_components: Number of dimensions to reduce to, default is 3."""

        # create names for the uploaded files' potential resources
        embeddings_pickle = (
            f"{self.name}_umap_{n_components}D_embeddings_{n_neighbors}.pickle"
        )
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
            embeddings = UMAP(
                n_neighbors=n_neighbors,
                min_dist=0.00,
                n_components=n_components,
                metric="correlation",
                random_state=24,
                transform_seed=24,
                verbose=True,
            ).fit_transform(self.features)
            pickle.dump(embeddings, open(path_to_embeddings_pickle, "wb"))

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
        def make_archive(source, destination):
            base = os.path.basename(destination)
            name = base.split(".")[0]
            format = base.split(".")[1]
            archive_from = os.path.dirname(source)
            archive_to = os.path.basename(source.strip(os.sep))
            shutil.make_archive(name, format, archive_from, archive_to)
            shutil.move("%s.%s" % (name, format), destination)

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

        for i in range(len(set(self.labels)) - 1):
            subFrame = df[df["cluster"] == i]
            clusters_dict["cluster_%s" % i] = subFrame["imgPath"].tolist()

        # [TODO]: Extract assets/clusters to config file
        cluster_dir = os.path.join(config["assets"], "clusters")

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

        make_archive(cluster_dir, cluster_dir + ".zip")

    def reset_dataset(self):
        new_dataset_obj = Dataset()
        shutil.rmtree(config["unzipped_dir"])
        os.mkdir(config["unzipped_dir"])
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
