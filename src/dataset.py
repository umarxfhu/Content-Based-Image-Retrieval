try:
    from cuml.manifold.Tumap import UMAP

    print("[INFO]: Using cuml UMAP")
except:
    from umap import UMAP

    print("[INFO]: CUML not available; using umap-learn UMAP")

import os
import io
import gc
import faiss
import torch
import pickle
import base64
import shutil
import orjson
import hdbscan
import numpy as np
import pandas as pd
from PIL import Image
from redis import Redis
from tqdm import tqdm
from zipfile import ZipFile

from torchvision import transforms, models
from featureExtraction import extract_features_paths

debug = False


def get_resource_dir(debug, session_id, dataset_name):
    if debug:
        resources_dir = f"debug_resources/resources"
    else:
        resources_dir = f"assets/{session_id}/{dataset_name}/resources"
    return resources_dir


################################################################################
"""Functions for Dataset Processing"""
################################################################################


def arrange_unzip_dir(unzip_dir, dataset_name):
    """The unzipped files get stored into assets folder as files or a directory containing files."""
    # A folder was created when they uploaded
    filenames = os.listdir(unzip_dir)
    if len(filenames) == 0:
        print("[DEBUGLOG]: Nothing was uploaded")
    else:
        if len(filenames) == 1:
            print("[DEBUG]: Uploaded zip file was stored as a folder.", filenames)
        else:
            print("[INFO]: Uploaded zip file stored as individual files.")
            # joining into the path of this folder that we want to remove
            destination_dir = os.path.join(unzip_dir, dataset_name)
            if not os.path.exists(destination_dir):
                os.makedirs(destination_dir)
            for file_name in os.listdir(unzip_dir):
                shutil.move(os.path.join(unzip_dir, file_name), destination_dir)


def move_unzip_uploaded_file(session_id, file_name):
    dataset_name = os.path.splitext(file_name[0])[0]
    user_dataset_dir = f"assets/{session_id}/{dataset_name}"
    temp_upload_dir = f"assets/temp/{session_id}"
    temp_zip_path = os.path.join(temp_upload_dir, file_name[0])
    user_unzip_path = os.path.join(user_dataset_dir, "unzipped")

    # print block
    print("[INFO]: dataset_name:", dataset_name)
    print("[INFO]: user_dataset_dir:", user_dataset_dir)
    print("[INFO]: temp_upload_dir:", temp_upload_dir)
    print("[INFO]: temp_zip_path:", temp_zip_path)
    print("[INFO]: user_unzip_path:", user_unzip_path)

    if not os.path.exists(user_unzip_path):
        os.makedirs(user_unzip_path)
    shutil.move(temp_zip_path, user_unzip_path)
    print(f"[INFO]: moved {temp_zip_path} to {user_unzip_path}")
    file_path = os.path.join(user_unzip_path, f"{file_name[0]}")
    print("file_path:", file_path)
    with ZipFile(file_path, "r") as zipObj:
        zipObj.extractall(user_unzip_path)
    print(f"[INFO]: extracted {file_path} to {user_unzip_path}")
    unzip_dir = os.path.join(user_unzip_path, dataset_name)
    # remove the zip file
    os.remove(file_path)
    # clean dir structure if needed
    print("[INFO]: Arranging", user_unzip_path, "and", dataset_name)
    arrange_unzip_dir(user_unzip_path, dataset_name)
    # remove the mac zip dir if necessary
    mac_zip_dir = os.path.join(user_unzip_path, dataset_name, "__MACOSX")
    print("mac_zip_dir:", mac_zip_dir)
    if os.path.exists(mac_zip_dir):
        print("removing the ")
        shutil.rmtree(mac_zip_dir)
    # remove the temp dir
    shutil.rmtree(temp_upload_dir)


def decode_and_extract_zip(session_id: str, dataset_name: str, content_string: str):
    """decode_and_extract_zip extract users uploaded zip file into the appropriate folder on the server filesystem

    Args:
        session_id (str): identifies user (stored in browser session)
        dataset_name (str): folder name to save the dataset under
        content_string (str): actual data string
    """
    # Decode the base64 string
    content_decoded = base64.b64decode(content_string)
    # Use BytesIO to handle the decoded content
    zip_str = io.BytesIO(content_decoded)
    # del content_decoded
    # Now you can use ZipFile to take the BytesIO output
    zip_obj = ZipFile(zip_str, "r")
    # check if users unzip path exists, else create it
    unzip_dir = f"assets/{session_id}/{dataset_name}/unzipped"
    # if same name dataset is uploaded delete previous files
    if os.path.exists(unzip_dir):
        shutil.rmtree(unzip_dir)
    # make sure to recreate the needed directory and unzip
    os.makedirs(unzip_dir)
    zip_obj.extractall(unzip_dir)
    # correct for inconsistent unzipping (sometimes save as files or as folder with files)
    gc.collect()
    arrange_unzip_dir(unzip_dir, dataset_name)


def load_features_paths(session_id: str, dataset_name: str, redis_client: Redis):
    """load_features_paths checks file system cache for cached features, if
    not found generate and save features/faiss index/img_paths.
    Note: features and faiss index are saved as pickle files and list of
    img_paths is saved in redis cache.

    Args:
        session_id (str):
        dataset_name (str):
        redis_client (Redis):
    """
    # Create path name for resources
    resources_dir = get_resource_dir(debug, session_id, dataset_name)
    print("resources_dir:", resources_dir)
    # create names for the uploaded files' potential features
    features_pickle = f"{dataset_name}_features.pickle"
    index_pickle = f"{dataset_name}_index.pickle"
    # create paths to these resource files
    path_to_features_pickle = os.path.join(resources_dir, features_pickle)
    path_to_index_pickle = os.path.join(resources_dir, index_pickle)
    # If img_paths/features for this dataset are available use them, else save after generating
    # so, list files currently in resources folder
    if not os.path.exists(resources_dir):
        os.mkdir(resources_dir)
    resources = os.listdir(resources_dir)
    # load pickle of features if present
    print("resources:", resources)
    if features_pickle and index_pickle in resources:
        with open(path_to_features_pickle, "rb") as f:
            features = pickle.load(f)
    else:
        # Extract using model and save features+img_paths
        unzip_dir = f"assets/{session_id}/{dataset_name}/unzipped"
        features, img_paths = extract_features_paths(unzip_dir)
        index = faiss.IndexFlatL2(2048)
        index.add(features)
        # Cache features and index on file system
        pickle.dump(features, open(path_to_features_pickle, "wb"))
        pickle.dump(index, open(path_to_index_pickle, "wb"))
        pickle.dump(
            img_paths,
            open(os.path.join(resources_dir, f"{dataset_name}_img_paths.pickle"), "wb"),
        )
        # Cache img_paths on redis as they are needed for rapid access later
        # serialize, returns byte instead of string
        img_paths_json_bytes = orjson.dumps(img_paths)
        redis_client.set(f"{session_id}:{dataset_name}:img_paths", img_paths_json_bytes)

    return features


def setup_umap(
    redis_client, session_id, dataset_name, n_neighbors, min_dist, n_components
):
    """- n_neighbors: UMAP Parameter
    - n_components: Number of dimensions to reduce to."""

    # create names for the uploaded files' potential resources
    if isinstance(min_dist, int):
        min_dist_str = str(min_dist)
    else:
        min_dist_str = str(min_dist).split(".")[1]

    # Create path name for resources
    resources_dir = get_resource_dir(debug, session_id, dataset_name)
    embeddings_pickle = f"{dataset_name}_umap_{n_components}D_embeddings_{n_neighbors}_{min_dist_str}.pickle"
    # create paths to these resource files
    path_to_embeddings_pickle = os.path.join(resources_dir, embeddings_pickle)
    # If embeddings for this dataset are available, use them, else save after generating
    if not os.path.exists(resources_dir):
        os.makedirs(resources_dir)
    resources = os.listdir(resources_dir)

    features = load_features_paths(session_id, dataset_name, redis_client)

    if embeddings_pickle in resources:
        with open(path_to_embeddings_pickle, "rb") as f:
            embeddings = pickle.load(f)
    else:
        print(
            f"[INFO][STARTED]: Dimensionality reduction... this could take a few minutes."
        )
        # print("features before umapping:", features)
        embeddings = UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric="correlation",
            random_state=24,
            transform_seed=24,
            verbose=True,
        ).fit_transform(features)
        pickle.dump(embeddings, open(path_to_embeddings_pickle, "wb"))
        print(f"[INFO][DONE]: Dimensionality reduction.")

    return embeddings, n_components, min_dist_str


def generate_clusters(
    redis_client: Redis,
    session_id,
    dataset_name,
    n_neighbors,
    min_dist,
    min_cluster_size,
    min_samples,
    n_components,
):
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
    embeddings, n_components, min_dist_str = setup_umap(
        redis_client, session_id, dataset_name, n_neighbors, min_dist, n_components
    )
    print(f"[INFO][STARTED]: Clustering with HDBSCAN.")
    labels = hdbscan.HDBSCAN(
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
    ).fit_predict(embeddings)
    print(f"[INFO][DONE]: Clustering with HDBSCAN.")
    # calculate percentage of the images that were clustered (i.e not labelled as noise)
    num_clustered = labels >= 0
    img_paths = get_img_paths(redis_client, session_id, dataset_name)
    percent_clustered = round(100 * np.sum(num_clustered) / len(img_paths), 2)

    resources_dir = get_resource_dir(debug, session_id, dataset_name)
    # create paths to these resource files
    labels_pickle = f"{dataset_name}_labels_{n_components}D_{n_neighbors}_{min_dist_str}_{min_cluster_size}_{min_samples}.pickle"
    path_to_labels_pickle = os.path.join(resources_dir, labels_pickle)
    pickle.dump(labels, open(path_to_labels_pickle, "wb"))

    return embeddings, labels, percent_clustered


def get_img_paths(redis_client: Redis, session_id, dataset_name):
    try:
        img_paths = orjson.loads(
            redis_client.get(f"{session_id}:{dataset_name}:img_paths")
        )
    except:
        # if using injected resources for debug, load from file and save into redis
        resources_dir = get_resource_dir(debug, session_id, dataset_name)
        with open(
            os.path.join(resources_dir, f"{dataset_name}_img_paths.pickle"), "rb"
        ) as f:
            img_paths = pickle.load(f)
        # replace the user_id from saved paths to that of current user
        old_id = img_paths[0].split("/")[1]
        img_paths = [path.replace(old_id, session_id) for path in img_paths]
        img_paths_json_bytes = orjson.dumps(img_paths)
        redis_client.set(f"{session_id}:{dataset_name}:img_paths", img_paths_json_bytes)
    return img_paths


def make_archive(source, destination):
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


def get_labels(
    session_id,
    dataset_name,
    n_neighbors,
    min_dist,
    min_cluster_size,
    min_samples,
    n_components,
):
    resources_dir = get_resource_dir(debug, session_id, dataset_name)
    # create paths to these resource files
    labels_pickle = f"{dataset_name}_labels_{n_components}D_{n_neighbors}_{min_dist}_{min_cluster_size}_{min_samples}.pickle"
    path_to_labels_pickle = os.path.join(resources_dir, labels_pickle)
    resources = os.listdir(resources_dir)
    if labels_pickle in resources:
        with open(path_to_labels_pickle, "rb") as f:
            labels = pickle.load(f)
            return labels
    else:
        print("[ERROR]: labels pickle not found.")
        return


def create_clusters_zip(
    redis_client: Redis,
    session_id,
    dataset_name,
    n_neighbors,
    min_dist,
    min_cluster_size,
    min_samples,
    n_components,
):
    img_paths = get_img_paths(redis_client, session_id, dataset_name)

    labels = get_labels(
        session_id,
        dataset_name,
        n_neighbors,
        min_dist,
        min_cluster_size,
        min_samples,
        n_components,
    )

    # Export clusters to folders
    imgs_with_clusters = [[labels[i], img_paths[i]] for i in range(len(labels))]

    df = pd.DataFrame(
        imgs_with_clusters,
        columns=["cluster", "imgPath"],
        index=range(len(labels)),
    )

    # First add the noise cluster to the list then the remaining clusters as lists
    clusters_dict = {}

    cluster_noise = df[df["cluster"] == -1]["imgPath"].tolist()

    # use set() to iterate over the number of unique classes minus the noise class
    if cluster_noise:
        num_clusters = range(len(set(labels)) - 1)
        clusters_dict["cluster_noise"] = cluster_noise
    else:
        num_clusters = range(len(set(labels)))

    for i in num_clusters:
        subFrame = df[df["cluster"] == i]
        clusters_dict["cluster_%s" % i] = subFrame["imgPath"].tolist()

    dataset_dir = f"assets/{session_id}/{dataset_name}"
    cluster_dir = os.path.join(dataset_dir, "clusters")

    # Remove pre-existing cluster directory
    if not os.path.exists(cluster_dir):
        os.makedirs(cluster_dir)
    else:
        shutil.rmtree(cluster_dir)  # Removes all cluster subdirectories!
        # If a zip download file already exists, delete it
        if os.path.exists(os.path.join(dataset_dir, "clusters.zip")):
            os.remove(os.path.join(dataset_dir, "clusters.zip"))
        os.makedirs(cluster_dir)

    for cluster in clusters_dict:
        # create a folder for the current cluster
        current_output_dir = os.path.join(cluster_dir, str(cluster))
        os.makedirs(current_output_dir)
        # copy images from original location to new folder
        for path in clusters_dict[str(cluster)]:
            shutil.copy(path, current_output_dir)
    # create the zip using our clusters folder
    make_archive(cluster_dir, cluster_dir + ".zip")


def gen_img_uri(
    redis_client: Redis, session_id, dataset_name, img_index, img_path=None
) -> str:
    """genImgURI Open image file at provided path with PIL and encode to
    img_uri string.

    Args:
        image_path (str): Path to image file.

    Returns:
        img_uri (str): str containing image bytes viewable by html.Img
    """

    if img_path:
        im = Image.open(img_path)
    else:
        img_paths = get_img_paths(redis_client, session_id, dataset_name)
        im = Image.open(img_paths[img_index])

    # dump it to base64
    buffer = io.BytesIO()
    try:
        im.save(buffer, format="jpeg")
    except:
        im = im.convert("RGB")
        im.save(buffer, format="jpeg")

    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url


def prepare_preview_download(
    redis_client: Redis, session_id, dataset_name, selected_img_idxs, filename
):
    dataset_dir = f"assets/{session_id}/{dataset_name}"
    preview_files_dir = os.path.join(dataset_dir, filename)
    preview_zip_path = os.path.join(dataset_dir, filename + ".zip")
    # remove previous files
    if os.path.exists(preview_files_dir):
        shutil.rmtree(preview_files_dir)
    if not os.path.exists(preview_files_dir):
        os.makedirs(preview_files_dir)
    if os.path.exists(preview_zip_path):
        os.remove(preview_zip_path)
    # create directory with the preview images copied from original dataset
    img_paths = get_img_paths(redis_client, session_id, dataset_name)
    for idx in selected_img_idxs:
        img_path = img_paths[idx]
        shutil.copy(img_path, preview_files_dir)
    make_archive(preview_files_dir, preview_files_dir + ".zip")


def del_folder_contents(folder: str) -> None:
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


def find_similar_imgs(session_id, dataset_name, test_image_path: str) -> list:
    transform = transforms.Compose(
        [
            transforms.Resize(size=[224, 224], interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

    # Create path name for resources
    resources_dir = get_resource_dir(debug, session_id, dataset_name)
    index_pickle = f"{dataset_name}_index.pickle"
    path_to_index_pickle = os.path.join(resources_dir, index_pickle)

    resources = os.listdir(resources_dir)
    if index_pickle in resources:
        with open(path_to_index_pickle, "rb") as f:
            index = pickle.load(f)

    with torch.no_grad():
        query_descriptors = pooling_output(input_tensor.to(DEVICE)).cpu().numpy()
        distance, indices = index.search(query_descriptors.reshape(1, 2048), 12)

    return indices
