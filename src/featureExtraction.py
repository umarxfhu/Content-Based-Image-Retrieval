"""
# Extract features from specific class in imageset
To prepare our data, we'll be following what is loosely known as an `ETL` process.

- `E`xtract data from a data source.
- `T`ransform data into a desirable format. (Put it into `tensor` form)
- `L`oad data into a suitable structure. (Put data into an `object` to 
    make it easily accessible.)
"""
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets


def extract_features_paths(data_directory, session_id):
    class ImageFolderWithPaths(datasets.ImageFolder):
        """Custom dataset that includes image file paths. Extends
        torchvision.datasets.ImageFolder
        Source: https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
        """

        # override the __getitem__ method. this is the method that dataloader calls
        def __getitem__(self, index):
            # this is what ImageFolder normally returns
            original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
            # the image file path
            path = self.imgs[index][0]
            # make a new tuple that includes original and the path
            tuple_with_path = original_tuple + (path,)
            return tuple_with_path

    def pooling_output(x, model):
        for layer_name, layer in model._modules.items():
            x = layer(x)
            if layer_name == "avgpool":
                break
        return x

    transform = transforms.Compose(
        [
            transforms.Resize(size=[224, 224], interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = ImageFolderWithPaths(
        data_directory, transform=transform
    )  # our custom dataset
    # strip away unnecessary info and store path to each image
    img_paths = [dataset.imgs[i][0] for i in range(len(dataset.imgs))]
    # initialize the dataloaders
    dataloader = DataLoader(dataset)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] Using Device:", DEVICE)
    model = models.resnet101(pretrained=True, progress=True)
    print(f"[INFO] Loading model: {model.__class__.__name__}")

    features = []
    print(f"[INFO][STARTED] Feature Extraction using {model.__class__.__name__}")
    model.to(DEVICE)
    with torch.no_grad():
        model.eval()
        # save current system exceptions file that tqdm writes to
        std_err_backup = sys.stderr
        file_prog = open(f"assets/{session_id}/progress.txt", "w")
        sys.stderr = file_prog
        # start writing to tqdm progress file
        for inputs, labels, paths in tqdm(
            dataloader, bar_format="{l_bar}{bar:3}{r_bar}{bar:-3b}"
        ):
            result = pooling_output(inputs.to(DEVICE), model)
            features.append(result.cpu().view(1, -1).numpy())
            torch.cuda.empty_cache()
        # close progress file
        file_prog.close()
        sys.stderr = std_err_backup

    print(f"[INFO][DONE] Feature Extraction using {model.__class__.__name__}")

    features = np.vstack(features)
    print("[DEBUG]: Preview Features:", features[0])
    print("[DEBUG]: Preview Image Paths:", img_paths[0])
    return features, img_paths
