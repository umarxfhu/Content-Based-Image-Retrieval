# Dataset Clustering Dashboard

A Dockerized Python web-app for image dataset clustering and exploration. Built using Dash by Plotly and accelerated with Redis.

The `src` folder contains code for a Dash webapp that allows (multiple) user to upload a folder containing images that has been zipped (.zip) to generate 3D and 2D UMAP visualisations of the clusters formed by HDBSCAN from the images of your dataset.

## To run the dashboard on a server, Docker and docker-compose must be installed on the host (server) computer, then execute the following commands:
- Once you have cloned this repository, `cd` into it.
- Next run `docker-compose build` (add `sudo` to the beginning of this/any following Docker commands if needed).
- Finally run `docker-compose run api`
- The terminal will now display a URL where the dashboard can be accessed in a browser.
- If your connection fails in the browser check `config.py` and `docker-compose.yml` to ensure the ports being used are available on your computer.
- To stop the app press `ctrl+C` on your keyboard.
- To bring the container down run `docker-compose down`

## To run without docker, create a new virtual environment and update pip.
  1. `python3 -m dashEnv ~/venv/testenv` (where `dashEnv` is the name you choose to give the virtual environment).
  2. `source ~/venv/dashEnv/bin/activate`
  3. `pip install -U pip`
- Next install dependencies from the provided requirements.txt file:
  1. Then from within the cloned repo install required packages: `pip install -r requirements.txt`
- Open a new terminal tab (session) and start the Redis server by running: `redis-server` 
  1. (Here you may need to add `--port 6380` (<--example port number) to the end of this command, and edit `config.py` if the default redis port `6379` is in use on your computer.
- Switch back to the previous terminal tab and start the dashboard:
  1. `cd src`
  2. `python app.py`
- The command line will display the URL with port that you can visit the dashboard at from a local browser.
- If your connection fails in the browser check `config.py` to ensure the ports used are available on your computer.

## What should I upload?
A folder of images (I tested datasets with ~150 to ~20k images within sizes of ~1 Mb to ~12 Gb)
### Expected directory structure for your uploaded `yourDatasetsName.zip` file when extracted:
```
yourDatasetsName
└───image_1.jpg
└───image_2.jpg
└───.
└───.
└───image_N.jpg
```
  - Image filenames are arbitrary but should be unique (do not include any subfolders), and could be of the following formats `.jpg`, `.jpeg`, `.png`, `.bmp`.
### Oh no! The second/third etc. dataset I upload sometimes fails!?
- The upload component used is from https://github.com/np-8/dash-uploader and sometimes it fails with larger uploads.
- If this happens you could try to reupload which works sometimes, if not close the current window you are using the dashboard in and reopen it to start a new session (a better fix is in progress).

[Deprecated] The CBIR notebook (backend algo testing) allows you to:
- Download a kaggle dataset (e.g. caltech256).
- Extract image features or use saved ones.
- Look up similar images in your dataset using FAISS to perform similarity matching on the features. 
- Average Precision calculation over the dataset, graphed for X number of similar images to be returned, where x = [2,4,6,8] against the corresponding precision.
- ![image](https://user-images.githubusercontent.com/97547817/151413730-b873ef99-29df-4789-a778-0ca013529d90.png)
- PCA feature length reduction and clustering visualisation using t-sne. 
- ![image](https://user-images.githubusercontent.com/97547817/151413781-5964fb5e-2d54-4062-832e-a160786cdd20.png)

