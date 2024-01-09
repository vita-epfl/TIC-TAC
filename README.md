# TIC-TAC: A Framework To Learn And Evaluate Your Covariance

Code repository for "TIC-TAC: A Framework To Learn And Evaluate Your Covariance". We derive a new expression, the _Taylor Induced Covariance (TIC)_, and a new metric, the _Task Agnostic Correlations (TAC)_, for covariance estimation. TIC expresses the covariance through the estimator's gradient and curvature, resulting in better covariance predictions. TAC leverages the conditioning property of the normal distribution to evaluate the covariance quantitatively.


## Table of contents
1. [Installation: Docker (recommended) or PIP](#installation)
1. [Organization](#organization)
1. [Code Execution](#execution)
1. [Citation](#citation)


## Installation: Docker (recommended) or PIP <a name="installation"></a>

**Docker**: We provide a Docker image which is pre-installed with all required packages. We recommend using this image to ensure reproducibility of our results. Using this image requires setting up Docker on Ubuntu: [Docker](https://docs.docker.com/engine/install/ubuntu/#installation-methods). Once installed, we can use the provided `docker-compose.yaml` file to start our environment with the following command:  `docker-compose run --rm covariance` <br>

**PIP**: In case using Docker is not possible, we provide a `requirements.txt` file containing a list of all the packages which can be installed with `pip`.  We recommend setting up a new virtual environment ([link](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)) and install the packages using:  `pip install -r requirements.txt`


## Organization <a name="organization"></a>

The repository contains four main folders corresponding to the four experiments: `Univariate, Multivariate, UCI`and `HumanPose`. While the first three are self contained, HumanPose requires us to download images corresponding to the [MPII Dataset](https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz), [LSP Dataset](http://sam.johnson.io/research/lsp.html) and [LSPET Dataset](http://sam.johnson.io/research/lspet.html). For each of these datasets copy-paste all the images `*.jpg` into `HumanPose/data/{mpii OR lsp OR lspet}/images/`. Within `HumanPose`, we have a separate folder `cached`, which holds the generated file `mpii_cache_imgs_{True/False}.npy`. This file stores the post-processed MPII dataset to avoid redundancy every time the code is run.


```bash
.
├── docker-compose.yaml
├── HumanPose
│   ├── cached
│   │   └── Stacked_HG_ValidationImageNames.txt
│   ├── code
│   │   ├── config.py
│   │   ├── configuration.yml
│   │   ├── dataloader.py
│   │   ├── loss.py
│   │   ├── main.py
│   │   ├── models
│   │   │   ├── auxiliary
│   │   │   │   └── AuxiliaryNet.py
│   │   │   └── stacked_hourglass
│   │   │       ├── layers.py
│   │   │       └── StackedHourglass.py
│   │   └── utils
│   │       ├── kl_divergence.py
│   │       └── pose.py
│   ├── data
│   │   ├── lsp
│   │   │   ├── images
│   │   │   ├── joints.mat
│   │   │   ├── lsp_filenames.txt
│   │   │   └── README.txt
│   │   ├── lspet
│   │   │   ├── images
│   │   │   ├── joints.mat
│   │   │   ├── lspet_filenames.txt
│   │   │   └── README.txt
│   │   └── mpii
│   │       ├── images
│   │       ├── joints.mat
│   │       ├── mpii_filenames.txt
│   │       └── README.md
│   └── results
├── LICENSE
├── Multivariate
│   ├── loss.py
│   ├── multivariate.py
│   ├── regressor.py
│   ├── sampler.py
│   └── utils.py
├── README.md
├── requirements.txt
├── UCI
│   ├── loss.py
│   ├── regressor.py
│   ├── sampler.py
│   ├── UCI_Datasets
│   │   ├── Abalone
│   │   │   └── abalone.csv
│   │   ├── Air Quality
│   │   │   └── AirQualityUCI.csv
│   │   ├── Appliances
│   │   │   └── energydata_complete.csv
│   │   ├── Concrete
│   │   │   └── Concrete_Data.csv
│   │   ├── Condition Naval Propulsion
│   │   │   ├── data.txt
│   │   │   └── README.txt
│   │   ├── Electrical
│   │   │   └── ElectricalData.csv
│   │   ├── Energy
│   │   │   └── ENB2012_data.csv
│   │   ├── Gas Turbine
│   │   │   ├── gt_2011.csv
│   │   │   ├── gt_2012.csv
│   │   │   ├── gt_2013.csv
│   │   │   ├── gt_2014.csv
│   │   │   └── gt_2015.csv
│   │   ├── Parkinson
│   │   │   ├── test_data.csv
│   │   │   └── train_data.csv
│   │   ├── Power
│   │   │   └── power.csv
│   │   ├── Superconductivity
│   │   │   ├── train.csv
│   │   │   └── unique_m.csv
│   │   └── Wine Quality
│   │       ├── winequality-red.csv
│   │       └── winequality-white.csv
│   ├── UCI.py
│   └── utils.py
└── Univariate
    ├── loss.py
    ├── regressor.py
    ├── sampler.py
    ├── univariate.py
    └── utils.py
```

## Code Execution <a name="execution"></a>
We first need to activate the environment. This requires us to start the container: `docker-compose run --rm covariance`, which loads our image containing all the pre-installed packages. Alternatively, we can activate the virtual environment which contains packages installed via `pip`.

The main files to run the experiments are: `univariate.py, multivariate.py, uci.py` and `main.py` in the directories `Univariate, Multivariate, UCI` and `HumanPose` respectively. These experiments can be run using `python <filename>.py`. The configuration to run the Univariate, Multivariate and UCI are available in the main files respectively, whereas the configuration for human pose experiments are stored in `configuration.yml` in the `HumanPose` directory. The results for the univariate sinusoidal experiments are saved as PDF files for each epoch. The Multivariate results are saved as `TAC.pdf`. For UCI, the results are saved as `output.txt` for each dataset. Similarly, human pose results are saved in `output_*.txt`.

Stopping a container once the code execution is complete can be done using:
1. `docker ps`: List running containers
2. `docker stop <container id>`
