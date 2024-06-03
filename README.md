![TIC-TAC](https://github.com/vita-epfl/TIC-TAC/blob/main/TIC-TAC_gif.gif)


# TIC-TAC: A Framework For Improved Covariance Estimation In Deep Heteroscedastic Regression

<a href="https://arxiv.org/abs/2310.18953"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2310.18953-%23B31B1B?logo=arxiv&logoColor=white"></a>
<a href="https://www.epfl.ch/labs/vita/heteroscedastic-regression/"><img alt="Project" src="https://img.shields.io/badge/-Project%20Page-lightgrey?logo=Google%20Chrome&color=informational&logoColor=white"></a>
<a href="https://openreview.net/forum?id=zdNTiTs5gU"><img alt="OpenReview" src="https://img.shields.io/badge/ICML%202024-OpenReview-%236DA252"></a>

<br>

Code repository for "TIC-TAC: A Framework For Improved Covariance Estimation In Deep Heteroscedastic Regression". We address the problem of sub-optimal covariance estimation in deep heteroscedastic regression by proposing a new parameterisation (TIC) and metric (TAC). We derive a new expression, the _Taylor Induced Covariance (TIC)_, which expresses the randomness of the prediction through its gradient and curvature. The _Task Agnostic Correlations (TAC)_ metric leverages the conditioning property of the normal distribution to evaluate the covariance quantitatively.


## Table of contents
1. [Installation: Docker (recommended) or PIP](#installation)
2. [Organization](#organization)
3. [Code Execution](#execution)
4. [Acknowledgement](#acknowledgement)
5. [Citation](#citation)


## Installation: Docker (recommended) or PIP <a name="installation"></a>

**Docker <a href="https://hub.docker.com/repository/docker/meghshukla/tictac/"><img alt="Docker" src="https://img.shields.io/badge/Image-tictac-%232496ED?logo=docker&logoColor=white"></a>**: We provide a Docker image which is pre-installed with all required packages. We recommend using this image to ensure reproducibility of our results. Using this image requires setting up Docker on Ubuntu: [Docker](https://docs.docker.com/engine/install/ubuntu/#installation-methods). Once installed, we can use the provided `docker-compose.yaml` file to start our environment with the following command:  `docker-compose run --rm tictac` <br>

**PIP**: In case using Docker is not possible, we provide a `requirements.txt` file containing a list of all the packages which can be installed with `pip`.  We recommend setting up a new virtual environment ([link](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)) and install the packages using:  `pip install -r requirements.txt`


## Organization <a name="organization"></a>

The repository contains four main folders corresponding to the four experiments: `Univariate, Multivariate, UCI`and `HumanPose`. While the first three are self contained, HumanPose requires us to download images corresponding to the [MPII Dataset](https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz), [LSP Dataset](http://sam.johnson.io/research/lsp.html) and [LSPET Dataset](http://sam.johnson.io/research/lspet.html). For each of these datasets copy-paste all the images `*.jpg` into `HumanPose/data/{mpii OR lsp OR lspet}/images/`. Within `HumanPose`, we have a separate folder `cached`, which holds the generated file `mpii_cache_imgs_{True/False}.npy`. This file stores the post-processed MPII dataset to avoid redundancy every time the code is run.
Running `python main.py` in the `code` folder executes the code, with configurations specified in `configuration.yml`


## Code Execution <a name="execution"></a>
We first need to activate the environment. This requires us to start the container: `docker-compose run --rm tictac`, which loads our image containing all the pre-installed packages. Alternatively, we can activate the virtual environment which contains packages installed via `pip`.

The main files to run the experiments are: `univariate.py, multivariate.py, uci.py` and `main.py` in the directories `Univariate, Multivariate, UCI` and `HumanPose` respectively. These experiments can be run using `python <filename>.py`. The configuration to run the Univariate, Multivariate and UCI are available in the main files respectively, whereas the configuration for human pose experiments are stored in `configuration.yml` in the `HumanPose` directory. The results for the univariate sinusoidal experiments are saved as PDF files for each epoch. The Multivariate results are saved as `TAC.pdf`. For UCI, the results are saved as `output.txt` for each dataset. Similarly, human pose results are saved in `output_*.txt`.

Stopping a container once the code execution is complete can be done using:
1. `docker ps`: List running containers
2. `docker stop <container id>`

## Acknowledgement <a name="acknowledgement"></a>

We thank `https://github.com/jaehyunnn/ViTPose_pytorch` for their implementation of ViTPose which was easily customizable.
We also borrow [code](https://github.com/meghshukla/ActiveLearningForHumanPose) from the Active Learning for Human Pose library.

## Citation <a name="citation"></a>

If you find this work useful, please consider starring this repository and citing this work!

```
@InProceedings{shukla2024tictac,
  title = {TIC-TAC: A Framework for Improved Covariance Estimation in Deep Heteroscedastic Regression},
  author = {Shukla, Megh and Salzmann, Mathieu and Alahi, Alexandre},
  booktitle = {Proceedings of the 41th International Conference on Machine Learning},
  year = {2024},
  series = {Proceedings of Machine Learning Research},
  month = {21--27 Jul},
  publisher = {PMLR}
}
```