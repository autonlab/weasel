<p>
    <img align="left" width="110" height="120" src="weasel.jpg">
</p>


<div align="center">

# WeaSEL: Weakly Supervised End-to-end Learning


<a href="https://pytorch.org/get-started/locally/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.7--3.9-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.7+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra-89b8cd?style=for-the-badge&labelColor=gray"></a>
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This is a PyTorch-Lightning-based framework, based on our [*End-to-End Weak Supervision* paper](https://arxiv.org/abs/2107.02233),
that allows you to **train your favorite neural network for weakly-supervised classification**<sup>1</sup>
</div>

- only with multiple labeling functions (LFs)<sup>2</sup>, i.e. *without any labeled training data*!
- in an *end-to-end* manner, i.e. directly train and evaluate your neural net (*end-model* from here on),
 there's no need to train a separate label model any more as in Snorkel & co,
- with *better test set performance and enhanced robustness* against correlated or inaccurate LFs than 
prior methods like Snorkel

<sup>1</sup> This includes learning from *crowdsourced labels* or annotations!
<br>
<sup>2</sup> LFs are labeling heuristics, that output noisy labels for (subsets of) the training data
 (e.g. crowdworkers or keyword detectors).

    
# Getting Started
This library assumes familiarity with (multi-source) weak supervision, if that's not the case you may want to first
learn its basics in e.g. [this overview slides from Stanford](http://cs229.stanford.edu/notes2019fall/weak_supervision_slides.pdf) or [this Snorkel tutorial](https://www.snorkel.org/use-cases/01-spam-tutorial).

That being said, ***have a look at our [examples](examples/) and the notebooks therein*** showing you how to use Weasel for your
own dataset, LF set, or end-model. E.g.:

- [***A high-level starter tutorial***](examples/1_bias_bios.ipynb), with few code, many explanations
 and including Snorkel as a baseline (so that if you are familiar with Snorkel you can see the similarities and differences to Weasel).

- [***See how the whole WeaSEL pipeline works with all details, necessary steps and definitions for a new dataset & custom end-model***](examples/0_full_pipeline.ipynb). 
    This notebook will probably make you learn the most about WeaSEL and *how to apply it to your own problem*.
    
- [***A realistic ML experiment script***](examples/1_bias_bios_full.py) with all that's part of a ML pipeline, including logging to Weight&Biases,
arbitrary callbacks, and eventually retrieving your fully trained end-model.

### Reproducibility
Please have a look at the research code branch, which operates on pure PyTorch.


# Installation

<details>
<summary><b>1. New environment </b>(recommended, but optional)</summary>

    conda create --name weasel python=3.7  # or other python version >=3.7
    conda activate weasel  
</details>

<details>
<summary><b> 2a: From source</b></summary>

    python -m pip install git+https://github.com/salvaRC/weasel#egg=weasel[all]

</details>

<details>
<summary><b> 2b: From source, <a href="https://huggingface.co/transformers/installation.html#editable-install">editable install</a></b></summary>

    git clone https://github.com/salvaRC/weasel.git
    cd weasel
    pip install -e .[all]

</details>

    
<details><p>
<summary><b>Minimal dependencies</b></summary>

Minimal dependencies, in particular not using [Hydra](https://hydra.cc), can be installed
with

    python -m pip install git+https://github.com/salvaRC/weasel

The needed environment corresponds to ``conda env create -f env_gpu_minimal.yml``.

*If you choose to use this variant, you won't be able to run some of the examples: You may want to have a look
at [this notebook](examples/1_bias_bios_no_hydra.ipynb) that walks you through how to use Weasel without Hydra as the config manager.*

</p></details>

**Note:** Weasel is under active development, some uncovered edge cases might exist, and any feedback is very welcomed! 

# Apply WeaSEL to your own problem 
### Configuration with Hydra
*Optional:* [This template config](configs/template.yaml) will help you get started with your own application,
[an analogous config](examples/configs/profTeacher_full.yaml) is used in [this tutorial script](examples/1_bias_bios_full.py)
 that you may want to check out. 

### Pre-defined or custom downstream models & Baselines

Please have a look at the detailed instructions in [this Readme](weasel/models/downstream_models/README.md).

### Using your own dataset and/or labeling heuristics
Please have a look at the detailed instructions in [this Readme](weasel/datamodules/README.md).

# Citation & Credits
<details>
    <summary><b> Citation</b></summary>
    The <a href="https://arxiv.org/abs/2107.02233">paper</a> is currently under review, its preprint reference is:
    
    @article{cachay2021endtoend,
      title={End-to-End Weak Supervision},
      author={Salva Rühling Cachay and Benedikt Boecking and Artur Dubrawski},
      year={2021},
      eprint={2107.02233},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
    }

</details>

<details><p>
    <summary><b> Credits</b></summary>

- The following template was extremely useful as source of inspiration and for getting started with the PL+Hydra implementation:
[ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)

- [Weasel image](weasel.jpg) credits go to [Rohan Chang for this](https://unsplash.com/photos/hn0AtxarNNw) Unsplash-licensed image

</p></details>

