# in-context-mt-analysis

The official code repository associated with the paper titled "[An Empirical Study of In-context Learning in LLMs for Machine Translation](https://arxiv.org/abs/2401.12097)" (ACL 2024 Findings). In this work, we comprehensively studies the in-context machine translation capabilities of LLMs.

## Overview

> Recent interest has surged in employing Large Language Models (LLMs) for machine translation (MT) via in-context learning (ICL) (Vilar et al., 2023). Most prior studies primarily focus on optimizing translation quality, with limited attention to understanding the specific aspects of ICL that influence the said quality. To this end, we perform the first of its kind, an exhaustive study of in-context learning for machine translation. We first establish that ICL is primarily example-driven and not instruction-driven. Following this, we conduct an extensive exploration of various aspects of the examples to understand their influence on downstream performance. Our analysis includes factors such as quality and quantity of demonstrations, spatial proximity, and source versus target originality. Further, we also investigate challenging scenarios involving indirectness and misalignment of examples to understand the limits of ICL. While we establish the significance of the quality of the target distribution over the source distribution of demonstrations, we further observe that perturbations sometimes act as regularizers, resulting in performance improvements. Surprisingly, ICL does not necessitate examples from the same task, and a related task with the same target distribution proves sufficient. We hope that our study acts as a guiding resource for considerations in utilizing ICL for MT.

## Installation

```bash
# Clone the github repository and navigate to the project directory.
git clone https://github.com/PranjalChitale/in-context-mt-analysis.git
cd in-context-mt-analysis

# Create a virtual environment using conda
conda create -n icl_mt python=3.10 -y
conda activate icl_mt

# Install all the dependencies and requirements associated with the project.
conda install pip -y
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Experiments

TBD

## License

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

## Citation

```bibtex
@inproceedings{chitale-etal-2024-empirical,
  title = {An Empirical Study of In-context Learning in LLMs for Machine Translation},
  author = {Pranjal A. Chitale and Jay Gala and Raj Dabre},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics},
  year = {2024},
  publisher = {Association for Computational Linguistics},
  url = {https://arxiv.org/abs/2401.12097}
}
```
