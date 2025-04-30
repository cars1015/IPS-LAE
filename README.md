# Beyond Power Laws: Propensity Score for Unbiased and Diverse Recommendations without Sacrificing Accuracy

## Dataset

- ML-20M
- Netflix-Prize
- MSD

The preprocessing and data splitting followed the procedure described at [Multi-VAE](https://github.com/dawenl/vae_cf).
The datasets are available at [this page]().

## Setup environment
### Install environment
```bash
conda env create -f environment.yml
```
### Activate
```bash
conda activate sigprop-rec
```

### Reproduce Evaluation Results in Paper
To reproduce the experiments on different datasets, use the following shell script:
```bash
./run.sh [dataset]
```
