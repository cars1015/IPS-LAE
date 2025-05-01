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
To reproduce the experimental results shown in Table 1, use the following shell script:
```bash
./run.sh ml-20m
./run.sh netflix-prize
./run.sh msd
```

## Optional: Custom Runs

```bash
python3 main.py \
  --dataset ml-20m \
  --model rdlae \
  --lambda 1000 \
  --drop_p 0.3 \
  --alpha 0.1 \
  --wflg \
  --wtype powerlaw \
  --wbeta 0.3
```
 
 ### Customization Notes

You may freely adjust the following options when running `main.py` manually:

- **Models**:
  - `ease`
  - `edlae`
  - `rdlae`

- **Datasets**:
  - `ml-20m`
  - `netflix-prize`
  - `msd`

- **Weighting**:
  - Add `--wflg` to enable weighting
  - Choose type via `--wtype logsigmoid` or `--wtype powerlaw`
  - Omit `--wflg` for no weighting

- **Hyperparameters**:
  - `--lambda`: regularization strength
  - `--drop_p`: dropout rate (for EDLAE/RDLAE)
  - `--alpha`: RDLAE constraint threshold
  - `--wbeta`: controls the strength of bias correction (used in both `logsigmoid` and `powerlaw`)

This setup allows flexible experimentation with different bias correction strategies and model architectures.