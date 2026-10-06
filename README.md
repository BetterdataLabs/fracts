# FracTS Model for Times Series Generation

This repository contains the official implementation of the paper
**"FracTS: Hierarchical and Autoregressive Time Series Generation"**, accepted at **NeurIPS 2026**.

FracTS model is a time series generative model using fractal generative model.

# Quick Start

To train and generate time series data using FracTS, one can run the following commands:

---

## 1) Unconditional: train **and** generate

Use this when there are **no static conditions**.

```bash
python main.py -t -g -o OUT_DIR -d DATA_DIR -c CONFIG_FILE
```

---

## 2) Conditional with provided conditions

Use this when you **already have a condition file**.
**Important:** the model should be trained with **`rtf-r`**, otherwise the run will fail.

```bash
python main.py -t -g -o OUT_DIR -d DATA_DIR -s STATIC_DATA_PATH -i STATIC_ID -c CONFIG_FILE
```

* The run will read conditions from `STATIC_DATA_PATH` using the column `STATIC_ID`.

---

## 3) Conditional where the model must also generate static conditions

In this scenario you should **train and generate separately**:

**Train:**

```bash
python main.py -t -o OUT_DIR -d DATA_DIR -s STATIC_DATA_PATH -i STATIC_ID -c CONFIG_FILE
```

**Generate:**

```bash
python main.py -g -o OUT_DIR -c CONFIG_FILE
```

---

where all time series should be saved under `DATA_DIR` where each series is saved in a `.csv` file named after its 
corresponding static ID, and the static data is in a `.csv` file `STATIC_DATA_PATH` where the column `STATIC_ID` 
refers to the static ID (and hence time series file names). If there is no static conditions, `-s` and `-i` can be left
empty.

One can also run training and generation separately by removing `-t` for skipping training and removing 
`-g` for skipping generation.

here’s a minimal example of how your **`DATA_DIR`** should be organized:

```
DATA_DIR/
├── 101.csv
├── 102.csv
├── 103.csv
└── ...
```

* Each file (`101.csv`, `102.csv`, …) contains the **time series** for a single entity.
* The file name (e.g., `101`) must match the `STATIC_ID` in your static data file (if you are using conditional mode).


## Dependencies and Trouble Shooting

Dependencies are provided in `requirements.txt`. The python version should be `>= 3.9`.

Note that our model uses `os.symlink` to avoid too intensive disk IO of identical files. Some computers and OS may not
have permission to do this operation by default. A solution that we have tried and worked is to make sure that the 
computer is on Developer Mode.

When out-of-memory from PyTorch is raised, besides using a smaller model or decreasing batch size, one may also choose
a smaller `max_batch_size` for controlling the batch sizes at lower levels.

## Reproducibility

Scripts running experiments on datasets provided in [`dataset`](../dataset) directory can be found in `Makefile`.

Samples of configurations of our experiments (including ablation) can be found in `config` directory.

## Task Scope

In this model, we focus on time series generation task, which is different from forecasting task. To be clear, this 
model does not aim to predict future behaviors of a time series, but to generate synthetic time series data in the
same time frame as training data. The following is a functionality comparison of FracTS model and some existing
time series generative models. Key challenges in time series generations listed below are:

- **Long TS?**: Whether the model is capable of handling long time series. By long, we mean time series up to a few 
  thousands or tens of thousands time steps. We consider a model being able to handle long time series not by the 
  model's default parameters settings or theoretical applicability, but the usability of the model under the limit of
  typical computation machines. 
- **Static Conditions?**: Whether the model can generate time-series with static conditions. Typical static conditions
  can be the class label of time series. However, here we talk about a broader scenario where there can be multiple 
  columns in static conditions. Such scenario is common, such as the customer profiles of the transactions of customers,
  or the metadata of the company in stocks.
- **Multivariate?**: Whether the model can generate multivariate time series with correlation between different 
  variables taken into consideration.
- **Event-based?**: Whether the model is capable of generating time series where the times are not a sequence with a
  fixed frequency (e.g., every day, every hour, etc.), but any time. We regard such time series as event-based. Typical
  examples include the transaction history.
- **Varying Lengths?**: Whether the model can generate time series of varying lengths, which is typically seen in 
  event-based data and some data with different fixed time frames (such as stocks, as each company may have the first 
  record on different dates).
- **Mixed Data Types?**: Whether the model is capable of handling both categorical and numeric data.

| Model         | Long TS?    | Static Conditions? | Multivariate? | Event-based? | Varying Lengths? | Mixed Data Types? |
|---------------|-------------|--------------------|---------------|--------------|------------------|-------------------|
| FracTS (ours) | No limit[*] | Yes                | Yes           | Yes          | Yes              | Yes               |
| AEC-GAN       | No limit[*] | No                 | Yes           | No           | No               | No                |
| Cosci-GAN     | No limit[*] | No                 | Yes           | No           | No               | No                |
| Diffusion-TS  | No limit[*] | No                 | Yes           | No           | No               | No                |
| KOVAE         | No limit[*] | No                 | Yes           | Yes          | No               | No                |
| TimeVQVAE     | No limit[*] | Yes (class label)  | Yes           | No           | No               | No                |

[*] By "no limit", we mean we have not encountered a dataset that exceeds the capability of the model in our 
experiments.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{
li2026fracts,
title={Frac{TS}: Hierarchical and Autoregressive Time Series Generation},
author={Li, Jiayu and Afzal, Umair and Zhao, Zilong and Abdollahzadeh, Milad and Javaid, Uzair and Sikdar, Biplab},
booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
year={2026},
url={https://openreview.net/forum?id=ptzy0mmiMh}
}
```

## License

This code is released for non-commercial research and academic purposes only. See [`LICENSE`](LICENSE) for details.
