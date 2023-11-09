# PyTorch BYOL for HPC Clusters with Slurm

## Overview

This project provides our PyTorch reimplementation of BYOL. It is specifically designed to leverage high-performance computing (HPC) clusters using Slurm workload manager, and it comes ready for use on the Jean Zay cluster. It includes full training and linear evaluation pipelines, with support for CIFAR10 and ImageNet datasets.

For an in-depth analysis of the reimplementation process and results, please refer to the replication report provided in `report.pdf`.

## Results

Our efforts have led to the successful replication of the BYOL methodology. We compared our results with those reported in an unofficial GitHub implementation for CIFAR10 and with the original paper for ImageNet. The comparison reveals that our implementation is closely aligned with these benchmarks, with only minor deviations that can be attributed to the natural variability of such experiments (our results were achieved in a single run constrained by computational resources).

For deeper insights, we have provided the weights of our trained models in the `checkpoints` folder.

### CIFAR10
| Implementation                  | Top-1 Accuracy | Top-5 Accuracy |
|---------------------------------|----------------|----------------|
| Unofficial Implementation       | 91.1%          | 99.8%          |
| Our Replication                 | 91.92%         | 99.69%         |

### ImageNet
| Implementation                  | Top-1 Accuracy | Top-5 Accuracy |
|---------------------------------|----------------|----------------|
| Original                        | 74.3%          | 91.6%          |
| Our Replication                 | 74.03%         | 91.51%         |

## Requirements

Before running the code, please install the necessary packages by running:

```bash
pip install -r requirements.txt
```

## Usage

The project is tailored for the Jean Zay cluster with ready-to-use Slurm scripts. Non-Jean Zay Slurm users can adjust the scripts to match their environment's configurations by modifying partition names, module loads, and other SBATCH directives.

For those not on Slurm or on systems with different specifications, the Python commands can be run directly from the command line. Note that due to the intensive computational requirements of these experimentations, you may need to alter the hyperparameters like batch size to accommodate to your system's capabilities, wich may lower performances.

### Arguments

The Python scripts (`main.py` and `eval.py`) accept several command-line arguments to tailor the execution to your hardware and experimental setup. Below is an explanation of some key arguments:

- --computer: Specifies the computing environment ('jeanzay' or 'other').
- --hardware: Defines the hardware setup ('cpu', 'mono-gpu', 'multi-gpu').
- --precision: Sets the numerical precision ('mixed' or 'normal').
- --nb_workers: The number of worker threads for data loading.
- --expe_name: A name for the experiment, used in saving models.
- --dataset_name: The dataset to use ('cifar10' or 'imagenet').
- --resnet_type: The type of ResNet model ('resnet18', 'resnet50', etc.).
- --nb_epochs: The total number of epochs to train for.

For a comprehensive list of arguments and their descriptions, consult the source code.

### Training

#### CIFAR10

```bash
python src/main.py --computer=other --hardware=mono-gpu --precision=mixed --nb_workers=10 --expe_name=byol_cifar10 --dataset_name=cifar10 --resnet_type=resnet18 --nb_epochs=800 --nb_epochs_warmup=10 --batch_size=512 --lr_init=2.0 --momentum=0.9 --weight_decay=1e-6 --eta=1e-3 --z_dim=256 --tau_base=0.996 --clsf_every=100 --save_every=100 --nb_epochs_clsf=90 --batch_size_clsf=256 --lr_init_clsf=0.2 --momentum_clsf=0.9 --weight_decay_clsf=0.0
```

#### ImageNet

```bash
python src/main.py --computer=other --hardware=mono-gpu --precision=mixed --nb_workers=10 --expe_name=byol_imagenet --dataset_name=imagenet --resnet_type=resnet50 --nb_epochs=1000 --nb_epochs_warmup=10 --batch_size=4096 --lr_init=3.2 --momentum=0.9 --weight_decay=1.5e-6 --eta=1e-3 --z_dim=256 --tau_base=0.996 --clsf_every=100 --save_every=100 --nb_epochs_clsf=90 --batch_size_clsf=256 --lr_init_clsf=0.2 --momentum_clsf=0.9 --weight_decay_clsf=0.0
```

### Evaluation

#### CIFAR10

```bash
python src/eval.py --computer=other --hardware=mono-gpu --precision=mixed --nb_workers=10 --dataset_name=cifar10 --resnet_type=resnet18 --z_dim=256 --nb_epochs_clsf=90 --batch_size_clsf=256 --lr_init_clsf=0.2 --momentum_clsf=0.9 --weight_decay_clsf=0.0 --checkpoint=./checkpoints/weights_byol_cifar10.pt
```

#### ImageNet

```bash
python src/eval.py --computer=other --hardware=mono-gpu --precision=mixed --nb_workers=10 --dataset_name=imagenet --resnet_type=resnet50 --z_dim=256 --nb_epochs_clsf=90 --batch_size_clsf=256 --lr_init_clsf=0.2 --momentum_clsf=0.9 --weight_decay_clsf=0.0 --checkpoint=./checkpoints/weights_byol_imagenet.pt
```
