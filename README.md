# 	[ACM CCS 2023] Stateful Defenses for Machine Learning Models Are Not Yet Secure Against Black-box Attacks

This is an implemenation of the OARS attack framework described in the ACM CCS 2023 paper:
[Stateful Defenses for Machine Learning Models Are Not Yet Secure Against Black-box Attacks](https://arxiv.org/abs/2303.06280).

## 1. Environment

I'd recommend having something that resembles the environment described below to make things easy:

- **OS:** Ubuntu 20.04.6 LTS
- **CPU:** Intel(R) Core(TM) i9-10920X CPU @ 3.50GHz 
- **GPU:** NVIDIA GeForce RTX 2080 Ti Rev. A w/ 11 GB VRAM
- **Conda:** 4.9.2

Set up your environment using the conda environment file `environment.yml` as follows:

```conda env create -f environment.yml```

After you're all set up, go ahead and activate the `oars` environment to run things:

```conda activate oars```

## 2. Code

### 2.1. Usage
You'll want to run things by invoking `main.py` with a config file as follows:

`python main.py --config [path to config file] --start_idx [start index] --num_images [number of images] --disable_logging`

Arguments are hopefully self-explanatory:

```
--disable_logging (bool): If present, disable logging to a results file. Pretty useful for debugging to avoid clutter.
--config (str): Path to a config file that specifies parameters for the experiment (dataset, attack to be run, model, etc.)
--start_idx (int): Index of the first image in the dataset to be attacked (useful when parallelizing experiments)
--num_images (int): Number of images to attack (again, useful when parallelizing experiments)
```

### 2.2. Organization

To get an idea of how things are organized, here's a brief overview of the codebase:

#### 2.2.1. Entrypoint

Your entry-point is going to be `main.py`, which takes a few command-line arguments as described above. At a high-level, `main.py` loads a dataset, model (wrapped by a stateful defense), and then runs an attack.

#### 2.2.2. Models
The `models` directory contains the code for the classifiers, as well as the stateful defenses. In general they're 
implemented as wrappers around the classifiers. The `models/pretrained` subdirectory here contains relevant checkpoints.

#### 2.2.3. Attacks
Attacks are implemented within the `attacks/adaptive` directory, as extensions of the `Attack.py` abstract class. The `attacks.py` file provides for an attack
loader routine that loads the attacks and runs them over a dataset. Indeed, it is this attack loader that is called by
`main.py`. 

#### 2.2.4. Data
You're going to want to edit `utils/datasets.py` to point to your dataset folder. To organize your data, structure your dataset folder as follows (using cifar10 as an example):

```
cifar10/
    - imgs/ 
        - 0.png
        - 1.png
        - 2.png
        ...
    - cifar10.json 
    - cifar10_targeted.json 
```
where `cifar10.json` is a json file that maps images to their labels, e.g., 

```
{
    "imgs/0.png": 3,
    "imgs/1.png": 8,
    "imgs/2.png": 8,
    ...
}
```
and where `cifar10_targeted.json` is a json file for targeted attacks, that maps target labels to an initialization image for that label, e.g.,

```
{
  "3": [
    "imgs/0.png"
  ],
  "8": [
    "imgs/1.png"
  ],
  ...
}
```

#### 2.2.5. Configs and Results
Configuration files in subdirectories of `configs` are just `.json` files that specify the parameters for an experiment. A config file specifies all the information
for the dataset, model, attack parameters, etc. When you run an experiment the results should be saved in the directory
that contains the config file that you used (unless you used `--disable_logging`). You can compute metrics on the results folder using `analysis.py` as follows:

`python analysis.py --log_path [path to results directory]`


# 3. Responsible Use, License, and Citation
This repository contains attack code. In general, please be responsible while executing this code and do not
use it on classification services for which you do not have permission. It is meant to be used for research purposes only. 
This code is licensed under the MIT license:

> MIT License
> 
> Copyright (c) 2023 Authors of "Stateful Defenses for Machine..."
> 
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
> 
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
> 
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.


If you use this code in your research, please cite the following paper:

```
@article{feng2023investigating,
  title={Stateful Defenses for Machine Learning Models Are Not Yet Secure Against Black-box Attacks},
  author={Feng (co-lead), Ryan and Hooda (co-lead), Ashish and Mangaokar (co-lead), Neal and Fawaz, Kassem and Jha, Somesh and Prakash, Atul},
  journal={Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications Security},
  year={2023}
}