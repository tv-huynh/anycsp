Install dependencies with conda:
```
conda create --name anycsp python=3.10
conda activate anycsp
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```
Change "+cu117" to "+cpu" if you have no gpu.

Our validation and test data can be found under the following link:

https://drive.google.com/file/d/1ZZndtrNJfiJRz18aSG_oxwIYzGyKNvrv/view?usp=sharing

Download the file and extract the data in the top-level directory of the repository:
```
unzip data.zip
```

We provide scripts for training and testing for each problem of our experiments.
For example, to train a model for MaxCut simply call:
```
bash scripts/train_maxcut.sh
```

To evaluate the models, use the corresponding evaluation scripts:
```
bash scripts/eval_maxcut.sh
```

### Note
If conda dependency installation as described above fails, the following alternative commands may help:

```
conda create --name anycsp python=3.10
conda activate anycsp
pip install torch==2.0.0+cu117 --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

Or, alternatively, only with pip:
```
python3.10 -m venv .venv
source .venv/bin/activate
pip install torch==2.0.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```
In this case, pip may need to downgrade numpy via
```
pip install "numpy<2"
```

## Installation on Capella (NVIDIA H100)

This setup uses Python 3.10.4, PyTorch 2.7.1 with CUDA 11.8,
and a dedicated virtual environment.

### 1. Load modules and create a virtual environment

Run all commands from the repository root directory.

If another virtual environment or Conda environment is already active,
deactivate it first using `deactivate` or `conda deactivate`, respectively.

```bash
module load release/24.04 GCCcore/11.3.0 Python/3.10.4

python -m venv .venv-capella
source .venv-capella/bin/activate

python -m pip install --upgrade pip
```

### 2. Install PyTorch

```bash
python -m pip install "torch==2.7.1" \
    --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
```

### 3. Install PyG extensions using prebuilt wheels

The extensions must match the installed PyTorch and CUDA build.
For PyTorch 2.7.1 with CUDA 11.8, use the `torch-2.7.0+cu118`
wheel directory.

```bash
python -m pip install --only-binary=:all: \
    torch-cluster torch-scatter \
    -f [https://data.pyg.org/whl/torch-2.7.0+cu118.html](https://data.pyg.org/whl/torch-2.7.0+cu118.html)
```

Install these packages before the remaining requirements to prevent pip
from attempting to build them from source.

### 4. Install the remaining project dependencies

```bash
python -m pip install -r requirements.txt \
    -f [https://data.pyg.org/whl/torch-2.7.0+cu118.html](https://data.pyg.org/whl/torch-2.7.0+cu118.html)

python -m pip check
```

**Important:** The `requirements.txt` file must not pin conflicting versions
of PyTorch or the PyG extensions.

### 5. Verify the installation

This test checks that the packages can be imported and does not require a GPU:

```bash
python - <<'PY'
import sys
import torch
import torch_cluster
import torch_scatter
import torch_geometric

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("torch-cluster:", torch_cluster.__version__)
print("torch-scatter:", torch_scatter.__version__)
print("torch-geometric:", torch_geometric.__version__)
PY
```

Expected PyTorch version: `2.7.1+cu118`.

### 6. Verify GPU functionality

Run the following test within a GPU allocation:

```bash
python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit(
        "No CUDA GPU available. Run this test within a GPU allocation."
    )

print("GPU:", torch.cuda.get_device_name(0))
print("CUDA build:", torch.version.cuda)
print("Supported architectures:", torch.cuda.get_arch_list())

x = torch.randn(1024, 1024, device="cuda")
y = x @ x.T
torch.cuda.synchronize()

print("GPU test successful:", y.shape, y.device)
PY
```

### Usage in subsequent sessions and Slurm jobs

The installation only needs to be performed once. In subsequent sessions
and in your job scripts, load the same modules and activate the environment:

```bash
module load release/24.04 GCCcore/11.3.0 Python/3.10.4

cd /path/to/anycsp
source .venv-capella/bin/activate

# Run the desired project command here.
```

Replace `/path/to/anycsp` with the actual path to the repository.