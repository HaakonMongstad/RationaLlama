# dsl-final-project

# GPU Installation

1. Create a conda environment and install the required dependencies:
```bash
conda env create -f gpu_environment.yaml
```

2. Activate the environment:
```bash
conda activate dsl-final-project
```

3. Install bitsandbytes package from source to enable quantization:
```bash
git clone https://github.com/TimDettmers/bitsandbytes.git && cd bitsandbytes/
pip install -r requirements-dev.txt
cmake -DCOMPUTE_BACKEND=cuda -S .
make
pip install .
```

4. Downloading dataset and preprocessing
## PUT KAGGLE DATASET DOWNLOAD HERE

```bash
python data_preprocess.py <source_file_path> <output_file_path>
```