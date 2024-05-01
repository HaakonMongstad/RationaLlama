# dsl-final-project

## GPU Installation

1. Create a conda environment and install the required dependencies:
```bash
conda env create -f environment.yaml
```

2. Activate the environment:
```bash
conda activate rationallama
```

3. Install bitsandbytes package from source to enable quantization:
```bash
git clone https://github.com/TimDettmers/bitsandbytes.git && cd bitsandbytes/
pip install -r requirements-dev.txt
cmake -DCOMPUTE_BACKEND=cuda -S .
make
pip install .
```

## Dataset Download

1. Download your kaggle API token from: https://www.kaggle.com/settings/account

2. Move kaggle.json to ```~/.kaggle/kaggle.json```

3. Change the file permissions for security:
```bash
chmod 600 ~/.kaggle/kaggle.json
``` 

4. Run the downloader script
```bash
bash download_dataset.sh
```