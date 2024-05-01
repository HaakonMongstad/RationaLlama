# RationaLlama 
<img src="RationaLlama.png" alt="Image Description" width="250" />


RationaLlama is a Llama 2 model fine-tuned to solve logical reasoning tasks on the LogiQA dataset.

Medium Article: [RationaLlama: Fine-tuning an LLM for Logical Reasoning, and Why it's Hard. . .](https://medium.com/p/c590ff4081fc/edit)


## Env Installation

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
## 🤗Hugging Face CLI
Log in to Hugging Face from the terminal:
```
huggingface-cli login
```

## Dataset Used for RationaLlama 
The data sets used in the article can be found here:

Training Dataset: [LogiQA](https://github.com/lgw863/LogiQA-dataset)

Baseline Datasets: 
[ReClor](https://arxiv.org/abs/2002.04326),
[LogiQA 2.0](https://github.com/csitfun/LogiQA2.0)
