# InferGen

Code generator for inference on Quantized Large Language Models. Quantization is done using [GPTQ](https://github.com/IST-DASLab/gptq).

## Current features

* Support for LlaMA and OPT 
* 4,3, and 2 bit inference
* x86 with AVX2 support
* Support for `pyTorch` and `transformers`
* Support for generic quantization group size

## TODOs

* Support for ARM Neon
* Support for AVX512
* Including quantization error analysis in code generation

## Usage

### Installation

1. Install dependencies via `pip install -r requirements.txt`
2. Install transformers from source `pip install git+https://github.com/huggingface/transformers`
3. Install the python module `python setup.py install`. This will run a search to find the best parameters for register usage.

### Example

We give an example notebook in `demo.ipynb`. The basic workflow is 

* load floating point model,
* load quantized checkpoint from GPTQ,
* call the `infergen.swap_modules_llama(model, quantized_checkpoint, bits=4, p=64, l1=l1, inplace=False)` function, where `model` is the full-size model, `quantized_checkpoint` is the quantized model, `bits` is the number of bits used for the quantization,`l1` is the size of the l1 data cache in bits, `p` is the number of cores to use, and `inplace` is a flag to swap in place or creating a copy.
* Use the quantized model as a normal transformer.
