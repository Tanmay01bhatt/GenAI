'''
Installing dependemcies:
pip3 install transformers==4.44.1
pip3 install accelerate
pip3 install bitsandbytes==0.43.3
!pip install -U "huggingface_hub[cli]"

Hugging Face Login:
!huggingface-cli login
'''

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

'''
AutoModelForCausalLM: The auto model class to load a pretrained model.
AutoTokenizer: The auto tokenizer class to load the tokenizer of the selected model.
BitsAndBytesConfig: The configuration class for bitsandbytes quantization.

'''

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map = "auto")

param_dtypes = [param.dtype for param in model.parameters()]
print("Parameter dtypes:", param_dtypes)

# Check the memory footprint
print(model.get_memory_footprint())
# 32121053440 = memory required to store the model is around 32121053440 bytes (30GB).

# inference
tokenizer = AutoTokenizer.from_pretrained(model_name)
input = tokenizer("Portugal is", return_tensors="pt").to('cuda')

response = model.generate(**input, max_new_tokens = 50)
print(tokenizer.batch_decode(response, skip_special_tokens=True))

#Implementing quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit = True
)

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
quantized_model = AutoModelForCausalLM.from_pretrained(model_name,
                    quantization_config = bnb_config,
                    device_map = "auto")
param_dtypes = [param.dtype for param in quantized_model.parameters()]
print("Parameter dtypes:", param_dtypes)

print(quantized_model.get_memory_footprint())
#9081209088 = the quantization significantly reduced the memory required to store the model from 30 GB to 9081209088 bytes, which is only around 8.45 GB.

# inference with quantized model
tokenizer = AutoTokenizer.from_pretrained(model_name)
input = tokenizer("Portugal is", return_tensors="pt").to('cuda')

response = quantized_model.generate(**input, max_new_tokens = 50)
print(tokenizer.batch_decode(response, skip_special_tokens=True))
