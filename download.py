cur_folder = './'

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
from huggingface_hub import login
login('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')
# login(os.environ['HUGGINGFACE_ACCESS_TOKEN'])
from transformers import AutoTokenizer, GenerationConfig, AutoModel, AutoModelForCausalLM


for model_name, model_label in [
    # ("Salesforce/xgen-7b-4k-base", "xgen-7b-4k-base"),
    # ("Salesforce/xgen-7b-8k-base", "xgen-7b-8k-base"),
    # ("Salesforce/xgen-7b-8k-inst", "xgen-7b-8k-inst"),
    # ("mosaicml/mpt-7b", "mpt-7b-2k-base"),
    # ("mosaicml/mpt-7b-instruct", "mpt-7b-2k-inst"),
    # ("mosaicml/mpt-7b-chat", "mpt-7b-2k-chat"),
    # ("mosaicml/mpt-7b-8k", "mpt-7b-8k-base"),
    # ("mosaicml/mpt-7b-8k-chat", "mpt-7b-8k-chat"),
    # ("mosaicml/mpt-7b-8k-instruct", "mpt-7b-8k-inst")
    # ("tiiuae/falcon-7b", "falcon-7b-2k-base"),
    # ("tiiuae/falcon-7b-instruct", "falcon-7b-2k-inst"),
    # ("togethercomputer/RedPajama-INCITE-7B-Base", "redpajama-7b-2k-base"),
    # ("togethercomputer/RedPajama-INCITE-7B-Chat", "redpajama-7b-2k-chat"),
    # ("togethercomputer/RedPajama-INCITE-7B-Instruct", "redpajama-7b-2k-inst"),
    # ("mistralai/Mistral-7B-v0.1", "mistral-7b-32k-base"),
    # ("mistralai/Mistral-7B-Instruct-v0.1", "mistral-7b-32k-inst"),
    # ("meta-llama/Llama-2-7b-hf", "llama-2-7b-4k-base"), 
    # ("meta-llama/Llama-2-7b-chat-hf", "llama-2-7b-4k-chat"), 
    # ("lmsys/vicuna-7b-v1.5", "vicuna-7b-4k-chat"), 
    # ("lmsys/vicuna-7b-v1.5-16k", "vicuna-7b-16k-chat"), 
    # ("togethercomputer/LLaMA-2-7B-32K", "llama-2-7b-32k-base"), 
    # ("togethercomputer/Llama-2-7B-32K-Instruct", "llama-2-7b-32k-inst"),
    # ("THUDM/chatglm-6b", "chatglm-6b-2k-chat"),
    # ("THUDM/chatglm2-6b", "chatglm2-6b-8k-chat"),
    # ("THUDM/chatglm2-6b-32k", "chatglm2-6b-32k-chat"),
    # ("THUDM/chatglm3-6b-base", "chatglm3-6b-32k-base"),
    # ("THUDM/chatglm3-6b", "chatglm3-6b-8k-chat"),
    # ("THUDM/chatglm3-6b-32k", "chatglm3-6b-32k-chat")
    ]:


    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    del model