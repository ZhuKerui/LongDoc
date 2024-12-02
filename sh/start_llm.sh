export CUDA_VISIBLE_DEVICES=2
# MODEL=meta-llama/Llama-2-7b-chat-hf
# MODEL=mistralai/Mistral-7B-Instruct-v0.3
# MODEL=microsoft/Phi-3-mini-128k-instruct
MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct

python -m vllm.entrypoints.openai.api_server --model $MODEL --tensor-parallel-size 1 --dtype auto --enable-prefix-caching --max-model-len 32000 #--port 8001
# python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3.1-8B-Instruct --tensor-parallel-size 2 --dtype auto --enable-prefix-caching --max-model-len 32000