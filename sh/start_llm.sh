export CUDA_VISIBLE_DEVICES=1
MODEL=meta-llama/Llama-2-7b-chat-hf
# MODEL=mistralai/Mistral-7B-Instruct-v0.3
# MODEL=microsoft/Phi-3-mini-128k-instruct

python -m vllm.entrypoints.openai.api_server --model $MODEL --tensor-parallel-size 1 --dtype auto #--port 8001 #--enable-prefix-caching #--max-model-len 32000
