export CUDA_VISIBLE_DEVICES=2
# MODEL=meta-llama/Llama-2-7b-chat-hf
# MODEL=mistralai/Mistral-7B-Instruct-v0.2
MODEL=microsoft/Phi-3-mini-128k-instruct

python -m vllm.entrypoints.openai.api_server --model $MODEL --tensor-parallel-size 1 --dtype auto --max_model_len 32000
