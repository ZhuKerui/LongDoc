export CUDA_VISIBLE_DEVICES=1
# MODEL=meta-llama/Llama-2-7b-chat-hf
MODEL=mistralai/Mistral-7B-Instruct-v0.2

python -m vllm.entrypoints.openai.api_server --model $MODEL --tensor-parallel-size 1 --dtype bfloat16