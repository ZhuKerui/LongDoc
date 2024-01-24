export CUDA_VISIBLE_DEVICES=1,2
MODEL=meta-llama/Llama-2-7b-hf

python -m vllm.entrypoints.openai.api_server --model $MODEL --tensor-parallel-size 2