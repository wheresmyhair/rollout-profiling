export HF_HOME=/cloud/cloud-ssd1/hfhome
export HF_ENDPOINT=https://hf-mirror.com
export NCCL_DEBUG=INFO
export NCCL_CUMEM_HOST_ENABLE=0
vllm serve Qwen/Qwen3-8B \
    --dtype bfloat16 \
    --max-model-len 32000 --host 127.0.0.1 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --gpu-memory-utilization 0.9 \
    --data-parallel-size 4