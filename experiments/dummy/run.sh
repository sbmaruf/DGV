#!/bin/bash
set -e

# QWEN 30b 3B AC
export CUDA_DEVICE_MAX_CONNECTIONS="1"
export NVTE_ALLOW_NONDETERMINISTIC_ALGO="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_NVLS_ENABLE=0


WORKSPACE="${1}"
DATA_PATH="${2}"
LOAD_PATH="${3}"
OUTPUT_PATH="${4}"
WANDB_PROJECT="${5}"
COMMENT="${6:-default}"
RDZV_ENDPOINT="${7}"


TP="${TP:-4}"
PP="${PP:-2}"
EP="${EP:-1}"
CP="${CP:-1}"
VPP="${VPP:-1}"
LAYERS_PER_VP="${LAYERS_PER_VP:-null}"


MBS="${MBS:-2}"
GBS="${GBS:-1024}"
SEQ_LEN="${SEQ_LEN:-4096}"
MOE_GROUPED_GEMM="${MOE_GROUPED_GEMM:-true}"
MOE_TOKEN_DISPATCHER="${MOE_TOKEN_DISPATCHER:-alltoall}"


MODEL_ARGS=(
    --use-mcore-models
    --num-layers 48
    --hidden-size 2048
    --ffn-hidden-size 6144
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 4
    --kv-channels 128
    --qk-layernorm
    --seq-length $SEQ_LEN
    --max-position-embeddings 40960
    --make-vocab-size-divisible-by 1187
    --untie-embeddings-and-output-weights
    --position-embedding-type rope
    --rotary-percent 1.0
    --rotary-base 1000000
    --rotary-seq-len-interpolation-factor 1
    --normalization RMSNorm
    --swiglu
    --norm-epsilon 1e-06
    --disable-bias-linear
    --no-create-attention-mask-in-dataloader
    --transformer-impl transformer_engine
)

MOE_ARGS=(
    --num-experts 128
    --moe-ffn-hidden-size 768
    --moe-router-load-balancing-type aux_loss
    --moe-router-topk 8
    --moe-aux-loss-coeff 1e-3
    --moe-token-dispatcher-type $MOE_TOKEN_DISPATCHER
    --moe-permute-fusion
    --moe-router-dtype fp32
    --moe-router-fusion
    --moe-grouped-gemm $MOE_GROUPED_GEMM
    --expert-model-parallel-size $EP
    --expert-tensor-parallel-size 1
)

source experiments/dummy/conf/dummy_iter.sh
FLAG=true
for path in "${DATA_PATH[@]}"; do
    if $FLAG; then
        DATA_ARGS=$path
        FLAG=false
    else
        DATA_ARGS+=" \"$path\""
    fi
done

DATA_ARGS=(
    --data-path $DATA_PATH
    --data-cache-path $WORKSPACE/data_cache
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model Qwen/Qwen3-30B-A3B
    --split 99,1,0
    --no-mmap-bin-files
    --num-workers 6
)

TRAINING_ARGS=(
    --micro-batch-size $MBS
    --global-batch-size $GBS
    --train-samples 268554688
    --sequence-parallel
    --use-flash-attn
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --clip-grad 1.0
    --weight-decay 0.1
    --lr 1.2e-4
    --min-lr 1.2e-5
    --lr-decay-style cosine
    --lr-decay-samples 255126953
    --lr-warmup-samples 162761
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.02
    --bf16
    --manual-gc
    --manual-gc-interval 5
    --cross-entropy-loss-fusion
    --cross-entropy-fusion-impl te
    --enable-experimental
    --exit-duration-in-mins 230
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP
    --pipeline-model-parallel-size $PP
    --context-parallel-size $CP
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    --distributed-timeout-minutes 60
)

if [ "$LAYERS_PER_VP" != "null" ] && [ -n "$LAYERS_PER_VP" ]; then
    MODEL_PARALLEL_ARGS+=(--num-layers-per-virtual-pipeline-stage $LAYERS_PER_VP)
fi

CHECKPOINT_ARGS=(
    --finetune
    --auto-detect-ckpt-format
    --load $LOAD_PATH
    --save $OUTPUT_PATH/checkpoints
    --save-interval 500
    --dist-ckpt-strictness log_all
)

VALIDATION_ARGS=(
    --eval-iters 32
    --eval-interval 500
)

LOGGING_ARGS=(
    --log-interval 1
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --log-num-zeros-in-grad
    --log-params-norm
    --log-validation-ppl-to-tensorboard
    --log-throughput
    --tensorboard-dir $OUTPUT_PATH/tensorboard
    --wandb-project $WANDB_PROJECT
    --wandb-exp-name Qwen3-30B-A3B-TP${TP}PP${PP}EP${EP}CP${CP}VPP${VPP}-MBS${MBS}GBS${GBS}-${COMMENT}
)

if [ -z "$RDZV_ENDPOINT" ]; then
    python third_party/Megatron-MoE-EA/pretrain_gpt.py \
        ${MODEL_ARGS[@]} \
        ${MOE_ARGS[@]} \
        ${DATA_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${MODEL_PARALLEL_ARGS[@]} \
        ${CHECKPOINT_ARGS[@]} \
        ${VALIDATION_ARGS[@]} \
        ${LOGGING_ARGS[@]}
else
    TORCHRUN_ARGS=(
        --nproc_per_node 8
        --nnodes $SLURM_JOB_NUM_NODES
        --rdzv_id $SLURM_JOB_ID
        --rdzv_backend c10d
        --rdzv_endpoint $RDZV_ENDPOINT
    )
    torchrun ${TORCHRUN_ARGS[@]} third_party/Megatron-MoE-EA/pretrain_gpt.py \
        ${MODEL_ARGS[@]} \
        ${MOE_ARGS[@]} \
        ${DATA_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${MODEL_PARALLEL_ARGS[@]} \
        ${CHECKPOINT_ARGS[@]} \
        ${VALIDATION_ARGS[@]} \
        ${LOGGING_ARGS[@]}
fi
