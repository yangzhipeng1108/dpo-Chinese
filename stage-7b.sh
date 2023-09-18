output_model=output
ZERO_STAGE=3
if [ ! -d ${output_model} ];then
    mkdir ${output_model}
fi
deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 29000 dpo_ds_stage.py \
    --model_name_or_path /root/llama1/rl_dpo/model/models--internlm--internlm-chat-7b-8k/snapshots/8e22a0e87f14ffaf6fd865b32c4ba496d76e6ed7  \
    --tokenizer_name /root/llama1/rl_dpo/model/models--internlm--internlm-chat-7b-8k/snapshots/8e22a0e87f14ffaf6fd865b32c4ba496d76e6ed7  \
    --train_files /root/llama1/rl_dpo/base/helpful_base_cn_train.jsonl \
    --validation_files  /root/llama1/rl_dpo/base/helpful_base_cn_test.jsonl \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 2000 \
    --zero_stage $ZERO_STAGE \
    --deepspeed \
    --lora_r 8 \
    --lora_alpha 16 \
    --output_dir ${output_model} \
    --logging_strategy steps \
    --logging_steps 10
