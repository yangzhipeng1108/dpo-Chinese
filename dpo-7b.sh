output_model=output
if [ ! -d ${output_model} ];then
    mkdir ${output_model}
fi
deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 29000 dpo.py \
    --model_name_or_path merged_model_output \
    --tokenizer_name /root/llama/llama2-lora-fine-tuning/model/models--daryl149--llama-2-7b-chat-hf/snapshots/bbc9b373dacff93e600e4426f2b3d3dd264e90ed  \
    --train_files /root/llama1/rl_dpo/base/helpful_base_cn_train.jsonl \
    --validation_files  /root/llama1/rl_dpo/base/helpful_base_cn_test.jsonl \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 200 \
    --lora_r 8 \
    --lora_alpha 16 \
    --output_dir ${output_model}
