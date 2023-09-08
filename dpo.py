#!/usr/bin/env python
# coding: utf-8
###导入必要的包
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
import deepspeed
from trl import DPOTrainer

import os
from datasets import load_dataset
import transformers
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")

    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--train_files",
        type=str,
        help=
        "The input training data file (a text file).",
        required=True,
    )
    parser.add_argument(
        "--validation_files",
        type=str,
        help=
        "An optional input evaluation data file to evaluate the perplexity on (a text file).",
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help=
        "Where to store the model.",
        required=True,
    )


    parser.add_argument(
        "--num_train_epochs",
        type=int,
        help=
        "Where to store the model.",
        required=True,
    )

    parser.add_argument(
        "--local_rank",
        type=int,
        help=
        "Where to store the model.",
    )


    # parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args
args = parse_args()

if args.tokenizer_name == '' or args.tokenizer_name == None:
    args.tokenizer_name = args.model_name_or_path

###定义dpo策略模型
# tokenizer_kwargs = {
#         "use_fast": True,
#         "use_auth_token": True ,
#         "padding_side": 'left'
#     }
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name ,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, ###替换成你的模型
                                             trust_remote_code=True,
                                             quantization_config=BitsAndBytesConfig(
                                                 load_in_4bit=True,
                                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                                 bnb_4bit_use_double_quant=True,
                                                 bnb_4bit_quant_type='nf4'
                                             ),
                                             device_map = {"": int(args.local_rank or 0)}
                                             # device_map="auto"
                                             )

model = prepare_model_for_kbit_training(model)

### 所有的线性layer都装配上lora
import bitsandbytes as bnb
def find_all_linear_names(model):
    #cls = bnb.nn.Linear8bitLt
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)
modules = find_all_linear_names(model)

print(modules)
config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=0.05,
    bias="none",
    target_modules=modules,
    task_type="CAUSAL_LM",
)


model = get_peft_model(model, config)



###定义参考模型
model_ref = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, ###替换成你的模型
                                             trust_remote_code=True,
                                             quantization_config=BitsAndBytesConfig(
                                                 load_in_4bit=True,
                                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                                 bnb_4bit_use_double_quant=True,
                                                 bnb_4bit_quant_type='nf4'
                                             ),
                                            device_map={"": int(args.local_rank or 0)}
                                             # device_map="auto"
                                                 )


###准备训练数据
data_files = {}
data_files["train"] = args.train_files
data_files["validation"] = args.validation_files

dataset = load_dataset("json", data_files=data_files)


train_data = dataset["train"]
val_data = dataset["validation"]

# dataset = load_dataset("json", data_files="base/harmless_base_cn_train.jsonl")
# train_val = dataset["train"].train_test_split(
#         test_size=2000, shuffle=True, seed=42
#     )
# train_data = train_val["train"]
# val_data = train_val["test"]



def extract_anthropic_prompt(prompt_and_response):
    final = ""
    for sample in prompt_and_response:
        final += sample["role"] + "\n" +sample["text"]
    final += "\n"
    return final


def get_hh(dataset,split: str, sanity_check: bool = False, silent: bool = False, cache_dir: str = None) -> Dataset:
    """Load the Anthropic Helpful-Harmless dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts should be structured as follows:
      \n\nHuman: <prompt>\n\nAssistant:
    Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
    """
    dataset = dataset
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def split_prompt_and_responses(sample) -> Dict[str, str]:
        prompt = extract_anthropic_prompt(sample["context"])
        return {
            "prompt": prompt,
            "chosen": sample["chosen"]["role"] + "\n" +sample["chosen"]["text"],
            "rejected": sample["rejected"]["role"]  + "\n" +sample["rejected"]["text"],
        }

    return dataset.map(split_prompt_and_responses)


train_dataset = get_hh(train_data,"train", sanity_check=True)
eval_dataset = get_hh(val_data,"test", sanity_check=True)


###定义dpo训练参数
training_args = TrainingArguments(
    per_device_train_batch_size=args.per_device_train_batch_size,
    max_steps=args.num_train_epochs,
    remove_unused_columns=False,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    evaluation_strategy="steps",
    output_dir=args.output_dir,
    report_to="tensorboard",
    local_rank=args.local_rank,
    do_train=True,
    do_eval=True,
    disable_tqdm=False,
    ddp_find_unused_parameters=False,
)

###定义dpo训练器
dpo_trainer = DPOTrainer(
    model,
    model_ref,
    args=training_args,
    beta=0.1,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)
###训练
dpo_trainer.train()
###模型保存
dpo_trainer.save_model()
