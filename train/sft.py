import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from dataclasses import dataclass, field, asdict
from typing import Optional
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datasets import load_dataset, concatenate_datasets, DatasetDict
import transformers
import trl

@dataclass
class TrainingConfig:
    model_name: str = field(default="/home/agvanu/models/DeepSeek-R1-Distill-Qwen-1.5B")
    block_size: int = field(default=4096)
    # wandb_project: Optional[str] = field(default="s1")
    # wandb_entity: Optional[str] = field(default="hashimoto-group")
    # train_file_path: Optional[str] = field(default='simplescaling/s1K_tokenized')
    train_file_path: str = field(default="data/train_finetune.jsonl")
    validation_split_percentage: int = field(default=15)
    dagger: bool = field(default=False)

def train():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, sft_args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(sft_args)}
    logging.info(f"Training config: {log_config}")

    raw_dataset = load_dataset("json", data_files={"train": config.train_file_path},)["train"]
    
    split = raw_dataset.train_test_split(
    test_size=config.validation_split_percentage / 100.0,
    seed=sft_args.seed)

    dataset = DatasetDict({
        "train": split["train"],
        "validation": split["test"]
    })

    # loading model
    kwargs = {}
    if "70B" in config.model_name:
        # Removed "low_cpu_mem_usage": True, for 70B, since by default we are in FSDP,
        # it's more efficient to do  "cpu_ram_efficient_loading": true, in fsdp_config.json
        kwargs = {"device_map": "auto", "torch_dtype": "auto",
                  "attn_implementation": "flash_attention_2", "use_cache": False}
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name,
                                                                  local_files_only=True,
                                                                  torch_dtype="auto",
                                                                  use_cache=False)


    # setting up trainer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name,
                                                            use_fast=True,
                                                            local_files_only=True)
    if "Llama" in config.model_name:
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    elif "Qwen" in config.model_name:
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|fim_pad|>"

    # Only compute loss over assistant responses
    # Verified that it precisely starts where the thinking tokens start and ends with the first pad token
    # via labels being set to -100
    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )
    sft_args.dataset_text_field = 'text'
    sft_args.max_seq_length = config.block_size

    def tokenize_fn(examples):
        return tokenizer(
            examples[sft_args.dataset_text_field],
            truncation=True,
            max_length=config.block_size
        )
    
    tokenized_ds = dataset.map(tokenize_fn, batched=True)

    trainer = trl.SFTTrainer(
        model,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        args=sft_args,
        data_collator=collator
    )

    trainer.train()
    trainer.save_model(output_dir=sft_args.output_dir)
    tokenizer.save_pretrained(sft_args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
