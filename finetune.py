import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


class QuestionDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input_ids": self.data.iloc[idx]["input_ids"][0],
            "labels": self.data.iloc[idx]["labels"][0],
        }


def load_model(
    model_name: str = "meta-llama/Llama-2-7b-chat-hf", load_on_gpu: bool = True
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    The load_model function loads a HuggingFace model and tokenizer.

    :param model_name: str: Specify the model to load
    :param load_on_gpu: bool: Determine whether to load the model on gpu or not
    :return: A tuple of the model and tokenizer
    """
    if load_on_gpu:
        # Load in Bits & Bytes configuration
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        # Load the pretrained model & tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            quantization_config=config,
            attn_implementation="flash_attention_2",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set the padding token to the eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def tokenize_queries(
    tokenizer: AutoTokenizer, query: str, max_length: int = 1024
) -> torch.Tensor:
    """
    The tokenize_queries function takes in a tokenizer, query string, and max length.
    It then creates a list of dictionaries that contain the role (system or user) and content (the actual text).
    The function then applies the chat template to this list of dictionaries using the tokenizer.

    :param tokenizer: AutoTokenizer: Tokenize the query
    :param query: str: Pass in the user's question
    :param max_length: int: Set the maximum length of the input
    :return: A dictionary with the following keys:
    """
    # Create the query
    query = [
        {
            "role": "system",
            "content": "You answer multiple choice questions with the correct letter answer. Your answer should be in this format: '{Letter}.{Answer}'",
        },
        {"role": "user", "content": query},
    ]

    # Tokenize the query
    input_ids = tokenizer.apply_chat_template(
        query,
        padding="max_length",
        max_length=max_length,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    return input_ids


def tokenize_responses(
    tokenizer: AutoTokenizer, query: str, response: str, max_length: int = 1024
) -> torch.Tensor:
    response = [
        {
            "role": "system",
            "content": "You answer multiple choice questions with the correct letter answer. Your answer should be in this format: '{Letter}.{Answer}'",
        },
        {"role": "user", "content": query},
        {"role": "assistant", "content": response},
    ]

    labels = tokenizer.apply_chat_template(
        response,
        padding="max_length",
        max_length=max_length,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    return labels


def main() -> None:
    # Load the model & tokenizer
    model, tokenizer = load_model()

    # Add Lora Matrices
    lora_config = LoraConfig(
        r=20,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.4,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    peft_model = get_peft_model(model, lora_config)

    # Print the number of trainable parameters
    print(
        f"Number of trainable parameters: {sum(p.numel() for p in peft_model.parameters() if p.requires_grad)}"
    )

    # Load in the data
    train_data = pd.read_csv("data/train.csv")
    eval_data = pd.read_csv("data/eval.csv")

    # Tokenize the data
    train_data["input_ids"] = train_data.apply(
        lambda row: tokenize_queries(tokenizer, row["query"]), axis=1
    )
    train_data["labels"] = train_data.apply(
        lambda row: tokenize_responses(tokenizer, row["query"], row["response"]), axis=1
    )

    eval_data["input_ids"] = eval_data.apply(
        lambda row: tokenize_queries(tokenizer, row["query"]), axis=1
    )
    eval_data["labels"] = eval_data.apply(
        lambda row: tokenize_responses(tokenizer, row["query"], row["response"]), axis=1
    )

    # Drop unecessary columns
    train_data = train_data.drop(columns="query").drop(columns="response")
    eval_data = eval_data.drop(columns="query").drop(columns="response")

    # Setup trainer
    training_args = TrainingArguments(
        output_dir="./checkpoints",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=1e-4,
        num_train_epochs=5,
        logging_steps=1,
        evaluation_strategy="steps",
        eval_steps=4000,
        save_steps=1000,
        report_to="wandb",
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=QuestionDataset(train_data),
        eval_dataset=QuestionDataset(eval_data),
    )

    # Now train!
    trainer.train()

    return


if __name__ == "__main__":
    main()
