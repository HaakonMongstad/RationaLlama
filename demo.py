import pandas as pd
import torch
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_lora_model(
    adapter_name: str,
    model_name: str = "meta-llama/Llama-2-7b-chat-hf",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    The load_lora_model function loads a pretrained model and tokenizer, then applies an adapter to the model.

    :param adapter_name: str: Specify the adapter to use
    :param model_name: str: Specify the name of the model to load
    :return: A tuple of the model and tokenizer
    """

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
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load the adapter
    model = PeftModel.from_pretrained(model, adapter_name)

    # Set the padding token to the eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def main() -> None:
    # Determine the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and data
    model, tokenizer = load_lora_model("checkpoints/checkpoint-36000")
    eval_data = pd.read_csv("data/test.csv")

    query, response = eval_data.iloc[70]

    # Create the query
    chat = [
        {
            "role": "system",
            "content": "You answer multiple choice questions with the correct letter answer. Your answer should be in this format: '{Letter}.{Answer}'",
        },
        {"role": "user", "content": query},
    ]

    # Tokenize the query
    input_ids = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, return_tensors="pt"
    ).cuda()

    # Generate the response
    with torch.no_grad():
        output = model.generate(input_ids=input_ids, max_new_tokens=256)

    # Decode the response
    outputs = tokenizer.decode(
        output[0][input_ids.shape[-1] :], skip_special_tokens=True
    )

    outputs = outputs.split("[/INST] ")[0]

    print(f"Question: {query}")
    print(f"Model Answer: {outputs}")
    print(f"Correct Answer: {response}")


if __name__ == "__main__":
    main()
