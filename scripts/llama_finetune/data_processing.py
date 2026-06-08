import pandas as pd
from datasets import Dataset
from .config import MAX_LENGTH

def load_data(file_path: str) -> (Dataset, Dataset):
    try:
        df = pd.read_csv(file_path).fillna('')
        df = df.dropna(subset=['Input_Text', 'Output'])
        df = df[df['Input_Text'].str.strip() != '']
        df = df[df['Output'].str.strip() != '']
    except (FileNotFoundError, KeyError) as e:
        print(f"Error: Could not read '{file_path}'. Make sure it exists and has 'Input_Text' and 'Output' columns. Details: {e}")
        exit()

    if len(df) == 0:
        print("Error: No valid data found in the CSV.")
        exit()

    dataset = Dataset.from_pandas(df)
    splits = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = splits["train"]
    eval_dataset = splits["test"]
    print(f"Dataset loaded: {len(train_dataset)} training samples, {len(eval_dataset)} evaluation samples.")
    return train_dataset, eval_dataset

def create_preprocess_function(tokenizer, system_prompt):
    formatted_system_prompt = system_prompt.format(
        tuple_delimiter="|",
        record_delimiter="\n",
        completion_delimiter="<END>"
    )

    INSTRUCTION_TEMPLATE = """Input_text: 
{input_text}
Output:
"""

    def preprocess_function(examples):
        all_input_ids =[]
        all_attention_masks = []
        all_labels = []

        for input_text, gt in zip(examples['Input_Text'], examples['Output']):
            user_content = INSTRUCTION_TEMPLATE.format(input_text=input_text).strip()
            
            clean_gt = gt.replace("{tuple_delimiter}", "|")
            clean_gt = clean_gt.replace("{record_delimiter}", "\n")
            clean_gt = clean_gt.replace("{completion_delimiter}", "<END>")
            
            messages =[
                {"role": "system", "content": formatted_system_prompt},
                {"role": "user", "content": user_content}
            ]
            
            prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_tokenized = tokenizer(prompt_str, add_special_tokens=False) 
            prompt_ids = prompt_tokenized["input_ids"]
            
            gt_str = clean_gt + tokenizer.eos_token
            gt_tokenized = tokenizer(gt_str, add_special_tokens=False)
            gt_ids = gt_tokenized["input_ids"]
            
            input_ids = prompt_ids + gt_ids
            
            if len(input_ids) > MAX_LENGTH:
                input_ids = input_ids[:MAX_LENGTH]
                
            labels = input_ids.copy()
            prompt_length = min(len(prompt_ids), MAX_LENGTH)
            labels[:prompt_length] = [-100] * prompt_length
            
            pad_len = MAX_LENGTH - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                attention_mask = [1] * (MAX_LENGTH - pad_len) + [0] * pad_len
                labels = labels + [-100] * pad_len
            else:
                attention_mask = [1] * MAX_LENGTH

            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_labels.append(labels)

        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
            "labels": all_labels
        }

    return preprocess_function