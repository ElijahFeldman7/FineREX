import os
from transformers import TrainingArguments
from .config import (
    DATASET_PATH,
    SYSTEM_PROMPT,
    NEW_MODEL_DIR,
)
from .data_processing import load_data, create_preprocess_function
from .model_setup import setup_model_and_tokenizer, setup_peft_model
from .metrics import compute_metrics_wrapper, preprocess_logits_for_metrics
from .trainer import CustomTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    FINAL_OUTPUT_DIR = "/scratch/efeldma5/uniner_project/model"
    
    REPORT_DIR = NEW_MODEL_DIR 
    
    model, tokenizer = setup_model_and_tokenizer()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
    model = setup_peft_model(model)
    formatted_system_prompt = SYSTEM_PROMPT.format(
        tuple_delimiter="|",
        record_delimiter="\n",
        completion_delimiter="<END>"
    )
    train_dataset, eval_dataset = load_data(DATASET_PATH)
    
    preprocess_function = create_preprocess_function(tokenizer, SYSTEM_PROMPT)
    
    print("Tokenizing train dataset...")
    tokenized_train = train_dataset.map(
        preprocess_function, 
        batched=True, 
        remove_columns=list(train_dataset.features)
    )
    print("Tokenizing evaluation dataset...")
    tokenized_eval = eval_dataset.map(
        preprocess_function, 
        batched=True, 
        remove_columns=list(eval_dataset.features)
    )

    training_args = TrainingArguments(
        output_dir=FINAL_OUTPUT_DIR,
        overwrite_output_dir=True,
        per_device_train_batch_size=1,      
        gradient_accumulation_steps=4,     
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        num_train_epochs=4,                
        learning_rate=2e-5,     
        lr_scheduler_type="cosine",           
        optim="paged_adamw_8bit",
        bf16=True,                         
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        report_to="tensorboard",           
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        raw_eval_dataset=eval_dataset,
        system_prompt=formatted_system_prompt,
        tokenizer=tokenizer,
        compute_metrics=lambda eval_pred: compute_metrics_wrapper(eval_pred, tokenizer),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        report_dir=REPORT_DIR
    )

    print("\n--- Starting Llama 3.1 QLoRA Fine-tuning ---")
    trainer.train()
    
    print(f"\n--- Saving Final Model to {FINAL_OUTPUT_DIR} ---")
    trainer.save_model(FINAL_OUTPUT_DIR)
    tokenizer.save_pretrained(FINAL_OUTPUT_DIR)
    
    print(f"\n--- Running Final Evaluation (Report: {REPORT_DIR}) ---")
    metrics = trainer.evaluate()
    print(metrics)
    print("\n--- Training Complete ---")

if __name__ == "__main__":
    main()