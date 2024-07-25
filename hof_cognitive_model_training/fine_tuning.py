from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

def main():
    # Load dataset
    dataset = load_dataset('imdb', split='train[:1%]')

    # Load tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

    # Tokenize data
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Prepare for training
    training_args = TrainingArguments(
        output_dir='./results', 
        num_train_epochs=3, 
        per_device_train_batch_size=2, 
        save_steps=10_000, 
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=tokenized_datasets,
    )

    # Fine-tune model
    trainer.train()

if __name__ == '__main__':
    main()
