from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import logging
from sklearn.model_selection import train_test_split
from datasets import Dataset

def main():
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Load dataset
        logging.info("Loading dataset...")
        dataset = load_dataset('imdb', split='train')
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            dataset['text'], dataset['label'], test_size=0.1, random_state=42
        )

        # Convert to Hugging Face Dataset format
        train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
        val_dataset = Dataset.from_dict({'text': val_texts, 'label': val_labels})
        
        # Load tokenizer and model
        logging.info("Loading tokenizer and model...")
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

        # Tokenize data
        def tokenize_function(examples):
            return tokenizer(examples['text'], padding='max_length', truncation=True)

        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)

        # Prepare for training
        training_args = TrainingArguments(
            output_dir='./results', 
            num_train_epochs=3, 
            per_device_train_batch_size=2, 
            evaluation_strategy="epoch",
            save_total_limit=2,
            logging_dir='./logs',
        )

        trainer = Trainer(
            model=model, 
            args=training_args, 
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        # Fine-tune model
        logging.info("Starting training...")
        trainer.train()

        # Save model
        logging.info("Saving model...")
        trainer.save_model('./results/fine-tuned-model')
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()

