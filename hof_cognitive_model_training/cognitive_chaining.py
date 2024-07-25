import argparse
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch
import logging

class HOFAgent:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def process(self, input_data):
        # Process data using the model
        inputs = self.tokenizer(input_data, return_tensors='pt', truncation=True, padding=True)
        outputs = self.model(**inputs)
        return outputs

def main():
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='HOF Cognitive Agent Chaining.')
    parser.add_argument('--username', type=str, required=True, help='Hugging Face username')
    args = parser.parse_args()

    try:
        # Load fine-tuned model and tokenizer
        logging.info("Loading model and tokenizer...")
        model = DistilBertForSequenceClassification.from_pretrained(f'{args.username}/fine-tuned-distilbert')
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        # Create HOF agents
        agent1 = HOFAgent(model, tokenizer)
        agent2 = HOFAgent(model, tokenizer)

        # Define chaining
        def cognitive_chaining(input_data):
            intermediate_result = agent1.process(input_data)
            final_result = agent2.process(intermediate_result)
            return final_result

        # Example usage
        input_data = "This movie was awesome!"
        result = cognitive_chaining(input_data)
        print(result)
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()

