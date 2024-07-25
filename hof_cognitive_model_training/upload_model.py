import argparse
from huggingface_hub import HfApi

def main():
    parser = argparse.ArgumentParser(description='Upload fine-tuned model to Hugging Face.')
    parser.add_argument('--token', type=str, required=True, help='Hugging Face API token')
    args = parser.parse_args()

    # Log in to Hugging Face
    api = HfApi()
    api.login(token=args.token)

    # Upload model
    trainer.push_to_hub("fine-tuned-distilbert")

if __name__ == '__main__':
    main()
