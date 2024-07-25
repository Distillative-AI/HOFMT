import argparse
from huggingface_hub import HfApi
import logging

def main():
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Upload fine-tuned model to Hugging Face.')
    parser.add_argument('--token', type=str, required=True, help='Hugging Face API token')
    args = parser.parse_args()

    try:
        # Log in to Hugging Face
        logging.info("Logging in to Hugging Face...")
        api = HfApi()
        api.login(token=args.token)

        # Upload model
        logging.info("Uploading model...")
        api.upload_folder(
            folder_path='./results/fine-tuned-model',
            repo_id='fine-tuned-distilbert',
            repo_type='model',
        )
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()

