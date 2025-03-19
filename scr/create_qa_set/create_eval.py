
from argparse import ArgumentParser
from openai import OpenAI

from transformers import (
    AutoTokenizer
)
import torch
import re
import pandas as pd
from tqdm import tqdm

from scr.base_data import prepare_train_datasets

def generate_response(sys_prompt, prompt):
    try:
        # Call the GPT-4 model
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2048 
        )

        # Extract the text response from the model
        return completion.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"The prompt is: {prompt}")
        return None


if __name__ == "__main__":
    parser = ArgumentParser() 
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="Name or path to the model locally or on HuggingFace Hub")
    parser.add_argument("--dataset", type=str, default="arxiv",
                        help="Dataset name passed to load_dataset")
    parser.add_argument("--output_dir", type=str, default="results/",
                        help="Directory to save results and models")
    parser.add_argument("--watermarked_dir", type=str,
                        help="Directory to the watermarked dataset") 
    parser.add_argument("--split", type=str, default="forget10",
                        help="Dataset split. Default to train on entire training set.")
    parser.add_argument("--duplicate", type=int, default=0,
                        help="Whether to duplicate the forget set") 
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # def load_tokenizer(args):
    #     # load tokenizer
    #     tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    #     tokenizer.pad_token = tokenizer.eos_token

    #     return tokenizer


    # tokenizer = load_tokenizer(args)
    client = OpenAI()


    train_data, _, _ = prepare_train_datasets(args.dataset.lower(), 
                                                args.watermarked_dir,
                                                forget_split=args.split,
                                                duplicate=args.duplicate)

    sys_prompt = "You will be provided with an excerpt of text. Your goal is to create a question-answer pair that assesses reading comprehension and memorization, ensuring that the question can only be answered using details from the excerpt.\
    Please submit your response in the following format:\
    question: A single question related to the excerpt. The question should be specific enough that it does not allow for an answer other than the one you provide. In particular, it should not be answerable based on common knowledge alone. Also, a few words extracted from the excerpt must suffice in answering this question.\
    answer: A precise answer extracted verbatim, character-by-character from the excerpt. The answer to this question must be short, phrase-level at most. The length of the extraction should be minimal, providing the smallest span of the excerpt that completely and efficiently answers the question."
    
    questions = []
    answers = []
    
    for data in tqdm(train_data):
        prompt = data['text']
        response = generate_response(sys_prompt, prompt)
        if response:
            response = response.replace('\n', '')
            question_match = re.search(r'question:\s*(.*?)(?=\s*answer:)', response, re.S)
            answer_match = re.search(r'answer:\s*(.*)', response, re.S)

            question = question_match.group(1).strip() if question_match else None
            answer = answer_match.group(1).strip() if answer_match else None
            
            questions.append(question)
            answers.append(answer)
            
            print(f"Question: {question}", f"Answer: {answer}", sep='\n')
    
    df = pd.DataFrame({'question': questions, 'answer': answers})
    df.to_csv('QA_watermarked.csv', index=False)
    
    # df_1 = pd.read_csv('QA_old.csv')
    # df_combined = pd.concat([df, df_1], ignore_index=True)
    
    # df_combined.to_csv('QA.csv', index=False)

    print("Data successfully saved to 'QA_watermarked.csv'")