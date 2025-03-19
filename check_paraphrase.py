import os
import datasets
import pandas as pd
from openai import OpenAI

import utils

client = OpenAI()

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

def check_paraphrase(sample):
    original_para = sample['Summary']
    paraphrased_para = sample['paraphrased_Summary']

    prompt = 'Original paragraph:\n\n' + original_para
    prompt += '\n\nParaphrased paragraph:\n\n' + paraphrased_para 
    sample['verification'] = generate_response(sys_prompt, prompt)
    print(sample['verification'])

    return sample
    

if __name__ == '__main__':
    utils.set_seed(42)
    sys_prompt = """
        You are a paraphrasing verifier.
        Your role is to check whether the provided paraphrased paragraph is semantically equivalent to the original paragraph.
    """
    os.makedirs('main_results', exist_ok=True)
    out_dir = 'main_results/check_paraphrased_arxiv.pkl'

    # load dataset
    # train_dataset = datasets.load_dataset('locuslab/TOFU', 'full')['train']
    # apply paraphrasing function  
    # train_dataset = train_dataset.map(paraphrase)

    train_dataset = pd.read_pickle('main_results/paraphrased_arxiv.pkl')
    # train_dataset = train_dataset[:3]
    train_dataset = train_dataset.apply(check_paraphrase, axis=1)
    train_dataset.to_pickle(out_dir)
    
            
    # save paraphrased dataset
    # train_dataset.save_to_disk(out_dir)
    print(f'Saved paraphrased data to {out_dir}')