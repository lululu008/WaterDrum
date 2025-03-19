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

def paraphrase(sample):
    keys = list(sample.keys())
    for k in keys:
        if k == 'Summary':
            prompt = sample[k]
            paraphrased_text = generate_response(sys_prompt, prompt)
            sample[f'paraphrased_{k}'] = paraphrased_text
            print('> before paraphrased:', prompt) 
            print('< after paraphrased:', paraphrased_text)
    
    return sample
    

if __name__ == '__main__':
    utils.set_seed(42)
    sys_prompt = """
        You are a paraphraser.
        Your role is to paraphrase the following sentences while preserving its semantic similarity.
    """
    os.makedirs('main_results', exist_ok=True)
    out_dir = 'main_results/paraphrased_arxiv.pkl'

    # load dataset
    # train_dataset = datasets.load_dataset('locuslab/TOFU', 'full')['train']
    # apply paraphrasing function  
    # train_dataset = train_dataset.map(paraphrase)

    train_dataset = pd.read_pickle('~/maplecg_nfs_public/watermark_arxiv/arxiv_full.pkl')
    # train_dataset = train_dataset[:3]
    train_dataset = train_dataset.apply(paraphrase, axis=1)
    train_dataset.to_pickle(out_dir)
    
            
    # save paraphrased dataset
    # train_dataset.save_to_disk(out_dir)
    print(f'Saved paraphrased data to {out_dir}')