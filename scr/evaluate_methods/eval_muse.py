from rouge_score import rouge_scorer
from typing import List, Dict, Tuple
from scipy.stats import bootstrap
import numpy as np
from tqdm.contrib import tzip
from typing import List
from tqdm import tqdm

class RougeEvalLogger:

    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(
            rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
            use_stemmer=False
        )
        self.history = []


    def log(self, prompt: str, gt: str, output: str, question: str | None = None):
        try:
            score = self.scorer.score(gt, output)
        except:
            print(f'gt: {gt}, \n output: {output}')
            gt = str(gt)
            output = str(output)
            score = self.scorer.score(gt, output)
        d = {
            'prompt': prompt,
            'gt': gt,
            'response': output,
            'rougeL': score['rougeL'].fmeasure,
            'rougeL_recall': score['rougeL'].recall,
            'rouge1': score['rouge1'].fmeasure,
            'rouge1_recall': score['rouge1'].recall
        }
        if question is not None: d['question'] = question
        self.history.append(d)


    def report(self) -> Tuple[Dict, Dict]:
        agg = {}
        for key, val in self.history[0].items():
            if isinstance(val, str): continue
            vals: List[float] = [item[key] for item in self.history]
            agg[f"max_{key}"] = max(vals)
            agg[f"mean_{key}"] = sum(vals) / len(vals)
            agg[f"{key}_ci_lo"], agg[f"{key}_ci_hi"] = bootstrap(
                (vals,), np.mean,
                confidence_level=0.95,
                method='percentile'
            ).confidence_interval
        return agg, self.history

def get_prefix_before_words_occur(string: str, words: List[str]) -> str:
    for word in words: string = string.split(word)[0]
    return string


def eval(
    model, tokenizer,
    questions: List[str], answers: List[str],
    icl_qs: List[str] = [], icl_as: List[str] = [],
    max_new_tokens: int = 32,
    batch_size: int = 64
):
    
    logger = RougeEvalLogger()
    general_prompt = ""
    for question, answer in zip(icl_qs, icl_as):
        general_prompt += f"Question: {question}\nAnswer: {answer}\n\n"
    
    num_batches = (len(questions) + batch_size - 1) // batch_size  # Calculate number of batches    
    # tokenizer.padding_side = 'left'
    left_pad_tokenizer = tokenizer
    left_pad_tokenizer.padding_side = 'left'
    left_pad_tokenizer.padding_size = 'longest'
    left_pad_tokenizer.pad_token = left_pad_tokenizer.eos_token
    left_pad_tokenizer.pad_token_id = left_pad_tokenizer.eos_token_id
    
    for batch_idx in tqdm(range(num_batches)):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(questions))

        batch_questions = questions[start:end]
        batch_answers = answers[start:end]
        
        batch_prompts = [general_prompt + f"Question: {question}\nAnswer: " for question in batch_questions]
        inputs = left_pad_tokenizer.batch_encode_plus(batch_prompts, add_special_tokens=True, return_tensors='pt', padding=True).to(model.device)
        # # now generate
        output_ids = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True, pad_token_id=left_pad_tokenizer.eos_token_id)
        output_strs = left_pad_tokenizer.batch_decode(output_ids[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        
        # Log results and calculate ROUGE
        for prompt, answer, output, question in zip(batch_prompts, batch_answers, output_strs, batch_questions):
            # Log results and calculate ROUGE
            logger.log(prompt, answer, output, question=question)
        # logger.log(prompt, answer, output, question=question)

    return logger.report()