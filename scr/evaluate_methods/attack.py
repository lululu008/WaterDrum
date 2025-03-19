import time
from transformers import T5Tokenizer, T5ForConditionalGeneration
import random
import torch

############################################### attack #############################################
def synonym_attack(prompt_sentence, k=0.2, device='cpu'):
    """
    :param prompt_sentence: prompt_sentence: list of string
    :param k: float: how many words are replaced to get synonym
    :return: list of string with words get replaced
    """

    import json
    import random
    import nltk
    from nltk.corpus import wordnet
    from nltk.tokenize import word_tokenize
    from PyDictionary import PyDictionary

    nltk.download('punkt')
    nltk.download('wordnet')

    def get_synonyms(word):
        synonyms = wordnet.synsets(word)
        return set(syn.lemmas()[0].name() for syn in synonyms)

    def replace_with_synonym(text, k):
        print(f"original sentence: {text}")
        words = word_tokenize(text)
        num_to_replace = int(len(words) * k)
        for _ in range(num_to_replace):
            word_to_replace = random.choice(words)
            synonyms = get_synonyms(word_to_replace)
            if synonyms:
                synonym_to_use = random.choice(list(synonyms))
                words[words.index(word_to_replace)] = synonym_to_use
        print(f"modified sentence: {' '.join(words)}")
        return ' '.join(words)

    synonym_data = [replace_with_synonym(item, k) for item in prompt_sentence]

    return synonym_data


def dipper_paraphrase_attack(prompt_sentence, k=0.2, device='cpu'):
    """
    :param prompt_sentence: prompt_sentence: list of string
    :return: list of string with words get replaced
    """

    import nltk
    nltk.download('punkt_tab')
    from nltk.tokenize import sent_tokenize

    class DipperParaphraser(object):
        def __init__(self, model="kalpeshk2011/dipper-paraphraser-xxl", verbose=True):
            time1 = time.time()
            self.tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-xxl')
            self.model = T5ForConditionalGeneration.from_pretrained(model)
            if verbose:
                print(f"{model} model loaded in {time.time() - time1}")
            self.model.cuda()
            self.model.eval()

        def paraphrase(self, input_text, lex_diversity, order_diversity, prefix="", sent_interval=3, **kwargs):
            """Paraphrase a text using the DIPPER model.

            Args:
                input_text (str): The text to paraphrase. Make sure to mark the sentence to be paraphrased between <sent> and </sent> blocks, keeping space on either side.
                lex_diversity (int): The lexical diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
                order_diversity (int): The order diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
                **kwargs: Additional keyword arguments like top_p, top_k, max_length.
            """
            assert lex_diversity in [0, 20, 40, 60, 80, 100], "Lexical diversity must be one of 0, 20, 40, 60, 80, 100."
            assert order_diversity in [0, 20, 40, 60, 80, 100], "Order diversity must be one of 0, 20, 40, 60, 80, 100."

            lex_code = int(100 - lex_diversity)
            order_code = int(100 - order_diversity)

            input_text = " ".join(input_text.split())
            sentences = sent_tokenize(input_text)
            prefix = " ".join(prefix.replace("\n", " ").split())
            output_text = ""

            for sent_idx in range(0, len(sentences), sent_interval):
                curr_sent_window = " ".join(sentences[sent_idx:sent_idx + sent_interval])
                final_input_text = f"lexical = {lex_code}, order = {order_code}"
                if prefix:
                    final_input_text += f" {prefix}"
                final_input_text += f" <sent> {curr_sent_window} </sent>"

                final_input = self.tokenizer([final_input_text], return_tensors="pt")
                final_input = {k: v.cuda() for k, v in final_input.items()}

                with torch.inference_mode():
                    outputs = self.model.generate(**final_input, **kwargs)
                outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                prefix += " " + outputs[0]
                output_text += " " + outputs[0]

            return output_text
    
    dp = DipperParaphraser()
    outputs = []
    for input_text in prompt_sentence:
        output = dp.paraphrase(input_text, lex_diversity=60, order_diversity=0, do_sample=True, top_p=0.75, top_k=None, max_length=512)
        outputs.append(output)

    return outputs

def pegasus_paraphrase_attack(prompt_sentence, k=0.2, device='cpu'):
    """
    :param prompt_sentence: prompt_sentence: list of string
    :return: list of string with words get replaced
    """
    from transformers import PegasusForConditionalGeneration, PegasusTokenizerFast
    model = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase")
    tokenizer = PegasusTokenizerFast.from_pretrained("tuner007/pegasus_paraphrase")
    model = model.to(device)

    def _get_paraphrased_sentences(model, tokenizer, sentence, num_return_sequences=5, num_beams=5):
        # tokenize the text to be form of a list of token IDs
        inputs = tokenizer([sentence], truncation=True, padding="longest", return_tensors="pt").to(device)
        # generate the paraphrased sentences
        outputs = model.generate(
            **inputs,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
        )
        # decode the generated sentences using the tokenizer to get them back to text
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)
    paraphrased_sentences = []
    for sentence in prompt_sentence:
        print(f"original sentence: {sentence}")
        buffer = _get_paraphrased_sentences(model, tokenizer, sentence, num_beams=10, num_return_sequences=10)
        # random_integer = random.randint(0, 9)
        random_integer = 0
        print(f"paraphrased sentence: {buffer[random_integer]}")
        paraphrased_sentences.append(buffer[random_integer])

    return paraphrased_sentences

def swap_chars_attack(prompt_sentence, k=0.2, device='cpu'):
    """
    :param prompt_sentence: prompt_sentence: list of string
    :param k: float: how many characters are swaped
    :return: list of string with words get replaced
    """
    import string
    modified_sentences = []

    for sentence in prompt_sentence:
        num_swaps = int(len(sentence) * k)
        print(f"original sentence: {sentence}")
        sentence = list(sentence)
        i = 0
        while i < num_swaps:
            swap_index = random.randint(0, len(sentence)-2)
            if sentence[swap_index] != ' ' and sentence[swap_index + 1] != ' ':
                sentence[swap_index], sentence[swap_index + 1] = sentence[swap_index + 1], sentence[swap_index]
                i+=1
        sentence = ''.join(sentence)
        print(f"modified sentence: {sentence}")
        modified_sentences.append(sentence)

    return modified_sentences


def insert_chars_attack(prompt_sentence, k=0.2, device='cpu'):
    """
    :param prompt_sentence: prompt_sentence: list of string
    :param k: float: how many characters are replaced to get synonym
    :return: list of string with words get replaced
    """
    import string
    modified_sentences = []

    for sentence in prompt_sentence:
        num_insertions = int(len(sentence) * k)
        print(f"original sentence: {sentence}")
        for _ in range(num_insertions):
            insert_index = random.randint(0, len(sentence))
            random_char = random.choice(string.ascii_letters)
            sentence = sentence[:insert_index] + random_char + sentence[insert_index:]
        print(f"modified sentence: {sentence}")
        modified_sentences.append(sentence)

    return modified_sentences


def insert_words_attack(prompt_sentence, k=0.2, device='cpu'):
    """
    :param prompt_sentence: prompt_sentence: list of string
    :param k: float: how many characters are replaced to get synonym
    :return: list of string with words get replaced
    """
    import random
    import nltk
    from nltk.corpus import words

    modified_sentences = []

    nltk.download('words')
    word_list = words.words()  # Get list of common English words

    for sentence in prompt_sentence:
        print(f"original sentence: {sentence}")
        words_in_sentence = sentence.split()
        num_insertions = int(len(words_in_sentence) * k)

        for _ in range(num_insertions):
            insert_index = random.randint(0, len(words_in_sentence))
            random_word = random.choice(word_list)  # Select random valid word from word_list
            words_in_sentence.insert(insert_index, random_word)

        modified_sentence = ' '.join(words_in_sentence)
        modified_sentences.append(modified_sentence)
        print(f"modified sentence: {modified_sentence}")
    return modified_sentences


def delete_chars_attack(prompt_sentence, k=0.2, device='cpu'):
    """
    :param prompt_sentence: prompt_sentence: list of string
    :param k: float: how many characters are replaced to get synonym
    :return: list of string with words get replaced
    """
    modified_sentences = []

    for sentence in prompt_sentence:
        print(f"original sentence: {sentence}")
        num_deletions = int(len(sentence) * k)

        for _ in range(num_deletions):
            delete_index = random.randint(0, len(sentence) - 1)
            sentence = sentence[:delete_index] + sentence[delete_index + 1:]

        modified_sentences.append(sentence)
        print(f"modified sentence: {sentence}")

    return modified_sentences


def delete_words_attack(prompt_sentence, k=0.2, device='cpu'):
    """
    :param prompt_sentence: prompt_sentence: list of string
    :param k: float: how many characters are replaced to get synonym
    :return: list of string with words get replaced
    """
    modified_sentences = []
    
    disrupt = 0
    for sentence in prompt_sentence:
        Match = True
        print("original sentence: {}".format(sentence.encode("unicode_escape").decode()))
        words = sentence.split()
        num_deletions = int(len(words) * k)

        for _ in range(num_deletions):
            delete_index = random.randint(0, len(words) - 1)
            del words[delete_index]

        modified_sentence = ' '.join(words)
        modified_sentences.append(modified_sentence)
        print("modified sentence: {}".format(modified_sentence.encode("unicode_escape").decode()))

    return modified_sentences