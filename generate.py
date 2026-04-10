from typing import List, Tuple
from dataclasses import dataclass
import logging
import string
import spacy
import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel
from retriever import BM25
import os
import json
import numpy as np
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class OOMError(Exception):

    def __init__(self, partial_text):
        self.partial_text = partial_text
        super().__init__(f'OOM with partial text length={len(partial_text)}')
DEBUG = True
FLAG = True
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
nlp = spacy.load('en_core_web_sm')

@dataclass
class Block:
    text: str = None
    tokens: List[str] = None
    range_: List[Tuple[int, int]] = None

    @property
    def len_tokens(self):
        return len(self.tokens)

    @property
    def len_words(self):
        return len(self.range_)

def merge_blocks(blocks: List[Block]) -> Block:
    text = ''.join([block.text for block in blocks])
    tokens = sum([block.tokens for block in blocks], [])
    range_ = []
    st = 0
    for block in blocks:
        if block.range_:
            for l, r in block.range_:
                range_.append((st + l, st + r))
            st = range_[-1][1]
    return Block(text=text, tokens=tokens, range_=range_)

class Counter:

    def __init__(self):
        self.retrieve = 0
        self.generate = 0
        self.hallucinated = 0
        self.token = 0
        self.sentence = 0

    def add_generate(self, text, tokenizer):
        self.generate += 1
        ids = tokenizer(text, return_tensors='pt')['input_ids'][0].tolist()
        self.token += len(ids)
        sentences = [sent.text for sent in nlp(text).sents]
        self.sentence += len(sentences)

    def calc(self, other_counter):
        return {'retrieve_count': self.retrieve - other_counter.retrieve, 'generate_count': self.generate - other_counter.generate, 'hallucinated_count': self.hallucinated - other_counter.hallucinated, 'token_count': self.token - other_counter.token, 'sentence_count': self.sentence - other_counter.sentence}

@dataclass
# class GeneratorOutput:
#     ended: bool
#     empty: bool
#     blocks: List[Block] = None
#     merged_blocks: Block = None
#     atten: Tensor = None
#     max_atten: Tensor = None
#     entropies: Tensor = None
#     entropies_s1: Tensor = None
#     entropies_s2: Tensor = None
#     smooth_s2: Tensor = None
#     mt_s2: Tensor = None
#     fun_word: Tensor = None
#     full_attention: Tensor = None
#     attention_to_context: Tensor = None
#     atten_dispersion: Tensor = None

    @property
    def new_text(self):
        return self.blocks[-1].text

    @property
    def len_new_words(self):
        return self.blocks[-1].len_words

class Generator:

    def __init__(self, model_name_or_path: str):
        logger.info(f'Loading model from {model_name_or_path}')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
        logger.info('Tokenizer loaded, now loading model weights...')
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map='balanced')
        self.model: PreTrainedModel
        logger.info(f'device = {self.model.device}')
        self.space_token = 'Ġ' if 'llama-3' in model_name_or_path.lower() else '▁'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokens_cannot_merged = {self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode('0' + ch)[-1:])[0] for ch in string.whitespace + string.punctuation} | {self.space_token, self.tokenizer.bos_token, self.tokenizer.eos_token}

    def simply_generate(self, input_text: str, max_length: int) -> Tuple[bool, str]:
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.model.device)
        input_length = input_ids.shape[1]
        output_ids = self.model.generate(input_ids=input_ids, max_new_tokens=max_length)[0, input_length:]
        decoded_text = self.tokenizer.decode(output_ids)
        if '\n' in decoded_text:
            newline_idx = decoded_text.index('\n')
            decoded_text = decoded_text[:newline_idx]
            output_ids = torch.tensor(self.tokenizer.encode(decoded_text, add_special_tokens=False), device=self.model.device)
        if len(output_ids) == 0:
            logger.info("generate '' in simply_generate()!")
            return (True, '')
        if output_ids[0] == self.tokenizer.bos_token_id:
            output_ids = output_ids[1:]
        if output_ids[-1] == self.tokenizer.eos_token_id:
            return (True, self.tokenizer.decode(output_ids[:-1]))
        return (False, self.tokenizer.decode(output_ids))

    def tokenize(self, text: str, is_start: bool=False):
        ids = self.tokenizer.encode(text)
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        if not is_start and tokens[0] == self.tokenizer.bos_token:
            tokens = tokens[1:]
        return tokens

    def merge_tokens(self, tokens) -> List[Tuple[int, int]]:
        range_ = []
        for i, t in enumerate(tokens):
            if i == 0 or t.startswith(self.space_token) or tokens[i] in self.tokens_cannot_merged or (tokens[i - 1] in self.tokens_cannot_merged):
                range_.append([i, i + 1])
            else:
                range_[-1][1] += 1
        return range_

    def build_block(self, text: str, is_start: bool=False) -> Block:
        tokens = self.tokenize(text, is_start=is_start)
        range_ = self.merge_tokens(tokens)
        return Block(text=text, tokens=tokens, range_=range_)

    def generate(self, input_texts: List[str], max_length: int) -> GeneratorOutput:
        blocks = []
        for text in input_texts:
            blocks.append(self.build_block(text, is_start=not blocks))
        input_tokens = sum([block.tokens for block in blocks], [])
        input_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(input_tokens)], device=self.model.device)
        input_len_tokens = len(input_tokens)
        outputs = self.model.generate(input_ids=input_ids, max_new_tokens=max_length, return_dict_in_generate=True, output_scores=True)
        new_token_ids = outputs.sequences[0, input_len_tokens:]
        print('len_tokens:', len(new_token_ids))
        decoded_text = self.tokenizer.decode(new_token_ids)
        if '\n' in decoded_text:
            newline_idx = decoded_text.index('\n')
            decoded_text = decoded_text[:newline_idx]
            new_token_ids = self.tokenizer.encode(decoded_text, add_special_tokens=False)
            new_token_ids = torch.tensor(new_token_ids, device=self.model.device)
            outputs.scores = outputs.scores[:len(new_token_ids)]
            outputs.sequences = torch.cat([input_ids, new_token_ids.unsqueeze(0)], dim=1)
        tokens = self.tokenizer.convert_ids_to_tokens(new_token_ids)
        if len(tokens) <= 1:
            return GeneratorOutput(empty=True, ended=True, blocks=None, merged_blocks=None, atten=None, max_atten=None, entropies=None, entropies_s1=None, entropies_s2=None, smooth_s2=None, fun_word=None, full_attention=None, attention_to_context=None)
        ended = tokens[-1] == self.tokenizer.eos_token
        if ended:
            tokens = tokens[:-1]
        text = self.tokenizer.convert_tokens_to_string(tokens)
        range_ = self.merge_tokens(tokens)
        new_block = Block(text=text, tokens=tokens, range_=range_)
        blocks.append(new_block)
        merged_blocks = merge_blocks(blocks)
        with torch.no_grad():
            model_output = self.model(outputs.sequences, output_attentions=True)
            full_attention_raw = model_output.attentions[-1][0][:, -new_block.len_tokens:, :]
            full_attention = full_attention_raw.cpu()
            attention_to_context_gpu = full_attention_raw.mean(dim=0)
            attention_to_context = torch.stack([attention_to_context_gpu[l:r, :].mean(dim=0) for l, r in range_], dim=0).cpu()
            atten_dispersion_list = []
            for word_i in range(attention_to_context.shape[0]):
                dist = attention_to_context[word_i]
                dist_norm = dist / (dist.sum() + 1e-10)
                dispersion = dist_norm.std().item()
                atten_dispersion_list.append(dispersion)
            atten_dispersion = torch.tensor(atten_dispersion_list, dtype=torch.float32)
            atten = full_attention_raw.mean(dim=0)
            atten = torch.stack([atten[:, l:r].sum(dim=-1) for l, r in merged_blocks.range_], dim=-1)
            atten = torch.stack([atten[l:r, :].mean(dim=-2) for l, r in range_], dim=-2)
            atten_to_new = atten[:, -new_block.len_words:]
            atten_to_new /= atten.sum(dim=-1, keepdim=True) + 1e-10
            max_atten, _ = atten_to_new.max(dim=1)
        del model_output, full_attention_raw, attention_to_context_gpu
        torch.cuda.empty_cache()
        probs = torch.stack(outputs.scores).softmax(dim=-1)
        entropies = (-probs * torch.log(probs + 1e-10)).sum(dim=-1)
        entropies = torch.stack([entropies[l:r, 0].max() for l, r in range_])
        func_words = []
        doc = nlp(new_block.text)
        real_words = set((token.text for token in doc if token.pos_ in ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM']))
        wl = 0
        wr = new_block.len_words
        for i in range(wl, wr):
            tl, tr = new_block.range_[i]
            word = self.tokenizer.convert_tokens_to_string(new_block.tokens[tl:tr])
            if not match(word, real_words):
                func_words.append(i)
        entropies_s1 = [{'key': i, 'val': torch.tensor(0, dtype=torch.float64)} for i in range(len(range_))]
        entropies_s2 = [{'key': i, 'val': torch.tensor(0, dtype=torch.float64)} for i in range(len(range_))]
        smooth_s2 = [{'key': i, 'val': torch.tensor(0, dtype=torch.float64)} for i in range(len(range_))]
        mt_s2 = [{'key': i, 'val': torch.tensor(0, dtype=torch.float64)} for i in range(len(range_))]
        fun_word = [{'key': i, 'val': torch.tensor(0, dtype=torch.float64)} for i in range(len(range_))]
        for i, (l, r) in enumerate(range_[:]):
            word = self.tokenizer.convert_tokens_to_string(new_block.tokens[l:r])
            if i not in func_words:
                fun_word[i]['val'] = torch.tensor(1, dtype=torch.float64)
        for i, (l, r) in enumerate(range_[1:]):
            word = self.tokenizer.convert_tokens_to_string(new_block.tokens[l:r])
            entropy = entropies[i + 1].item()
            if i + 1 not in func_words:
                j = i
                while j >= 0:
                    if j not in func_words:
                        s1 = entropies[i + 1].to(torch.float64) - entropies[j].to(torch.float64)
                        entropies_s1[i + 1]['val'] = s1
                        break
                    if j == 0:
                        break
                    else:
                        j -= 1
        for i, (l, r) in enumerate(range_[2:]):
            word = self.tokenizer.convert_tokens_to_string(new_block.tokens[l:r])
            entropy = entropies[i + 2].item()
            if i + 2 not in func_words:
                j = i + 1
                while j >= 1:
                    if entropies_s1[j]['val'].item() != 0:
                        s2 = entropies_s1[i + 2]['val'].to(torch.float64) - entropies_s1[j]['val'].to(torch.float64)
                        entropies_s2[i + 2]['val'] = s2
                        break
                    if j == 1:
                        break
                    else:
                        j -= 1
        count_fun = 0
        sum_s2 = 0
        Mt_1 = torch.tensor(0, dtype=torch.float64)
        for i, (l, r) in enumerate(range_[2:]):
            word = self.tokenizer.convert_tokens_to_string(new_block.tokens[l:r])
            if entropies_s2[i + 2]['val'] != 0:
                count_fun += 1
                sum_s2 += entropies_s2[i + 2]['val'].item()
                s2_mean = sum_s2 / count_fun
                w = torch.abs(Mt_1 - s2_mean) / (torch.abs(entropies_s2[i + 2]['val'] - s2_mean) + torch.abs(Mt_1 - s2_mean))
                α = 0.9 + 0.1 * w
                Mt = α * entropies_s2[i + 2]['val'] + (1 - α) * Mt_1
                mt_s2[i + 2]['val'] = Mt
                Mt_1 = entropies_s2[i + 2]['val']
        return GeneratorOutput(empty=False, ended=ended, blocks=blocks, merged_blocks=merged_blocks, atten=atten, max_atten=max_atten, entropies=entropies, entropies_s1=entropies_s1, entropies_s2=entropies_s2, smooth_s2=smooth_s2, mt_s2=mt_s2, fun_word=fun_word, full_attention=full_attention, attention_to_context=attention_to_context, atten_dispersion=atten_dispersion)

def join_if_nonempty(*li, sep=' '):
    return sep.join([s for s in li if len(s) > 0])

def match(word: str, real_words):
    for real_word in real_words:
        if real_word in word:
            return True
    return False

def get_top_sentence(text):
    prev = ''
    for sent in nlp(text).sents:
        prev += sent.text
        sent = sent.text.strip()
        if len(sent) > 0:
            return prev
    return ''
_attention_meta_cache = {}

def save_attention_matrix(outputs: GeneratorOutput, qid: str, step_idx: int, output_dir: str):
    if outputs.empty or outputs.full_attention is None:
        return
    os.makedirs(output_dir, exist_ok=True)
    temp_file = os.path.join(output_dir, f'{qid}_step{step_idx}_temp.npz')
    np.savez_compressed(temp_file, attention=outputs.full_attention.detach().numpy(), context=outputs.attention_to_context.detach().numpy())
    if qid not in _attention_meta_cache:
        _attention_meta_cache[qid] = {'steps': [], 'temp_files': []}
    new_block = outputs.blocks[-1]
    words_data = []
    for i in range(new_block.len_words):
        tl, tr = new_block.range_[i]
        word = outputs.merged_blocks.tokens[tl:tr]
        word_str = ''.join(word).replace('▁', ' ').replace('Ġ', ' ').strip()
        words_data.append({'idx': i, 'word': word_str, 'max_atten': outputs.max_atten[i].item(), 'entropy': outputs.entropies[i].item(), 'mt_s2': outputs.mt_s2[i]['val'].item(), 'fun_word': outputs.fun_word[i]['val'].item()})
    _attention_meta_cache[qid]['steps'].append({'step_idx': step_idx, 'new_text': outputs.new_text, 'words': words_data})
    _attention_meta_cache[qid]['temp_files'].append(temp_file)

def flush_attention_cache(qid: str, output_dir: str):
    if qid not in _attention_meta_cache:
        return
    cache = _attention_meta_cache[qid]
    npz_file = os.path.join(output_dir, f'{qid}.npz')
    save_dict = {}
    for i, temp_file in enumerate(cache['temp_files']):
        if os.path.exists(temp_file):
            data = np.load(temp_file)
            step_idx = cache['steps'][i]['step_idx']
            save_dict[f'step{step_idx}_attention'] = data['attention']
            save_dict[f'step{step_idx}_context'] = data['context']
            data.close()
            os.remove(temp_file)
    if save_dict:
        np.savez_compressed(npz_file, **save_dict)
    meta_file = os.path.join(output_dir, f'{qid}_meta.json')
    meta = {'qid': qid, 'num_steps': len(cache['steps']), 'steps': cache['steps']}
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False)
    del _attention_meta_cache[qid]
    print(f"Saved attention data for {qid} ({len(meta['steps'])} steps)")

def compute_attention_dispersion(attention_to_context: Tensor) -> dict:
    results = []
    for i in range(attention_to_context.shape[0]):
        atten_dist = attention_to_context[i]
        atten_dist = atten_dist / (atten_dist.sum() + 1e-10)
        entropy = -torch.sum(atten_dist * torch.log(atten_dist + 1e-10)).item()
        max_atten = atten_dist.max().item()
        concentration = max_atten
        top5_sum = torch.topk(atten_dist, min(5, len(atten_dist))).values.sum().item()
        sorted_atten = torch.sort(atten_dist).values
        n = len(sorted_atten)
        cum_atten = torch.cumsum(sorted_atten, dim=0)
        gini = (n + 1 - 2 * cum_atten.sum() / cum_atten[-1]) / n
        results.append({'entropy': entropy, 'concentration': concentration, 'top5_ratio': top5_sum, 'gini': gini.item() if isinstance(gini, Tensor) else gini})
    return results

@dataclass
class CheckerOutput:
    hallucination: bool
    curr_st: int = None
    curr_en: int = None
    curr_thres: List[bool] = None

class ETC:

    def __init__(self, args):
        for k, v in args.__dict__.items():
            setattr(self, k, v)
        self.generator = Generator(self.model_name_or_path)
        self.tokenizer = self.generator.tokenizer
        self.model = self.generator.model
        self.model: PreTrainedModel
        self.retriever = BM25('wiki' if 'es_index_name' not in args.__dict__ else self.es_index_name)
        self.counter = Counter()

    def hallucination_check(self, outputs: GeneratorOutput) -> CheckerOutput:
        if DEBUG:
            print('Start detecting hallucinations')
        new_block = outputs.blocks[-1]
        sentences = [sent.text.strip() for sent in nlp(new_block.text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        if DEBUG:
            print('Clauses')
            for i, sent in enumerate(sentences):
                print(f'sentence{i}：{sent}')
        wid = 0
        word_counts = [0] * len(sentences)
        thres_sum = []
        for sid, sent in enumerate(sentences):
            wl, wr = (wid, wid)
            if wid == new_block.len_words:
                break
            while wr < new_block.len_words and sent not in self.tokenizer.convert_tokens_to_string(new_block.tokens[new_block.range_[wl][0]:new_block.range_[wr][1]]):
                wr += 1
            if wr < new_block.len_words:
                wr += 1
            wid = wr
            len_sent = wid
            if wl == wr:
                continue
            if sid == 0:
                word_counts[sid] = wid
            else:
                for t in range(0, sid):
                    len_sent -= word_counts[t]
                word_counts[sid] = len_sent
            print('Current sentence length:', word_counts[sid])
            index_sent = 0
            for j in range(0, sid):
                index_sent += word_counts[j]
            if DEBUG:
                print('Current sentence:', self.tokenizer.convert_tokens_to_string(new_block.tokens[new_block.range_[wl][0]:new_block.range_[wr - 1][1]]), sep='\n')
            dispersion_sent = outputs.atten_dispersion[wl:wr]
            max_atten_sent = outputs.max_atten[wl:wr]
            max_atten_sent = max_atten_sent * (wr - wl) / (max_atten_sent.sum() + 1e-10)
            value = dispersion_sent * torch.tensor([entry['val'] for entry in outputs.mt_s2[wl:wr]]).to(dispersion_sent.device)
            value_original = max_atten_sent * torch.tensor([entry['val'] for entry in outputs.mt_s2[wl:wr]]).to(max_atten_sent.device)
            thres_abs = self.thres_abs
            if thres_abs == True:
                thres = torch.abs(value) > self.hallucination_threshold
            else:
                thres = value > self.hallucination_threshold
            thres_sum.append(thres)
            if DEBUG:
                print('wid|word|dispersion|max_atten|entropy|s1|s2|mt_s2|value_new|value_orig|thres：')
                for i in range(wl, wr):
                    print(i, self.tokenizer.convert_tokens_to_string(new_block.tokens[new_block.range_[i][0]:new_block.range_[i][1]]), dispersion_sent[i - wl].item(), max_atten_sent[i - wl].item(), outputs.entropies[i - wl].item(), outputs.entropies_s1[i]['val'].item(), outputs.entropies_s2[i]['val'].item(), outputs.mt_s2[i]['val'].item(), value[i - wl].item(), value_original[i - wl].item(), thres[i - wl].item(), sep='|')
            if True in thres:
                for i in range(wl, wr):
                    if thres[i - wl].item() == True:
                        count_k_2 = 0
                        j = i - 1
                        while j >= 0 and count_k_2 < 2:
                            if outputs.fun_word[j]['val'].item() != 0:
                                count_k_2 += 1
                            if count_k_2 == 2:
                                break
                            else:
                                j -= 1
                        return CheckerOutput(hallucination=True, curr_st=i, curr_en=wr, curr_thres=thres[i - wl:wr])
            if DEBUG:
                print('No hallucinations were detected in the current sentence. Prepare for the next sentence.')
        return CheckerOutput(hallucination=False)

    def generate_retrieve_qry(self, outputs: GeneratorOutput, check_info: CheckerOutput):
        ques_st = outputs.blocks[0].len_words + outputs.blocks[1].len_words
        ques_en = ques_st + outputs.blocks[2].len_words
        question_words = []
        for i in range(ques_st, ques_en):
            tl, tr = outputs.merged_blocks.range_[i]
            word = self.tokenizer.convert_tokens_to_string(outputs.merged_blocks.tokens[tl:tr])
            question_words.append(word)
        print('question', ' '.join(question_words))
        text_st = ques_en + outputs.blocks[3].len_words
        text_en = text_st + outputs.blocks[4].len_words + check_info.curr_st
        ques_atten = outputs.atten[check_info.curr_st:check_info.curr_en, ques_st:ques_en]
        text_atten = outputs.atten[check_info.curr_st:check_info.curr_en, text_st:text_en]
        print('ques_atten.shape:', ques_atten.shape)
        print('text_atten.shape:', text_atten.shape)
        print(check_info.curr_thres.shape)
        ques_atten = ques_atten[check_info.curr_thres, :].sum(dim=0)
        text_atten = text_atten[check_info.curr_thres, :].sum(dim=0)
        doc = nlp(outputs.merged_blocks.text)
        real_words = set((token.text for token in doc if token.pos_ in ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM']))
        real_pairs = []
        for i in range(ques_st, ques_en):
            a = ques_atten[i - ques_st]
            tl, tr = outputs.merged_blocks.range_[i]
            word = self.tokenizer.convert_tokens_to_string(outputs.merged_blocks.tokens[tl:tr])
            if match(word, real_words):
                real_pairs.append((a, word, i))
        for i in range(text_st, text_en):
            a = text_atten[i - text_st]
            tl, tr = outputs.merged_blocks.range_[i]
            word = self.tokenizer.convert_tokens_to_string(outputs.merged_blocks.tokens[tl:tr])
            if match(word, real_words):
                real_pairs.append((a, word, i))
        if 'retrieve_keep_top_k' in self.__dict__:
            top_k = min(self.retrieve_keep_top_k, len(real_pairs))
        elif 'retrieve_keep_ratio' in self.__dict__:
            top_k = int(len(real_pairs) * self.retrieve_keep_ratio)
        real_pairs.sort(key=lambda x: -x[0])
        real_pairs = real_pairs[:top_k]
        real_pairs.sort(key=lambda x: x[2])
        return ' '.join([x[1] for x in real_pairs])

    def inference(self, question, demo, case):
        text = ''
        demo = '\n'.join([d['case'] for d in demo])
        if DEBUG:
            print('Begin reasoning')
        while True:
            old_len = len(text)
            try:
                outputs = self.generator.generate(input_texts=[demo, '\nQuestion:', question, '\nAnswer:', text], max_length=self.generate_max_length)
            except torch.cuda.OutOfMemoryError:
                logger.warning(f'OOM in generate()! Partial text len={len(text)}')
                torch.cuda.empty_cache()
                raise OOMError(text)
            if DEBUG:
                if outputs.empty == False:
                    print('Initial generation of new text', outputs.new_text, sep='\n')
                    if self.use_counter == True:
                        self.counter.add_generate(outputs.new_text, self.generator.tokenizer)
            if outputs.empty == True:
                if DEBUG:
                    print('If only blank characters are detected, the generation process will be interrupted.')
                break
            check_info = self.hallucination_check(outputs)
            if not check_info.hallucination:
                if DEBUG:
                    print('No hallucinations')
                text = join_if_nonempty(text, outputs.new_text.strip())
                if DEBUG:
                    print('Currently generated text', text, sep='\n')
                if outputs.ended or outputs.merged_blocks.len_tokens > self.generate_max_length:
                    if DEBUG:
                        if outputs.ended:
                            print('Terminator detected.' if outputs.ended else 'The text has reached its maximum length.')
                    break
            else:
                if DEBUG:
                    print('Hallucination detected. Preparing to retrieve information.')
                retrieve_qry = self.generate_retrieve_qry(outputs, check_info)
                if DEBUG:
                    print(f'retrieve_qry: {retrieve_qry}')
                docs = self.retriever(retrieve_qry, topk=self.retrieve_topk)
                self.counter.retrieve += 1
                prompt = demo
                prompt += '\nContext:\n'
                for i, doc in enumerate(docs):
                    print(f'doc{i}:{doc}')
                    prompt += f'[{i + 1}] {doc}\n'
                prompt += 'Answer in the same format as before.\n'
                for i in [1, 2, 3]:
                    prompt += outputs.blocks[i].text
                text = self.tokenizer.convert_tokens_to_string(outputs.blocks[-2].tokens + outputs.blocks[-1].tokens[:outputs.blocks[-1].range_[check_info.curr_st][0]])
                prompt += text
                try:
                    ended, new_texts = self.generator.simply_generate(prompt, max_length=self.generate_max_length)
                except torch.cuda.OutOfMemoryError:
                    logger.warning(f'OOM in simply_generate()! Partial text len={len(text)}')
                    torch.cuda.empty_cache()
                    raise OOMError(text)
                if self.use_counter == True:
                    self.counter.add_generate(new_texts, self.generator.tokenizer)
                    self.counter.hallucinated += 1
                new_text = get_top_sentence(new_texts)
                text = join_if_nonempty(text, new_text.strip())
                if DEBUG:
                    print('Regenerate new text:', new_text, sep='\n')
                if DEBUG:
                    print('The text currently generated is:', text, sep='\n')
                if ended and len(new_text) >= len(new_texts.strip()):
                    if DEBUG:
                        print('Terminator detected.')
                    break
                if len(self.tokenizer.encode(text)) > self.generate_max_length:
                    if DEBUG:
                        print('The text has reached its maximum length.')
                    break
            if old_len >= len(text):
                logger.info('old_len >= len(text) !')
                break
        if DEBUG:
            print('finished', text, sep='\n')
        return text
