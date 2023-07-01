import copy
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Sequence

import torch
import torch.distributed as dist
import transformers
from torch.utils.data import Dataset
from tqdm import tqdm

from colossalai.logging import get_dist_logger
import itertools

from collections import defaultdict
from colossalai.logging import get_dist_logger
import io
import json
logger = get_dist_logger()

def is_rank_0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

IGNORE_INDEX = -100
PROMPT_DICT = {
    "prompt_input":
        ("Below is an instruction that describes a task, paired with an input that provides further context. "
         "Write a response that appropriately completes the request.\n\n"
         "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"),
    "prompt_no_input": ("Below is an instruction that describes a task. "
                        "Write a response that appropriately completes the request.\n\n"
                        "### Instruction:\n{instruction}\n\n### Response:"),
}

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    max_length: int,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer, max_length) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, max_length: int) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


"""Prompt"""
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, max_datasets_size: int = None, max_length: int = 512):
        super(SupervisedDataset, self).__init__()
        logger.info("Loading data...")
        list_data_dict = jload(data_path)
        logger.info(f"Loaded {len(list_data_dict)} examples.")
        print(list_data_dict[:2])
        if max_datasets_size is not None:
            logger.info(f"Limiting dataset to {max_datasets_size} examples.")
            list_data_dict = list_data_dict[:max_datasets_size]

        logger.info("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logger.info("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer, max_length)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        #print(self.input_ids[:2])
        #print(self.labels[:2])
        #print("-"*50)
        
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


def PersonaPromptOnlyProcess(data_file, max_datasets_size):
    dataset_prompt = []
    
    persona, query, response, _ = create_data(data_file, max_datasets_size)
    # print('persona', len(persona), max_datasets_size)
    for i in range(len(persona)):
        dialogue = []
        dialogue.append(query[i][0])

        dataset_prompt.append( "\n".join(persona[i] + [query[i][0]] ))
        #print(len(query[i])-1)
        for j in range(len(query[i])-1):
            dialogue.append(response[i][j])
            dialogue.append(query[i][j+1])

            dataset_prompt.append( "\n".join(persona[i] + dialogue ))
    #print(len(dataset_prompt))
    return dataset_prompt

class PersonaPromptOnlyDataset(Dataset):
    
    def __init__(self,
                 data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_datasets_size: int = None,
                 max_length: int = 96):
        super(PersonaPromptOnlyDataset, self).__init__()
        self.keyed_prompt = defaultdict(list)
        logger.info("Loading data...")
        list_data_dict = PersonaPromptOnlyProcess(data_path, max_datasets_size)
        logger.info(f"Loaded {len(list_data_dict)} examples.")

        if max_datasets_size is not None:
            logger.info(f"Limiting dataset to {max_datasets_size} examples.")
            list_data_dict = list_data_dict[:max_datasets_size]
        # print('-'*50, len(list_data_dict))
        for data_dict in tqdm(list_data_dict):
            #print(data_dict)
            #print(type(data_dict))
            #print("-"*50)
            token = tokenizer(data_dict,
                              return_tensors='pt',
                              max_length=max_length,
                              padding='max_length',
                              truncation=True)
            for k, tensor in token.items():
                self.keyed_prompt[k].extend(tensor.to(torch.cuda.current_device()).unbind())

    def __len__(self):
        return len(self.keyed_prompt["input_ids"])

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return {k: v[i] for k, v in self.keyed_prompt.items()}

    

class PromptDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self,
                 data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_datasets_size: int = None,
                 max_length: int = 96):
        super(PromptDataset, self).__init__()
        self.keyed_prompt = defaultdict(list)
        logger.info("Loading data...")
        list_data_dict = jload(data_path)
        logger.info(f"Loaded {len(list_data_dict)} examples.")

        if max_datasets_size is not None:
            logger.info(f"Limiting dataset to {max_datasets_size} examples.")
            list_data_dict = list_data_dict[:max_datasets_size]

        # print('-'*50, len(list_data_dict))
        for data_dict in list_data_dict:
            #print(data_dict["instruction"])
            #print(type(data_dict["instruction"]))
            #print("-"*50)
            token = tokenizer(data_dict["instruction"],
                              return_tensors='pt',
                              max_length=max_length,
                              padding='max_length',
                              truncation=True)
            for k, tensor in token.items():
                self.keyed_prompt[k].extend(tensor.to(torch.cuda.current_device()).unbind())

    def __len__(self):
        return len(self.keyed_prompt["input_ids"])

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return {k: v[i] for k, v in self.keyed_prompt.items()}


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, max_datasets_size: int = None, max_length: int = 512):
        super(SupervisedDataset, self).__init__()
        logger.info("Loading data...")
        list_data_dict = jload(data_path)
        logger.info(f"Loaded {len(list_data_dict)} examples.")
        print(list_data_dict[:2])
        if max_datasets_size is not None:
            logger.info(f"Limiting dataset to {max_datasets_size} examples.")
            list_data_dict = list_data_dict[:max_datasets_size]

        logger.info("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logger.info("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer, max_length)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        #print(self.input_ids[:2])
        #print(self.labels[:2])
        #print("-"*50)
        
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


"""reward"""
def PersonaPromptProcess(data_file, max_datasets_size):
    dataset_chosen = []
    dataset_rejected = []
    
    persona, query, response, cand = create_data(data_file, max_datasets_size)

    for i in range(len(persona)):
        dialogue = []
        dialogue.append(query[i][0])

        dataset_chosen.append( "\n".join(persona[i] + [query[i][0]] + [response[i][0]] ))
        dataset_rejected.append( "\n".join(persona[i] + [query[i][0]] + [cand[i][0][0]] ))

        for j in range(len(query[i])-1):
            dialogue.append(response[i][j])
            dialogue.append(query[i][j+1])

            dataset_chosen.append( "\n".join(persona[i] + dialogue + [response[i][j+1]] ))
            dataset_rejected.append( "\n".join(persona[i] + dialogue + [cand[i][j+1][0]] ))

    return dataset_chosen, dataset_rejected


class PersonaPromptDataset(Dataset):
    def __init__(self, dataset_chosen, dataset_rejected, dataset_chosen_mask, dataset_rejected_mask, max_length: int = 512, ) -> None:
        super().__init__()
        self.chosen = dataset_chosen
        self.reject = dataset_rejected
        self.chosen_mask = dataset_chosen_mask
        self.reject_mask = dataset_rejected_mask
        
    def __len__(self):
        length = len(self.chosen)
        return length

    def __getitem__(self, idx):
        return self.chosen[idx], self.chosen_mask[idx], self.reject[idx], self.reject_mask[idx]


class HhRlhfDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        tokenizer: tokenizer for reward model
        max_length: max length of input
        special_token: special token at the end of sentence
    """

    def __init__(self, dataset, tokenizer: Callable, max_length: int, special_token=None) -> None:
        super().__init__()
        self.chosen = []
        self.reject = []
        if special_token is None:
            self.end_token = tokenizer.eos_token
        else:
            self.end_token = special_token
        for data in tqdm(dataset):
            # print(data['chosen'])
            # print(type(data['chosen']))
            # print("-"*25)
            chosen = data['chosen'] + self.end_token
            chosen_token = tokenizer(chosen,
                                     max_length=max_length,
                                     padding="max_length",
                                     truncation=True,
                                     return_tensors="pt")
            self.chosen.append({
                "input_ids": chosen_token['input_ids'],
                "attention_mask": chosen_token['attention_mask']
            })
            # print(data['rejected'])
            # print("-"*50)
            reject = data['rejected'] + self.end_token
            reject_token = tokenizer(reject,
                                     max_length=max_length,
                                     padding="max_length",
                                     truncation=True,
                                     return_tensors="pt")
            self.reject.append({
                "input_ids": reject_token['input_ids'],
                "attention_mask": reject_token['attention_mask']
            })
    
        #print(self.chosen[:2])
        #print(tokenizer.batch_decode(self.chosen[0]['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
        #print(self.reject[:2])
        #print(tokenizer.batch_decode(self.reject[0]['input_ids'], skip_special_tokens=True,  clean_up_tokenization_spaces=False)[0])
        #print('inputids length: ', len(self.chosen))
        #print('outputids length: ', len(self.reject))
        #print(len(self.input_ids[0]))
        #print(len(self.labels[0]))
        
    def __len__(self):
        length = len(self.chosen)
        return length

    def __getitem__(self, idx):
        return self.chosen[idx]["input_ids"], self.chosen[idx]["attention_mask"], self.reject[idx][
            "input_ids"], self.reject[idx]["attention_mask"]



"""STF"""
def PersonaPretrainProcess(data_file, max_datasets_size):
    inputs = []
    outputs = []
    persona, query, response, _ = create_data(data_file, max_datasets_size)
    #print('persona', len(persona), max_datasets_size)
    #print(query[0][:10])
    #print(response[0][:10])
    for i in range(len(persona)):
        dialogue = []
        dialogue.append(query[i][0])
        inputs.append("\n".join(persona[i]+[query[i][0]]))
        outputs.append(response[i][0])

        for j in range(len(query[i])-1):
            dialogue.append(response[i][j])
            dialogue.append(query[i][j+1])
            inputs.append("\n".join(persona[i] + dialogue))
            outputs.append(response[i][j+1])
    return inputs, outputs
    
    
class PersonaPretrainDataset(Dataset):
    """
    Dataset for sft model

    Args:
        dataset: dataset for supervised model
        tokenizer: tokenizer for supervised model
        max_length: max length of input
    """

    def __init__(self, input_ids, labels_ids, max_length: int = 512) -> None:
        super().__init__()
        self.input_ids = input_ids
        self.labels_ids = labels_ids

    def __len__(self):
        length = len(self.input_ids)
        return length

    def __getitem__(self, idx):
        return dict(input_ids=self.input_ids[idx], labels=self.labels_ids[idx])

   
class SFTDataset(Dataset):
    """
    Dataset for sft model

    Args:
        dataset: dataset for supervised model
        tokenizer: tokenizer for supervised model
        max_length: max length of input
    """

    def __init__(self, dataset, tokenizer: Callable, max_length: int = 512) -> None:
        super().__init__()
        self.input_ids = []

        for data in tqdm(dataset, disable=not is_rank_0()):
            prompt = data['prompt'] + data['completion'] + tokenizer.eos_token
            prompt_token = tokenizer(prompt,
                                     max_length=max_length,
                                     padding="max_length",
                                     truncation=True,
                                     return_tensors="pt")

            self.input_ids.append(prompt_token['input_ids'][0])
        self.labels = copy.deepcopy(self.input_ids)
        
        #print(self.input_ids[:2])
        #print(self.labels[:2])
        #print(len(self.input_ids))
        #print(len(self.labels))
        #print(len(self.input_ids[0]))
        #print(len(self.labels[0]))

    def __len__(self):
        length = len(self.input_ids)
        return length

    def __getitem__(self, idx):
        return dict(input_ids=self.input_ids[idx], labels=self.labels[idx])



"""Make Dataset"""
def create_data(data_file, max_datasets_size):
    with open(data_file, "r", encoding="utf8") as f:
        persona =[]
        query = []
        response = []
        cand = []
        is_persona = False
        tmp_persona = []
        tmp_query = []
        tmp_response = []
        tmp_cand = []
        first = True
        cnt = 0
        sum_u = 0
        for line in f:
            cnt += 1
            line = line.strip()
            if "your persona: " in line:
                if not is_persona and not first:
                    query.append(tmp_query)
                    response.append(tmp_response)
                    cand.append(tmp_cand)
                    sum_u += len(tmp_query)
                    tmp_query = []
                    tmp_response = []
                    tmp_cand = []
                first = False
                is_persona = True
                line = line.split(": ", maxsplit=1)[1]
                tmp_persona.append(line)
            else:
                if is_persona:
                    persona.append(tmp_persona)
                    is_persona = False
                    tmp_persona = []
                line = line[line.find(" ")+1:]
                tmp_query.append(line.split("\t")[0])
                tmp_response.append(line.split("\t")[1])
                tmp_cand.append(line.split("\t")[3].split("|"))

        query.append(tmp_query)
        response.append(tmp_response)
        cand.append(tmp_cand)
        sum_u += len(tmp_query)
        
        if(max_datasets_size is not None):
            query = query[:max_datasets_size]
            response = response[:max_datasets_size]
            cand = cand[:max_datasets_size]
            persona = persona[:max_datasets_size]
        # print(len(query), len(response), len(persona), len(cand), )
        assert len(query) == len(response) == len(persona) == len(cand)

    print("{} has {} dialog and {} query".format(data_file, len(query), sum_u))

    return persona, query, response, cand