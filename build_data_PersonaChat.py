import random
from tqdm import tqdm
import torch
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import json



def get_token_id(tokenizer):
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    sep_id = tokenizer.sep_token_id

    query_id, res_id, latent_id, persona_id = tokenizer.convert_tokens_to_ids(
        ['<query>', '<response>', '<latent>', '<persona>'])
    return bos_id, eos_id, pad_id, sep_id, query_id, res_id, latent_id, persona_id

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
        assert len(query) == len(response) == len(persona) == len(cand)

    if(max_datasets_size is not None):
        query = query[:max_datasets_size]
        response = response[:max_datasets_size]
        cand = cand[:max_datasets_size]
        persona = persona[:max_datasets_size]
    print("{} has {} dialog and {} query".format(data_file, len(query), sum_u))

    return persona, query, response, cand


def create_encoder_input(per, history, query_id, res_id, latent_id, persona_id, sep_id, eos_id):
    # print(sep_id, "-"*50)
    encoder_input_ids = []
    per_input_ids = [latent_id] + [persona_id]

    for x in per:
        per_input_ids += x + [sep_id]

    encoder_input_ids += per_input_ids
    for i in range(len(history)):
        if i % 2 == 0:
            encoder_input_ids += [query_id] + history[i] + [eos_id]
        else:
            #print(encoder_input_ids, [res_id], history[i], [eos_id])
            #print("-"*50)
            encoder_input_ids += [res_id] + history[i] + [eos_id]
    attention_mask = [1] * len(encoder_input_ids)
    per_attention_mask = [1] * len(per_input_ids)
    
    return encoder_input_ids, attention_mask, per_input_ids, per_attention_mask


def create_decoder_input(response_ids, res_id, eos_id, golden=None):
    assert golden != None

    decoder_lmlabel= response_ids + [eos_id]
    decoder_input_ids = [res_id] + response_ids
    decoder_cls_index = [-100] * (len(decoder_lmlabel) - 1) + [eos_id]
    decoder_attention_mask = [1] * len(decoder_input_ids)


    if golden == False:
        decoder_lmlabel = [-100] * len(decoder_lmlabel)

    assert len(decoder_lmlabel) == len(decoder_input_ids)

    return decoder_lmlabel, decoder_input_ids, decoder_cls_index, decoder_attention_mask



def build_dataloader(persona, query, response, cand, tokenizer, max_history=4, n_cand=5, use_all=False):
    bos_id, eos_id, pad_id, sep_id, query_id, res_id, latent_id, persona_id = get_token_id(tokenizer)
    dataset = defaultdict(list)
    for i in range(len(persona)):
        persona_ = persona[i]
        per_list = []
        for per in persona_:
            persona_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(per))
            per_list.append(persona_ids)
        query_ = query[i]
        response_ = response[i]
        cand_ = cand[i]
        history = []
        assert len(query_) == len(response_)
        for j in range(len(query_)):
            if use_all:
                noise_candidate = cand_[j][:-1]
                #print("return cand")
            else:
                noise_candidate = random.sample(cand_[j][:-1], n_cand-1)
                #print("return answer")

            query_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(query_[j]))
            response_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(response_[j]))

            noise_cand_ids_list = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
                           for text in noise_candidate]
            history.append(query_ids)
            history.append(response_ids)

            tmp_history = history[-2 * max_history:]

        encoder_input_ids, attention_mask, per_input_ids, per_attention_mask = create_encoder_input(per_list, tmp_history, query_id, res_id,
                                                                                                     latent_id, persona_id, sep_id, eos_id)
        decoder_lmlabel, decoder_input_ids, decoder_cls_idx, decoder_attention_mask = create_decoder_input(response_ids, res_id, eos_id, golden=True)


        dataset["input_ids"].append(encoder_input_ids)
        dataset["attention_mask"].append(attention_mask)
        dataset["per_input_ids"].append(per_input_ids)
        dataset["per_attention_mask"].append(per_attention_mask)
        dataset["lmlabels"].append(decoder_lmlabel)
        dataset["decoder_input_ids"].append(decoder_input_ids)
        dataset["decoder_attention_mask"].append(decoder_attention_mask)
        dataset["cls_index"].append(decoder_cls_idx)
        dataset["clslabel"].append([0])
            
    for item_name, item in dataset.items():
        if item_name == "input_ids" or item_name == "per_input_ids":
            item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                              batch_first=True, padding_value=pad_id)

            dataset[item_name] = item
        elif item_name == "lmlabels":
            item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                batch_first=True, padding_value=pad_id)
            dataset[item_name] = item
        elif item_name == "attention_mask" or item_name == "decoder_attention_mask" or item_name == "per_attention_mask":
            item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                batch_first=True, padding_value=pad_id)
            dataset[item_name] = item
        elif item_name == "decoder_input_ids":
            item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                batch_first=True, padding_value=pad_id)
            dataset[item_name] = item
        elif item_name == "clslabel":
            dataset[item_name] = torch.tensor(item).view(-1,1)
        elif item_name == "cls_index":
            item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                batch_first=True, padding_value=pad_id)
            dataset[item_name] = item

    return dataset


def build_dataloader_step2(persona, query, response, cand, tokenizer, max_history=4, n_cand=5, use_all=False):
    bos_id, eos_id, pad_id, sep_id, query_id, res_id, latent_id, persona_id = get_token_id(tokenizer)
    dataset = defaultdict(list)
    for i in range(len(persona)):
        persona_ = persona[i]
        per_list = []
        for per in persona_:
            persona_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(per))
            per_list.append(persona_ids)
        query_ = query[i]
        response_ = response[i]
        cand_ = cand[i]
        history = []
        assert len(query_) == len(response_)
        for j in range(len(query_)):
            if use_all:
                noise_candidate = cand_[j][:-1]
                #print("return cand")
            else:
                noise_candidate = random.sample(cand_[j][:-1], n_cand-1)
                #print("return answer")

            query_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(query_[j]))
            response_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(response_[j]))

            noise_cand_ids_list = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
                           for text in noise_candidate]
            history.append(query_ids)
            history.append(response_ids)
            
            for k in range(len(noise_cand_ids_list)):
                tmp_history = history[-2 * max_history:-1] + [noise_cand_ids_list[k]]
                #print(tmp_history)
                encoder_input_ids_chosen, attention_mask_chosen, _, _ = create_encoder_input(per_list, tmp_history, query_id, res_id,
                                                                                                             latent_id, persona_id, sep_id, eos_id)

                tmp_history = history[-2 * max_history:]
                encoder_input_ids_rejected, attention_mask_rejected, _, _ = create_encoder_input(per_list, tmp_history, query_id, res_id,
                                                                                                             latent_id, persona_id, sep_id, eos_id)

        dataset["input_ids_chosen"].append(encoder_input_ids_chosen)
        dataset["attention_mask_chosen"].append(attention_mask_chosen)
        
        dataset["input_ids_rejected"].append(encoder_input_ids_rejected)
        dataset["attention_mask_rejected"].append(attention_mask_rejected)

        
    for item_name, item in dataset.items():
        if item_name == "input_ids_rejected" or item_name == "input_ids_chosen":
            item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                              batch_first=True, padding_value=pad_id)

            dataset[item_name] = item

        elif item_name == "attention_mask_rejected" or item_name == "attention_mask_chosen":
            item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                batch_first=True, padding_value=pad_id)
            dataset[item_name] = item

    return dataset
        
        
def build_dataloader_step3_prompt(persona, query, response, cand, tokenizer, max_history=4, n_cand=5, use_all=False):
    bos_id, eos_id, pad_id, sep_id, query_id, res_id, latent_id, persona_id = get_token_id(tokenizer)
    dataset = defaultdict(list)
    for i in range(len(persona)):
        persona_ = persona[i]
        per_list = []
        for per in persona_:
            persona_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(per))
            per_list.append(persona_ids)
        query_ = query[i]
        response_ = response[i]
        cand_ = cand[i]
        history = []
        assert len(query_) == len(response_)
        for j in range(len(query_)):
            if use_all:
                noise_candidate = cand_[j][:-1]
                #print("return cand")
            else:
                noise_candidate = random.sample(cand_[j][:-1], n_cand-1)
                #print("return answer")

            query_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(query_[j]))
            response_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(response_[j]))

            noise_cand_ids_list = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
                           for text in noise_candidate]
            history.append(query_ids)
            history.append(response_ids)

            tmp_history = history[-2 * max_history:-1]

            encoder_input_ids, attention_mask, per_input_ids, per_attention_mask = create_encoder_input(per_list, tmp_history, query_id, res_id,
                                                                                                         latent_id, persona_id, sep_id, eos_id)
            decoder_lmlabel, decoder_input_ids, decoder_cls_idx, decoder_attention_mask = create_decoder_input(response_ids, res_id, eos_id, golden=True)


        dataset["input_ids"].append(encoder_input_ids)
        dataset["attention_mask"].append(attention_mask)
        dataset["per_input_ids"].append(per_input_ids)
        dataset["per_attention_mask"].append(per_attention_mask)
        dataset["lmlabels"].append(decoder_lmlabel)
        dataset["decoder_input_ids"].append(decoder_input_ids)
        dataset["decoder_attention_mask"].append(decoder_attention_mask)
        dataset["cls_index"].append(decoder_cls_idx)
        dataset["clslabel"].append([0])
            
    for item_name, item in dataset.items():
        if item_name == "input_ids" or item_name == "per_input_ids":
            item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                              batch_first=True, padding_value=pad_id)

            dataset[item_name] = item
        elif item_name == "lmlabels":
            item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                batch_first=True, padding_value=pad_id)
            dataset[item_name] = item
        elif item_name == "attention_mask" or item_name == "decoder_attention_mask" or item_name == "per_attention_mask":
            item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                batch_first=True, padding_value=pad_id)
            dataset[item_name] = item
        elif item_name == "decoder_input_ids":
            item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                batch_first=True, padding_value=pad_id)
            dataset[item_name] = item
        elif item_name == "clslabel":
            dataset[item_name] = torch.tensor(item).view(-1,1)
        elif item_name == "cls_index":
            item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                batch_first=True, padding_value=pad_id)
            dataset[item_name] = item

    return dataset
        
            
    for item_name, item in dataset.items():
        if item_name == "input_ids_rejected" or item_name == "input_ids_chosen":
            item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                              batch_first=True, padding_value=pad_id)

            dataset[item_name] = item

        elif item_name == "attention_mask_rejected" or item_name == "attention_mask_chosen":
            item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                batch_first=True, padding_value=pad_id)
            dataset[item_name] = item

    return dataset


def build_dataloader_step3_pertrain(persona, query, response, cand, tokenizer, max_history=4, n_cand=5, use_all=False):
    bos_id, eos_id, pad_id, sep_id, query_id, res_id, latent_id, persona_id = get_token_id(tokenizer)
    dataset = defaultdict(list)
    for i in range(len(persona)):
        persona_ = persona[i]
        per_list = []
        for per in persona_:
            persona_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(per))
            per_list.append(persona_ids)
        query_ = query[i]
        response_ = response[i]
        cand_ = cand[i]
        history = []
        assert len(query_) == len(response_)
        for j in range(len(query_)):
            if use_all:
                noise_candidate = cand_[j][:-1]
                #print("return cand")
            else:
                noise_candidate = random.sample(cand_[j][:-1], n_cand-1)
                #print("return answer")

            query_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(query_[j]))
            response_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(response_[j]))

            noise_cand_ids_list = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
                           for text in noise_candidate]
            history.append(query_ids)
            history.append(response_ids)

            tmp_history = history[-2 * max_history:]

        encoder_input_ids, attention_mask, per_input_ids, per_attention_mask = create_encoder_input(per_list, tmp_history, query_id, res_id,
                                                                                                     latent_id, persona_id, sep_id, eos_id)
        decoder_lmlabel, decoder_input_ids, decoder_cls_idx, decoder_attention_mask = create_decoder_input(response_ids, res_id, eos_id, golden=True)

        per_input_ids = [pad_id]*len(encoder_input_ids)
        per_input_ids = per_input_ids[:len(encoder_input_ids)-len(decoder_lmlabel)] + decoder_lmlabel

        dataset["input_ids"].append(encoder_input_ids)
        dataset["attention_mask"].append(attention_mask)
        dataset["per_input_ids"].append(per_input_ids)
        dataset["per_attention_mask"].append(per_attention_mask)
        dataset["lmlabels"].append(decoder_lmlabel)
        dataset["decoder_input_ids"].append(decoder_input_ids)
        dataset["decoder_attention_mask"].append(decoder_attention_mask)
        dataset["cls_index"].append(decoder_cls_idx)
        dataset["clslabel"].append([0])
            
    for item_name, item in dataset.items():
        if item_name == "input_ids" or item_name == "per_input_ids":
            item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                              batch_first=True, padding_value=pad_id)

            dataset[item_name] = item
        elif item_name == "lmlabels":
            item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                batch_first=True, padding_value=pad_id)
            dataset[item_name] = item
        elif item_name == "attention_mask" or item_name == "decoder_attention_mask" or item_name == "per_attention_mask":
            item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                batch_first=True, padding_value=pad_id)
            dataset[item_name] = item
        elif item_name == "decoder_input_ids":
            item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                batch_first=True, padding_value=pad_id)
            dataset[item_name] = item
        elif item_name == "clslabel":
            dataset[item_name] = torch.tensor(item).view(-1,1)
        elif item_name == "cls_index":
            item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                batch_first=True, padding_value=pad_id)
            dataset[item_name] = item

    return dataset


def build_infer_dataset(tokenizer, file_path):
    bos_id, eos_id, pad_id, sep_id, query_id, res_id, latent_id, persona_id = get_token_id(tokenizer)
    positive_set = defaultdict(list)

    with open(file_path, "r") as f:
        row_data = json.load(f)
        for obj in tqdm(row_data, desc='Generate infer data'):
            if obj['label'] == "neutral" or obj['label'] == "negative":
                continue
            pre = obj["sentence1"].lower()
            hyp = obj["sentence2"].lower()
            pre_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(pre))
            hyp_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(hyp))
            encoder_input_ids = [latent_id] + [persona_id] + pre_ids + [eos_id]
            attention_mask = [1] * len(encoder_input_ids)

            decoder_input_ids = [res_id] + hyp_ids
            decoder_attention_mask = [1] * len(decoder_input_ids)

            if obj['label'] == 'positive':
                positive_set["encoder_input_ids"].append(encoder_input_ids)
                positive_set["decoder_input_ids"].append(decoder_input_ids)
                positive_set["attention_mask"].append(attention_mask)
                positive_set["decoder_attention_mask"].append(decoder_attention_mask)
                decoder_lmlabel = hyp_ids + [eos_id]
                positive_set["lmlabels"].append(decoder_lmlabel)



        for item_name, item in positive_set.items():
            if item_name == "encoder_input_ids":
                item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                    batch_first=True, padding_value=pad_id)

                positive_set[item_name] = item
            elif item_name == "lmlabels":
                item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                    batch_first=True, padding_value=-100)
                positive_set[item_name] = item
            elif item_name == "attention_mask" or item_name == "decoder_attention_mask":
                item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                    batch_first=True, padding_value=0)
                positive_set[item_name] = item
            elif item_name == "decoder_input_ids":
                item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                    batch_first=True, padding_value=pad_id)
                positive_set[item_name] = item

    return positive_set


def pad_dataset(dataset, pad_id):
    for item_name, item in dataset.items():
        if item_name == "input_ids" or item_name == "per_input_ids":
            item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                              batch_first=True, padding_value=pad_id)

            dataset[item_name] = item
        elif item_name == "lmlabels":
            item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                batch_first=True, padding_value=pad_id)
            dataset[item_name] = item
        elif item_name == "attention_mask" or item_name == "decoder_attention_mask" or item_name == "per_attention_mask":
            item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                batch_first=True, padding_value=pad_id)
            dataset[item_name] = item
        elif item_name == "decoder_input_ids":
            item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                batch_first=True, padding_value=pad_id)
            dataset[item_name] = item
        elif item_name == "clslabel":
            dataset[item_name] = torch.tensor(item).view(-1,1)
        elif item_name == "cls_index":
            item = pad_sequence([torch.from_numpy(np.array(x)) for x in item],
                                batch_first=True, padding_value=pad_id)
            dataset[item_name] = item


    return dataset




