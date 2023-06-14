import argparse

import torch
from coati.models.bloom import BLOOMActor
from coati.models.gpt import GPTActor
from coati.models.opt import OPTActor
from coati.models.roberta import RoBERTaActor
from transformers import AutoTokenizer, RobertaTokenizer
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer


def eval(args):
    # configure model
    if args.model == 'gpt2':
        actor = GPTActor(pretrained=args.pretrain).to(torch.cuda.current_device())
    elif args.model == 'bloom':
        actor = BLOOMActor(pretrained=args.pretrain).to(torch.cuda.current_device())
    elif args.model == 'opt':
        actor = OPTActor(pretrained=args.pretrain).to(torch.cuda.current_device())
    elif args.model == 'roberta':
        actor = RoBERTaActor(pretrained=args.pretrain).to(torch.cuda.current_device())
    else:
        raise ValueError(f'Unsupported model "{args.model}"')
    
    if args.model_path is not None:
        state_dict = torch.load(args.model_path)
        actor.model.load_state_dict(state_dict, strict=False)

    # configure tokenizer
    if args.model == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == 'bloom':
        tokenizer = AutoTokenizer.from_pretrained('bigscience/bloom-560m')
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == 'opt':
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    elif args.model == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    else:
        raise ValueError(f'Unsupported model "{args.model}"')

    actor.eval()
    #input = args.input
    input = "i like to remodel homes.\ni like to go hunting.\ni like to shoot a bow.\nmy favorite holiday is halloween.\nhi , how are you doing ? i\'m getting ready to do some cheetah chasing to stay in shape .\n"
    input_ids = tokenizer.encode(input, return_tensors='pt').to(torch.cuda.current_device())
    outputs = actor.generate(input_ids,
                             max_length=args.max_length,
                             do_sample=True,
                             top_k=50,
                             top_p=0.95,
                             num_return_sequences=1)
    output = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)
    print(output[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt2', choices=['gpt2', 'bloom', 'opt', 'roberta'])
    # We suggest to use the pretrained model from HuggingFace, use pretrain to configure model
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--input', type=str, default='Question: How are you ? Answer:')
    parser.add_argument('--max_length', type=int, default=100)
    parser.add_argument('--tokenizer', type=str, default=None)
    args = parser.parse_args()
    eval(args)