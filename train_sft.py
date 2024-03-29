import argparse
import os

import loralib as lora
import torch
import torch.distributed as dist
from coati.dataset import DataCollatorForSupervisedDataset, SupervisedDataset, SFTDataset
from coati.models import convert_to_lora_module
from coati.trainer import SFTTrainer
from coati.trainer.strategies import ColossalAIStrategy, DDPStrategy, NaiveStrategy
from coati.utils import prepare_llama_tokenizer_and_embedding
from datasets import load_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, BloomConfig, BloomForCausalLM, BloomTokenizerFast, LlamaConfig, LlamaForCausalLM
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from transformers.models.opt.configuration_opt import OPTConfig
from transformers.models.opt.modeling_opt import OPTForCausalLM

from colossalai.logging import get_dist_logger
from colossalai.nn.optimizer import HybridAdam
from colossalai.tensor import ColoParameter

from dataprocess import PersonaPretrainProcess, PersonaPretrainDataset
from build_data_PersonaChat import create_data, build_dataloader
from build_dstc import create_data_dstc, build_dataloader_dstc

def train(args):
    # configure strategy
    if args.strategy == 'naive':
        strategy = NaiveStrategy()
    elif args.strategy == 'ddp':
        strategy = DDPStrategy()
    elif args.strategy == 'colossalai_gemini':
        raise NotImplementedError(
            'Gemini is not supported .from_pretrained() yet. We will update this after checkpoint io is ready.')
        strategy = ColossalAIStrategy(stage=3, placement_policy='cuda')
    elif args.strategy == 'colossalai_zero2':
        strategy = ColossalAIStrategy(stage=2, placement_policy='cuda')
    elif args.strategy == 'colossalai_zero2_cpu':
        strategy = ColossalAIStrategy(stage=2, placement_policy='cpu')
    else:
        raise ValueError(f'Unsupported strategy "{args.strategy}"')


    # configure tokenizer
    if args.model == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model == 'bloom':
        tokenizer = BloomTokenizerFast.from_pretrained(args.pretrain)
        special_tokens_dict = {"sep_token": "</sep>"}
        tokenizer.add_special_tokens(special_tokens_dict)
        tokenizer.add_tokens(['<query>', '<response>', '<latent>', '<persona>'])
    elif args.model == 'opt':
        tokenizer = AutoTokenizer.from_pretrained(args.pretrain)
        if tokenizer.sep_token_id == None:
            tokenizer.bos_token = '<s>'
            special_tokens_dict = {"sep_token": "</sep>"}
            tokenizer.add_special_tokens(special_tokens_dict)
            tokenizer.add_tokens(['<query>', '<response>', '<latent>', '<persona>'])
        
    elif args.model == 'llama':
        tokenizer = AutoTokenizer.from_pretrained(args.pretrain)
        
        if tokenizer.sep_token_id == None:
            tokenizer.bos_token = '<s>'
            tokenizer.eos_token = '</s>'
            special_tokens_dict = {"sep_token": "</sep>", "pad_token": "<pad>"}
            tokenizer.add_special_tokens(special_tokens_dict)
            tokenizer.add_tokens(['<query>', '<response>', '<latent>', '<persona>'])
        
    else:
        raise ValueError(f'Unsupported model "{args.model}"')
    #tokenizer.pad_token = tokenizer.eos_token
    max_len = args.max_len

    
    # configure model
    with strategy.model_init_context():
        if args.model == 'bloom':
            model = BloomForCausalLM.from_pretrained(args.pretrain)
            model.resize_token_embeddings(len(tokenizer.get_vocab()))
            model = convert_to_lora_module(model, args.lora_rank).half().cuda()

        elif args.model == 'opt':
            model = OPTForCausalLM.from_pretrained(args.pretrain)
            model.resize_token_embeddings(len(tokenizer.get_vocab()))
            model = convert_to_lora_module(model, args.lora_rank).half().cuda()
        elif args.model == 'gpt2':
            model = convert_to_lora_module(GPT2LMHeadModel.from_pretrained(args.pretrain), args.lora_rank).half().cuda()
        elif args.model == 'llama':
            model = LlamaForCausalLM.from_pretrained(args.pretrain)
            model.resize_token_embeddings(len(tokenizer.get_vocab()))
            model = convert_to_lora_module(model, args.lora_rank).half().cuda()
        else:
            raise ValueError(f'Unsupported model "{args.model}"')
    
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
    
    
    # configure optimizer
    if args.strategy.startswith('colossalai'):
        optim = HybridAdam(model.parameters(), lr=args.lr, clipping_norm=1.0)
    else:
        optim = Adam(model.parameters(), lr=args.lr)

    logger = get_dist_logger()

    # configure dataset
    if args.dataset == 'PersonaChat':
        print('-'*50)
        print(args.revised)
        if args.revised:
            datatype = 'revised'
            print("use revised dataset")
        else:
            datatype = 'original'
            print("use oringinal dataset")
        
        logger.info("Build train data")
        persona, query, response, cand = create_data("./datasets/convai/train_self_"+datatype+".txt", args.max_datasets_size)
        train_data = build_dataloader(persona, query, response, cand, tokenizer, max_history=10, use_all=False)
        logger.info("Build valid data")
        persona, query, response, cand = create_data("./datasets/convai/valid_self_"+datatype+".txt", args.max_datasets_size)
        eval_dataset = build_dataloader(persona, query, response, cand, tokenizer, max_history=10, use_all=False)
        
        #print('-'*50)
        #print(len(train_data['input_ids']))
        #print(train_data['input_ids'][:10])
        #for i in range(20):
        #    print(tokenizer.decode(train_data['input_ids'][i]))
        train_dataset = PersonaPretrainDataset(torch.Tensor(train_data['input_ids']), torch.Tensor(train_data['input_ids']))
        eval_dataset = PersonaPretrainDataset(torch.Tensor(eval_dataset['input_ids']), torch.Tensor(eval_dataset['input_ids']))
        #print(train_dataset[0:2])
        #print('-'*50)
        
    elif args.dataset == 'dstc':
        logger.info("Build train data")
        persona, query, response = create_data_dstc("./datasets/dstc/train_set4DSTC7-AVSD.json", args.max_datasets_size)
        train_data = build_dataloader_dstc(persona, query, response, tokenizer, max_query=10)
        logger.info("Build valid data")
        persona, query, response = create_data_dstc("./datasets/dstc/valid_set4DSTC7-AVSD.json", args.max_datasets_size)
        eval_dataset = build_dataloader_dstc(persona, query, response, tokenizer, max_query=10)
        
        #print('-'*50)
        #print(len(train_data['input_ids']))
        #print(train_data['input_ids'][:10])
        #for i in range(20):
        #    print(tokenizer.decode(train_data['input_ids'][i]))
        train_dataset = PersonaPretrainDataset(torch.Tensor(train_data['input_ids']), torch.Tensor(train_data['input_ids']))
        eval_dataset = PersonaPretrainDataset(torch.Tensor(eval_dataset['input_ids']), torch.Tensor(eval_dataset['input_ids']))
        #print(train_dataset[0:2])
        #print('-'*50)
        
    elif args.dataset == 'yizhongw/self_instruct':
        train_data = load_dataset(args.dataset, 'super_natural_instructions', split='train')
        eval_data = load_dataset(args.dataset, 'super_natural_instructions', split='test')

        train_dataset = SFTDataset(train_data, tokenizer, max_len)
        eval_dataset = SFTDataset(eval_data, tokenizer, max_len)

    else:
        train_dataset = SupervisedDataset(tokenizer=tokenizer,
                                          data_path=args.dataset,
                                          max_datasets_size=args.max_datasets_size,
                                          max_length=max_len)
        eval_dataset = None
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    if dist.is_initialized() and dist.get_world_size() > 1:
        train_sampler = DistributedSampler(train_dataset,
                                           shuffle=True,
                                           seed=42,
                                           drop_last=True,
                                           rank=dist.get_rank(),
                                           num_replicas=dist.get_world_size())
        if eval_dataset is not None:
            eval_sampler = DistributedSampler(eval_dataset,
                                              shuffle=False,
                                              seed=42,
                                              drop_last=False,
                                              rank=dist.get_rank(),
                                              num_replicas=dist.get_world_size())
    else:
        train_sampler = None
        eval_sampler = None

    train_dataloader = DataLoader(train_dataset,
                                  shuffle=(train_sampler is None),
                                  sampler=train_sampler,
                                  batch_size=args.batch_size,
                                  collate_fn=data_collator,
                                  pin_memory=True)
    print('train_dataloader lenth per gpus: ', len(train_dataloader))
    '''
    for index, data in enumerate(train_dataloader):
        print(data)
        #print("-"*25)
        #print(data['input_ids'][0])
        print(tokenizer.decode(data['input_ids'][0]))
        #print(len(data['input_ids']))
        #print(len(data['input_ids'][0]))
        print("-"*25)
        #print(data['labels'][0])
        print(tokenizer.decode(data['labels'][0]))
        #print(len(data['attention_mask']))
        #print(len(data['attention_mask'][0]))
        print("-"*50)
        if(index > 10):
            break
    '''
    if eval_dataset is not None:
        eval_dataloader = DataLoader(eval_dataset,
                                     shuffle=(eval_sampler is None),
                                     sampler=eval_sampler,
                                     batch_size=args.batch_size,
                                     collate_fn=data_collator,
                                     pin_memory=True)
    else:
        eval_dataloader = None

    (model, optim) = strategy.prepare((model, optim))
    trainer = SFTTrainer(model=model,
                         strategy=strategy,
                         optim=optim,
                         train_dataloader=train_dataloader,
                         eval_dataloader=eval_dataloader,
                         max_epochs=args.max_epochs,
                         accumulation_steps=args.accumulation_steps)

    trainer.fit(logger=logger, use_wandb=args.use_wandb)

    # save model checkpoint after fitting on only rank0
    strategy.save_pretrained(model, path=args.save_path, only_rank0=True, tokenizer=tokenizer)
    # save optimizer checkpoint on all ranks
    if args.need_optim_ckpt:
        strategy.save_optimizer(trainer.optimizer,
                                'rm_optim_checkpoint_%d.pt' % (torch.cuda.current_device()),
                                only_rank0=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy',
                        choices=['naive', 'ddp', 'colossalai_gemini', 'colossalai_zero2', 'colossalai_zero2_cpu'],
                        default='colossalai_zero2')
    parser.add_argument('--model', choices=['gpt2', 'bloom', 'opt', 'llama'], default='bloom')
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--max_datasets_size', type=int, default=None)
    parser.add_argument('--save_path', type=str, default='output')
    parser.add_argument('--need_optim_ckpt', type=bool, default=False)
    parser.add_argument('--max_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--lora_rank', type=int, default=0, help="low-rank adaptation matrices rank")
    parser.add_argument('--log_interval', type=int, default=100, help="how many steps to log")
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--accumulation_steps', type=int, default=8)
    parser.add_argument('--use_wandb', default=False, action='store_true')
    parser.add_argument('--grad_checkpoint', default=False, action='store_true')
    parser.add_argument('--test', default=False, help="use samll dataset[:128]")
    parser.add_argument('--revised', action='store_true', default=False, help='use revised')
    args = parser.parse_args()
    train(args)