import random
import logging
from pprint import pformat
from collections import defaultdict
from functools import partial
import torch
from parlai.core.agents import Agent
from parlai.scripts.eval_model import setup_args as base_setup_args
from ParlAI.projects.convai2.eval_hits import eval_hits, setup_args as setup_args_hits
from ParlAI.projects.convai2.eval_f1 import eval_f1, setup_args as setup_args_f1
from transformers import BartTokenizer
from model.modeling_bart import LMEDRModel
from utils import AttrDict
#from eval_PersonaChat_build import  create_encoder_input, create_decoder_input, pad_dataset

from transformers import AutoTokenizer, LlamaForCausalLM, OPTForCausalLM, BloomForCausalLM, LlamaForSequenceClassification
#from transformers.models.opt.modeling_opt import OPTForCausalLM
from build_data_PersonaChat import create_encoder_input, create_decoder_input, pad_dataset
from coati.models import convert_to_lora_module
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '1' 

class TransformerAgent(Agent):
    @staticmethod
    def add_cmdline_args(argparser):
        agent_args = argparser.add_argument_group('Agent parameters')
        agent_args.add_argument("--model_checkpoint", type=str, default="./checkpoint/step1/epoch1", help="Path, url or short name of the model. Must be OpenAIGPT.")
        agent_args.add_argument("--max_history", type=int, default=6, help="Number of previous utterances to keep in history")
        agent_args.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
        agent_args.add_argument("--gpu", type=int, default=0)
        agent_args.add_argument("--eval_type", type=str, default="hits@1", help="hits@1, ppl or f1")
        agent_args.add_argument("--no_sample", action='store_true')
        agent_args.add_argument("--max_length", type=int, default=512)
        agent_args.add_argument("--min_length", type=int, default=1)
        agent_args.add_argument("--seed", type=int, default=0)
        agent_args.add_argument("--temperature", type=int, default=0.7)
        agent_args.add_argument("--top_k", type=int, default=20)
        agent_args.add_argument("--beam", type=int, default=1)
        agent_args.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
        agent_args.add_argument('--revised', action='store_true', default=False, help='use revised')
        #agent_args.add_argument('--model', type=str, default="")
        #agent_args.add_argument('--lora_rank', type=int, default=None)
        return argparser

    def __init__(self, opt, shared=None):
        super(TransformerAgent, self).__init__(opt, shared)

        args = AttrDict(opt)
        self.args = args

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__file__)
        self.logger.info(pformat(args))

        random.seed(args.seed)
        torch.random.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        
        self.logger.info("Get pretrained model and tokenizer")
        
        
        if args.model_checkpoint == 'persona_original' :
            self.tokenizer = BartTokenizer.from_pretrained(args.model_checkpoint)
            self.model_checkpoint = LMEDRModel.from_pretrained(args.model_checkpoint)

            self.logger.info("Build BPE prefix dictionary")
        
        else:
            #args.model_checkpoint == './checkpoint/step1/epoch5':
            self.tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
            #self.tokenizer = BartTokenizer.from_pretrained(args.model_checkpoint)
            #self.model_checkpoint = convert_to_lora_module(LlamaForCausalLM.from_pretrained(args.model_checkpoint), 16).half()
            #self.model_checkpoint = BloomForCausalLM.from_pretrained(args.model_checkpoint)
            #self.model_checkpoint = OPTForCausalLM.from_pretrained(args.model_checkpoint)
            
            self.model_checkpoint = LlamaForCausalLM.from_pretrained(args.model_checkpoint)
            #self.model_checkpoint = LMEDRModel.from_pretrained(args.model_checkpoint)
            #state_dict = torch.load('./checkpoint_opt1.3Biml/step3/epoch51/rmpersona.pt', map_location='cpu')
            #self.model_checkpoint.load_state_dict(state_dict, strict=False)
            #reward_model.to(torch.float16).to(torch.cuda.current_device())
                        
        '''
        else:
            self.model_checkpoint = shared['model']
            self.tokenizer = shared['tokenizer']
        '''
        if args.gpu != 0:
            self.model_checkpoint.to(args.gpu)
        else:
            self.model_checkpoint.to(args.device)
        
        self.query_id, self.res_id, self.latent_id, self.persona_id = self.tokenizer.convert_tokens_to_ids(['<query>', '<response>', '<latent>', '<persona>'])

        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id
        self.sep_id = self.tokenizer.sep_token_id
        #print(self.bos_id, self.eos_id, self.pad_id, self.sep_id)
        self.persona = []
        self.history = []
        self.labels = []

        self.reset()



    def observe(self, observation):
        if self.episode_done:
            self.reset()

        if self.labels:
            # Add the previous response to the history
            self.history.append(self.labels)

        if 'labels' in observation or 'eval_labels' in observation:
            text = observation.get('labels', observation.get('eval_labels', [[]]))[0]
            self.labels = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))

        if 'text' in observation:
            text = observation['text']
            for subtext in text.split('\n'):
                subtext = subtext.strip()
                if subtext.startswith('your persona:'):
                    subtext = subtext.replace('your persona:', '').strip()
                    self.persona.append(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(subtext)))
                else:
                    self.history.append(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(subtext)))

        self.history = self.history[(-2*self.args.max_history+1):]

        candidates = []
        if 'label_candidates' in observation:
            for candidate in observation['label_candidates']:
                candidates.append((self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(candidate)), candidate))
        self.candidates = candidates

        self.episode_done = observation['episode_done']
        self.observation = observation
        return observation

    def act(self):
        if self.args.eval_type == "hits@1" and len(self.candidates) > 0:
            dataset = defaultdict(list)
            for candidate, _ in self.candidates:
                encoder_input_ids, attention_mask, per_input_ids, per_attention_mask = create_encoder_input(self.persona, self.history, self.query_id,
                                                                         self.res_id, self.latent_id, self.persona_id,
                                                                         self.sep_id, self.eos_id)
                decoder_lmlabel, decoder_input_ids, decoder_cls_idx, \
                decoder_attention_mask = create_decoder_input(candidate, self.res_id, self.eos_id, golden=False)

                dataset["input_ids"].append(encoder_input_ids)
                dataset["attention_mask"].append(attention_mask)
                dataset["per_input_ids"].append(per_input_ids)
                dataset["per_attention_mask"].append(per_attention_mask)
                dataset["decoder_input_ids"].append(decoder_input_ids)
                dataset["decoder_attention_mask"].append(decoder_attention_mask)
                dataset["cls_index"].append(decoder_cls_idx)



            inputs = pad_dataset(dataset, self.pad_id)

            tensor_inputs = {}
            for input_name in ["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "cls_index", "per_input_ids", "per_attention_mask"]:
            #for input_name in ["input_ids", "attention_mask"]:
                tensor = torch.tensor(inputs[input_name], device=self.args.device if self.args.gpu ==0 else self.args.gpu)
                tensor = tensor.view((-1, len(self.candidates)) + tensor.shape[1:])
                tensor_inputs[input_name] = tensor

            self.model_checkpoint.eval()
            with torch.no_grad():
                #print(next(self.model_checkpoint.parameters()).device)
                #print(tensor_inputs)
                #print('-'*50)
                #print(**tensor_inputs)
                #print(tensor_inputs['input_ids'])
                print("-"*50)
                mc_logits = self.model_checkpoint(input_ids=tensor_inputs['input_ids'][0], attention_mask=tensor_inputs['attention_mask'][0]).logits[1]
                #mc_logits = self.model_checkpoint(**tensor_inputs).logits[1]
                print(mc_logits)
                print(len(mc_logits))
                print(len(mc_logits[0]))
            
            val, ind = torch.sort(mc_logits.view(1, -1)[0], descending=True)
            #print(val)
            #print(ind)
            #print(self.candidates[ind[0].item()])
            print('-'*50)
            
            ypred = self.candidates[ind[0].item()][1] # match
            tc = []
            for j in range(len(self.candidates)):
                tc.append(self.candidates[ind[j].item()][1])
            reply = {'text': ypred, 'text_candidates': tc}
        else:
            input_ids, attention_mask, per_input_ids, per_attention_mask = create_encoder_input(self.persona, self.history, self.query_id,
                                                                     self.res_id, self.latent_id, self.persona_id, self.sep_id, self.eos_id)
            tensor_input_ids = torch.tensor(input_ids, device=self.args.device if self.args.gpu ==0 else self.args.gpu).unsqueeze(0)
            tensor_per_input_ids = torch.tensor(per_input_ids, device=self.args.device if self.args.gpu ==0 else self.args.gpu).unsqueeze(0)
            tensor_attention_mask = torch.tensor(attention_mask, device=self.args.device if self.args.gpu ==0 else self.args.gpu).unsqueeze(0)
            tensor_per_attention_mask = torch.tensor(per_attention_mask,
                                                 device=self.args.device if self.args.gpu == 0 else self.args.gpu).unsqueeze(0)
            self.model_checkpoint.eval()
            with torch.no_grad():
                '''
                #if args.model_checkpoint == 'persona_original' :
                out_ids = self.model_checkpoint.generate(input_ids=tensor_input_ids,
                                                         attention_mask=tensor_attention_mask,
                                                         per_input_ids=tensor_per_input_ids,
                                                         per_attention_mask=tensor_per_attention_mask,
                                                        max_length=self.args.max_length, num_beams=self.args.beam)
                out_text = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True,
                                                   spaces_between_special_tokens=False,
                                                   clean_up_tokenization_spaces=(self.args.eval_type != 'f1'))
                #else:
                '''
                
                
                #print(tensor_input_ids)
                #print("-"*50)
                out_ids = self.model_checkpoint.generate(input_ids=tensor_input_ids,
                                                         attention_mask=tensor_attention_mask,
                                                        max_length=self.args.max_length,
                                                        top_k=self.args.top_k,
                                                        top_p=self.args.top_p,
                                                        num_return_sequences=1,
                                                        num_beams=self.args.beam)
                
                out_text = self.tokenizer.batch_decode([out_ids[0][len(tensor_input_ids[0])+1:]], 
                                                        skip_special_tokens=True,
                                                        spaces_between_special_tokens=False,
                                                        clean_up_tokenization_spaces=(self.args.eval_type != 'f1'))
                '''
                
                out_ids = self.model_checkpoint.generate(input_ids=tensor_input_ids,
                                                         attention_mask=tensor_attention_mask,
                                                        max_length=self.args.max_length,
                                                        top_k=self.args.top_k,
                                                        top_p=self.args.top_p,
                                                        num_return_sequences=3,
                                                        num_beams=self.args.beam)
                
                out_text = self.tokenizer.batch_decode([out_ids[0]])
                print(out_ids[0])
                print(out_text)
                print("-"*50)
                out_text = self.tokenizer.batch_decode([out_ids[2]])
                print(out_ids[2])
                print(out_text)
                print("-"*100)
                '''
            #print(tensor_input_ids)
            #print('-'*25)
            #print([out_ids[0][len(tensor_input_ids[0]):]])
            #print('-'*50)
            #print(out_ids)
            #print('-'*50)

            #print(out_text)
            #print('-'*100)
            
            ans = out_text[0].strip()
            print(ans)
            reply = {'text': ans}
        
        return reply


    def share(self):
        shared = super(TransformerAgent, self).share()
        shared['tokenizer'] = self.tokenizer
        shared['model'] = self.model_checkpoint
        return shared

    def reset(self):
        self.persona = []
        self.history = []
        self.labels = []
        self.candidates = []
        self.episode_done = True
        self.observation = None


if __name__ == '__main__':
    parser = base_setup_args(None)
    parser.add_argument("--beam", type=int, default=1)
    parser.add_argument("--model_checkpoint", type=str, default="")
    parser.add_argument('--revised', action='store_true', default=False, help='use revised')
    parser.set_params(
        model='evaluation_PersonaChat:TransformerAgent')
    opt = parser.parse_args(print_args=False)

    if opt['eval_type'] == "hits@1":
        setup_args = setup_args_hits(None, opt["revised"])
        eval_fct = partial(eval_hits, print_parser=setup_args, output_file=opt["model_checkpoint"])
    elif opt['eval_type'] == "f1":
        setup_args = setup_args_f1(None, revised=opt["revised"])
        eval_fct = partial(eval_f1, print_parser=setup_args, output_file=opt["model_checkpoint"], beam=opt["beam"])
    else:
        raise ValueError

    setup_args.set_params(
        model='evaluation_PersonaChat:TransformerAgent')
    opt = setup_args.parse_args(print_args=False)

    eval_fct(opt)