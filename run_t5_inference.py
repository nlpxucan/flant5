import argparse
import torch
import torch.nn as nn
from config import Config
import os, sys, math
from MetaPreTrainingDataset import get_finetune_data, _data_wrapper, read_conll
import t5_model
import numpy as np
import random
import utils
from tqdm import tqdm
import re
from seqeval.metrics import f1_score as seq_f1_score
from sklearn.metrics import f1_score as sen_f1_score
from fast_bleu import SelfBLEU
import json
import edit_distance
import copy
from nltk.translate.bleu_score import sentence_bleu

#fr = open('prompt_test.txt','r')
#lines = fr.readlines()

#prompt = ''
#for line in lines:
#	print(line)
#	prompt += line


def online_inference(_C, model, tokenizer, device, input_string, task_ids=None, task_type_ids=None, prefix_ids=None):
	input_ids = tokenizer(input_string, return_tensors="pt")['input_ids']
	input_ids = input_ids.to(device)

	if task_ids is not None:
		task_ids = torch.tensor([[task_ids]]).long()
		task_ids = task_ids.to(device)
	if task_type_ids is not None:
		task_type_ids = torch.tensor([[task_type_ids]]).long()
		task_type_ids = task_type_ids.to(device)
	if prefix_ids is not None:
		prefix_ids = torch.tensor([[prefix_ids]]).long()
		prefix_ids = prefix_ids.to(device)

	outputs = model.generate(
		input_ids=input_ids, 
		task_ids=task_ids,
		task_type_ids=task_type_ids,
		prefix_ids=prefix_ids,
		max_length=_C.max_length,
		min_length=_C.min_length,
		eos_token_id=tokenizer.eos_token_id,
		num_return_sequences=_C.sample_num, 
		do_sample=True,
		top_p=_C.top_p,
		top_k=0,
		early_stopping=True
	)

	#for i in range(outputs.size(0)):
	output_seq = tokenizer.decode(outputs[0]).replace("<pad>", "").replace("</s>", "")
	return output_seq
	#print("system answer: "+output_seq+'\n')



parser = argparse.ArgumentParser("Train a MT5 for Machine Translation")
parser.add_argument(
    "--config", required=True, help="Path to a config file with all configuration parameters."
)
parser.add_argument(
    "--config-override",
    default=[],
    nargs="*",
    help="A sequence of key-value pairs specifying certain config arguments (with dict-like "
    "nesting) using a dot operator. The actual config will be updated and recorded in "
    "the serialization directory.",
)
parser.add_argument(
    "--serialization-dir",
    default=None,
    help="Path to a (non-existent) directory for serializing checkpoints and tensorboard logs.",
)
parser.add_argument(
    "--start-from-checkpoint",
    default=None,
    help="Path to load checkpoint and continue training [only supported for module_training].",
)
parser.add_argument(
    "--output-path",
    default=None,
    help="Path to save output captions",
)
parser.add_argument(
    "--multi-gpu",
    action='store_true'
)
parser.add_argument(
    "--online-inference",
    action='store_true'
)
parser.add_argument(
    "--mix-precision",
    action='store_true'
)
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
group = parser.add_mutually_exclusive_group()
group.add_argument('--train', action='store_true')
group.add_argument('--validation', action='store_true')
group.add_argument('--test', action='store_true')

if __name__ == "__main__":
	_A = parser.parse_args()
	_C = Config(_A.config, _A.config_override)

	np.random.seed(_C.random_seed)
	random.seed(_C.random_seed)
	torch.manual_seed(_C.random_seed)
	torch.cuda.manual_seed_all(_C.random_seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

	os.environ["TOKENIZERS_PARALLELISM"] = "false"
	os.environ["NCCL_DEBUG"] = "WARN"
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	if _C.load_from_pretrained:
		if _C.enable_pretrain_task_embeddings:
			_C.task_embed_count = len(TASK_NAME_LIST)
	else:
		if _C.enable_new_task_embeddings:
			_C.task_embed_count = _C.prefix_set_number if _C.prefix_set_number > 0 else 1

	if _C.enable_full_finetune:
		tokenizer, model = t5_model.get_full_finetune_t5_model(_C)
	elif _C.enable_full_pretrain:
		tokenizer, model = t5_model.get_full_pretrain_t5_model(_C)
	else:
		tokenizer, model = t5_model.get_t5_model(_C)

	if _A.multi_gpu:
		model.parallelize({
		    0: [0,1, 2],
		    1: [3, 4, 5],
		    2: [6, 7, 8],
		    3: [9, 10, 11],
		    4: [12, 13, 14],
		    5: [15, 16, 17],
		    6: [18, 19, 20],
		    7: [21, 22, 23]
		})

	if _A.mix_precision:
		scaler = GradScaler()

	val_batch_size = _C.val_batch_size
	dev_loader = get_finetune_data(_C, _C.dev_path, "validation", val_batch_size, tokenizer, _C.max_length, shuffle=True, distributed=False, is_root=True, is_train=False)

	if _C.enable_adam_opt:
		optimizer = utils.build_optimizer(_C, model)
	elif _C.enable_full_finetune:
		optimizer = utils.build_adam_optimizer(_C, model)
	else:
		optimizer = utils.build_t5_optimizer(_C, model)

	assert _A.start_from_checkpoint is not None, "start_from_checkpoint must not be None"
	#model.from_pretrained(_A.start_from_checkpoint, local_config=_C)
	if torch.cuda.is_available():
		model.load_state_dict(torch.load(_A.start_from_checkpoint), strict=False)
	else:
		model.load_state_dict(torch.load(os.path.join(_A.start_from_checkpoint, 'best.pth'), map_location=torch.device('cpu'))['model'], strict=False)

	if _C.enable_new_task_embeddings and _C.load_from_pretrained:
		model.update_task_embedding(_C.prefix_set_number if _C.prefix_set_number > 0 else 1)

	total_parameter_count = 0
	trainable_parameter_count = 0
	for p in model.parameters():
		total_parameter_count += p.numel()
		if p.requires_grad:
			trainable_parameter_count += p.numel()
	print('Total Parameter Count %d' % total_parameter_count)
	print('Trainable Parameter Count %d' % trainable_parameter_count)

	print(_C)
	for arg in vars(_A):
		print("{:<20}: {}".format(arg, getattr(_A, arg)))

	model.eval()

	if _A.online_inference:
		
		fr = open('Techspec_Wiki_Bing_Sum_test.jsonl','r')
		fw = open('Domain_knowledge_predicted','w')
		fw_ref = open('Domain_knowledge_references','w')

		lines = fr.readlines()

		

		count = 0

		for line in lines:
			print(count)
			count += 1
			
			line_obj = json.loads(line.strip())

			predicted_result = online_inference(_C, model, tokenizer, device, line_obj['input'], prefix_ids=0)	

			fw.write(predicted_result+'\n')
			fw_ref.write(line_obj['output']+'\n')

		

		fw.close()
		fw_ref.close()



	else:
		eval_iter = iter(eval_data)
		output_lines = []
		with torch.no_grad():
			pbar = tqdm(eval_data)

			for batch in pbar:

				for n in batch:
					if n in ['gt_x', 'gt_y', 'data_index']: continue
					batch[n] = batch[n].to(device)

				batch_size = batch['encoder_input_ids'].size(0)

				outputs = model.generate(
					input_ids=batch['encoder_input_ids'], 
					attention_mask=batch['encoder_mask'], 
					task_ids=batch['task_ids'],
					task_type_ids=batch['task_type_ids'],
					prefix_ids=batch['prefix_ids'],
					max_length=_C.max_length,
					min_length=_C.min_length,
					eos_token_id=tokenizer.eos_token_id,
					num_return_sequences=_C.sample_num, 
					do_sample=False,
                    num_beams=20,
					top_p=_C.top_p,
					top_k=50,
					early_stopping=True
				)

				outputs = outputs.view(batch_size, _C.sample_num, -1)

				for i in range(batch_size):
					for j in range(_C.sample_num):
						output_seq = tokenizer.decode(outputs[i][j]).replace("<pad>", "").replace("</s>", "")
						output_lines.append(output_seq)


