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
from torch.cuda.amp import GradScaler
from torch import autocast
from tqdm.contrib.logging import logging_redirect_tqdm

def _average_all(tensor):
	# We copy because modification happens in-place
	averaged = tensor.detach().clone()
	# We use `all_reduce` because it is better supported than `reduce`
	torch.distributed.all_reduce(averaged, torch.distributed.ReduceOp.SUM)
	return averaged / torch.distributed.get_world_size()

def evaluation(_C, eval_data, model, device, is_root=True):
	model.eval()
	loss_list = []

	with logging_redirect_tqdm():
		with torch.no_grad():
			#if is_root:
			pbar = tqdm(eval_data)
			#else:
			#	pbar = eval_data
			
			for batch in pbar:
				
				for n in batch:
					if n in ['gt_x', 'gt_y', 'data_index']: continue
					batch[n] = batch[n].to(device)

				outputs = model(
				    input_ids=batch['encoder_input_ids'], 
				    attention_mask=batch['encoder_mask'], 
				    labels=batch['decoder_input_ids'],
				    task_ids=batch['task_ids'],
					task_type_ids=batch['task_type_ids'],
					prefix_ids=batch['prefix_ids'],
				)
				loss = outputs.loss
				#loss_list.append(_average_all(loss).item())
				loss_list.append(loss)

	final_loss = sum(loss_list) / len(loss_list)

	#if is_root:
	#print("EVAL LOSS %.2f" % final_loss)

	return final_loss


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

	

	#is_root = (not _A.deepspeed) or torch.distributed.get_rank() == 0
	is_root = 0

	
	tokenizer, model = t5_model.get_full_finetune_t5_model(_C)
	

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

	#if _C.enable_nlu:
	#	test_loader = get_single_h5py_nlp_data(_C, _C.test_path, _C.train_path, "test", val_batch_size, tokenizer, #_C.max_length, shuffle=True, distributed=False, is_root=True, is_train=False)

	train_batch_size = _C.batch_size
	train_loader = get_finetune_data(_C, _C.train_path, "train", train_batch_size, tokenizer, _C.max_length, shuffle=True, distributed=False, is_root=True, is_train=True)

	if _C.enable_adam_opt:
		optimizer = utils.build_optimizer(_C, model)
	elif _C.enable_full_finetune:
		optimizer = utils.build_adam_optimizer(_C, model)
	else:
		optimizer = utils.build_t5_optimizer(_C, model)

	if _C.enable_new_task_embeddings and _C.load_from_pretrained:
		dist_model.module.update_task_embedding(_C.prefix_set_number if _C.prefix_set_number > 0 else 1)

	#if torch.cuda.is_available():
	#	model.load_state_dict(torch.load(_A.start_from_checkpoint), strict=False)
	#else:
	#	model.load_state_dict(torch.load(os.path.join(_A.start_from_checkpoint, 'best.pth'), 
	#map_location=torch.device('cpu'))['model'], strict=False)

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

	if _A.train:
		train_iter = iter(train_loader)
		#print("length of train iter:"+str(len(train_iter)))
		if _C.num_training_steps == 0:
			length_train_data = len(train_iter) // (_C.batch_size // torch.distributed.get_world_size())
			_C.num_training_steps = int(length_train_data * _C.max_epoch / _C.gradient_accumulation_steps)
		epoch_num = math.ceil(_C.num_training_steps / _C.checkpoint_every_step)

		os.makedirs(_A.serialization_dir, exist_ok=True)
		_C.dump(os.path.join(_A.serialization_dir, "config.yml"))

		eval_every = _C.checkpoint_every_step * _C.gradient_accumulation_steps
		total_step = 0
		lowest_loss = -1e10
		best_test_performance = 0

		for epoch in range(epoch_num):
			run_step = eval_every if total_step + eval_every < _C.num_training_steps * _C.gradient_accumulation_steps else  _C.num_training_steps * _C.gradient_accumulation_steps - total_step
			model.train()

			print('EPOCH %d / %d' % (epoch + 1, epoch_num))
			pbar = tqdm(total=math.ceil(run_step / _C.gradient_accumulation_steps), file=sys.stdout)

			for step in range(run_step):
				try:
					batch = next(train_iter)
				except:
					train_iter = iter(train_loader)
					batch = next(train_iter)

				for n in batch:
					if n in ['gt_x', 'gt_y', 'data_index']: continue
					batch[n] = batch[n].to(device if not _A.multi_gpu else model.encoder.first_device)

				#optimizer.zero_grad()


				#input_seq = tokenizer.decode(batch['encoder_input_ids'][0]).replace("<pad>", "").replace("</s>", "")
				#output_seq = tokenizer.decode(batch['decoder_input_ids'][0]).replace("<pad>", "").replace("</s>", "")
				#print("input seq:"+str(input_seq))
				#print("output seq:"+str(output_seq))

				total_step += 1
				if not _A.mix_precision:
					outputs = model(
					    input_ids=batch['encoder_input_ids'], 
					    attention_mask=batch['encoder_mask'], 
					    labels=batch['decoder_input_ids'],
					    task_ids=batch['task_ids'],
						task_type_ids=batch['task_type_ids'],
						prefix_ids=batch['prefix_ids'],
					)
					loss = outputs.loss / _C.gradient_accumulation_steps
					loss.backward()
				else:
					with autocast(device_type='cuda', dtype=torch.float16):
						outputs = model(
						    input_ids=batch['encoder_input_ids'], 
						    attention_mask=batch['encoder_mask'], 
						    labels=batch['decoder_input_ids'],
						    task_ids=batch['task_ids'],
							task_type_ids=batch['task_type_ids'],
							prefix_ids=batch['prefix_ids'],
						)
						loss = outputs.loss / _C.gradient_accumulation_steps
					scaler.scale(loss).backward()
					
				if (step + 1) % _C.gradient_accumulation_steps == 0:
					if not _A.mix_precision:
						optimizer.step()
					else:
						scaler.step(optimizer)
						scaler.update()
					optimizer.zero_grad()
					ave_loss = loss.item()

					pbar.set_description("loss %.2f" % (ave_loss * _C.gradient_accumulation_steps))
					pbar.update(1)
					pbar.refresh()
			
			

            #torch save
			print("start saving")
			_score = evaluation(_C, dev_loader, model, device, is_root=is_root)
			print("dev loss:"+str(_score))

			torch.save(model.state_dict(), '/vc_data/users/caxu/KnowDA4R/FLANT5_conv/best-'+str(epoch)+ '-' + str(_score.item()) + '.pth')
			
			print("saving end")
			#break

			pbar.close()