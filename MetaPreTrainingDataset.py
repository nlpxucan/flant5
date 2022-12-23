import random
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import numpy as np
import torch

from tqdm import tqdm
import itertools
import os
import h5py
import math
import copy
from pathlib import Path
import re
import json
import nltk

def read_conll(file_path):
    file_path = Path(file_path)

    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)

    data_list = []
    for doc in raw_docs:
        tokens = []
        tags = []
        for line in doc.split('\n'):
            items = line.split()
            if len(items) == 2:
                token, tag = items
                tokens.append(token)
                tags.append(tag)
        data_list.append((tokens, tags))

    return data_list
    
def process_tensor(tensor_list, last_dim, output_mask=False):
    tensor_len = [d.shape[0] for d in tensor_list]
    tensor_max_lenth = max(tensor_len)
    d_type = tensor_list[0].dtype
    if last_dim > 0:
        tensor_np = np.zeros((len(tensor_list), tensor_max_lenth, last_dim), dtype=d_type)
    else:
        tensor_np = np.zeros((len(tensor_list), tensor_max_lenth), dtype=d_type)
    mask_np = np.zeros((len(tensor_list), tensor_max_lenth), dtype=np.float32)
    for i, (d, l) in enumerate(zip(tensor_list, tensor_len)):
        if l > 0:
            tensor_np[i, :l] = d
            mask_np[i, :l] = 1
    if output_mask:
        return torch.from_numpy(tensor_np), torch.from_numpy(mask_np)
    else:
        return torch.from_numpy(tensor_np)

def _data_wrapper(dataset):
    encoder_input_ids, encoder_mask = process_tensor([d[0] for d in dataset], 0, output_mask=True)
    decoder_input_ids, decoder_mask = process_tensor([d[1] for d in dataset], 0, output_mask=True)
    decoder_input_ids[decoder_mask == 0] = -100
    gt_y, gt_x, data_index = None, None, None
    task_index = torch.tensor([0 for d in dataset]).long()
    task_type_index = torch.tensor([0 for d in dataset]).long()
    prefix_ids = torch.tensor([0 for d in dataset]).long()
    
    if len(dataset[0]) == 7:
    	data_index = [d[5] for d in dataset]
    	prefix_ids = torch.tensor([d[4] for d in dataset]).long()
    	task_index = torch.tensor([d[4] for d in dataset]).long()
    	task_type_index = torch.tensor([d[6] for d in dataset]).long()
    	gt_y = [d[3] for d in dataset]
    	gt_x = [d[2] for d in dataset]   	
    elif len(dataset[0]) == 6:
    	data_index = [d[5] for d in dataset]
    	prefix_ids = torch.tensor([d[4] for d in dataset]).long()
    	gt_y = [d[3] for d in dataset]
    	gt_x = [d[2] for d in dataset]
    elif len(dataset[0]) == 4:
    	task_index = torch.tensor([d[2] for d in dataset]).long()
    	task_type_index = torch.tensor([d[3] for d in dataset]).long()
    elif len(dataset[0]) == 3:
    	prefix_ids = torch.tensor([d[2] for d in dataset]).long()
    	

    return {"encoder_input_ids": encoder_input_ids, "encoder_mask": encoder_mask, "decoder_input_ids": decoder_input_ids, "task_ids": task_index, "task_type_ids": task_type_index, "prefix_ids": prefix_ids, "gt_x": gt_x, "gt_y": gt_y, "data_index": data_index}

class FinetuneDataset(Dataset):

	SKIP_ATTRIBUTES = ['gt_x', 'gt_y']

	def __init__(self, config, data_path, tokenizer, is_training=False, is_root=True):
		self.tokenizer = tokenizer
		self.is_training = is_training
		self.config = config
		self.sep_token_id = 0

		self.data_list = []
		data_instance_list = self.get_data_set(data_path)


		for index, data_instance in enumerate(data_instance_list):					
			self.data_list.append((index, None, data_instance))
		
		

		
		if is_root:
			print("Data Size %d" % len(self.data_list))

	def __len__(self):
		return len(self.data_list)

	def __getitem__(self, idx):		
		data_generator = self.gen_from_tag_sequence

		(index, prompt_id, data_instance) = self.data_list[idx]
		x_ids, y_ids, gt_x, gt_y = data_generator(data_instance)
		input_ids = x_ids
		total_length = len(input_ids)

		input_np = np.array(input_ids).astype(np.int64)
		output_np = np.array(y_ids).astype(np.int64)

		return input_np, output_np, gt_x, gt_y, index

	def gen_from_tag_sequence(self, data_instance, is_train_instance=False):
		input, output = data_instance
		
		
		input_x = "%s" % input
		input_y = "%s" % output
		

		gt_y = input_y
		gt_x = input_x
		y_ids = self.tokenizer(input_y, return_tensors="np")['input_ids'][0, :self.config.max_length].tolist()
		x_ids = self.tokenizer(input_x, return_tensors="np")['input_ids'][0, :self.config.max_length].tolist()


		return x_ids, y_ids, gt_x, gt_y


	def gen_from_nlu(self, data_instance, mask_token=False, add_seperator=False):
		raise NotImplementedError

	def get_data_set(self, path, filtering=False):
		data_list = []
		with open(path) as out:
			for l in out:
				items = json.loads(l)
				data_list.append((items['input'], items['output']))
		return data_list

	def is_identical(self, instance_a, instance_b):
		raise NotImplementedError



def get_finetune_data(config, path, split, batch_size, tokenizer, max_length, shuffle=False, distributed=False, is_root=True, is_train=True):

		
	combined_dataset = FinetuneDataset(config, path, tokenizer, is_training=is_train, is_root=is_root)
		
	if is_root:
		print("%s Data Size %d" % (split, len(combined_dataset)))

	if distributed:
		dist_sampler = torch.utils.data.distributed.DistributedSampler(combined_dataset, shuffle=shuffle)
		dist_loader = DataLoader(combined_dataset, pin_memory=True, batch_size=batch_size, num_workers=8, collate_fn=_data_wrapper, sampler=dist_sampler)
		return dist_loader
	else:
		data_loader = DataLoader(combined_dataset, pin_memory=True, batch_size=batch_size, num_workers=8, collate_fn=_data_wrapper, shuffle=shuffle)
		return data_loader