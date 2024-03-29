from typing import Any, List
from yacs.config import CfgNode as CN

class Config(object):

	def __init__(self, config_yaml: str, config_override: List[Any] = []):
		self._C = CN()
		self._C.random_seed = 0
		self._C.train_path = ""
		self._C.dev_path = ""
		self._C.test_path = ""
		self._C.lm_type = ""
		self._C.tokenizer_type = ""
		self._C.label_path = ""
		self._C.nlu_model_path = ""
		self._C.batch_size = 10
		self._C.val_batch_size = -1
		self._C.beam_size = 5
		self._C.sample_num = 5
		self._C.eval_data_replication = 1
		self._C.min_length = 5
		self._C.learning_rate = 1e-3
		self._C.adam_epsilon = 1e-8
		self._C.weight_decay = 1e-6
		self._C.num_training_steps = 0
		self._C.gradient_accumulation_steps = 1
		self._C.warmup_ratio = 0.0
		self._C.warmup_step = 0
		self._C.oversample = 1
		self._C.max_epoch = 10
		self._C.checkpoint_every_step = 500
		self._C.max_grad_norm = 1.0
		self._C.filter_by_min_length = -1
		self._C.prefix_length = 0
		self._C.prefix_set_number = 0
		self._C.enable_layer_wise_prefix = True
		self._C.load_from_pretrained = False
		self._C.max_length = 128
		self._C.enable_full_finetune = False
		self._C.enable_filtering_error = False
		self._C.enable_consistency_filtering = False
		self._C.enable_full_pretrain = False
		self._C.enable_adam_opt = False
		self._C.enable_sentence_classification = False
		self._C.enable_pair_sentence_classification = False
		self._C.select_model_by_ppl = False
		self._C.score_top_ratio = -1.0
		self._C.enable_eval_oversample = False
		self._C.only_save_trainable_parameters = False
		self._C.enable_nlg = False
		self._C.enable_nlu = False
		self._C.nlu_beam_size = 1
		self._C.nlg_with_annotation = False
		self._C.top_p = 0.9
		self._C.enable_meta_training = False
		self._C.use_t5_span = False
		self._C.factorization_dim = -1
		self._C.gen_instance_count = 500
		self._C.in_context_instance_count = 0
		self._C.meta_task_max_sample = 300000
		self._C.meta_task_max_value_length = 256
		self._C.enable_label_constrained_decode = False
		self._C.nlu_only = False
		self._C.nlg_only = False
		self._C.tri_path = ""
		self._C.enable_tri_training = False
		self._C.enable_prompt_ensemble = False
		self._C.task_embed_count = 0
		self._C.enable_new_task_embeddings = False
		self._C.enable_pretrain_task_embeddings = False
		self._C.task_type_vector_count_per_layer = 0
		self._C.shuffle_example = False
		self._C.output_as_json = False
		self._C.enable_flip_filtering = False
		self._C.mask_warmup_epochs = -1
		self._C.delta_prompt_text = ""
		self._C.delta_prompt_z_dim = 100
		self._C.node_count = 0
		self._C.eval_by_loss = False
		self._C.save_model_each_epoch = True
		self._C.running_task = ""
		self._C.full_ranking_top_k = 256
		self._C.enable_small_tune = False
		self._C.only_load_module = True
		self._C.t5_dropout = 0.1
		self._C.key_replacement_path = ""
		self._C.tri_training_additional_model_output_path = []
		self._C.pre_training_modes = []
		self._C.training_da_mode = []
		self._C.eval_da_mode = []
		self._C.lm_gen_train_path_list = []

		# Override parameter values from YAML file first, then from override list.
		self._C.merge_from_file(config_yaml)
		self._C.merge_from_list(config_override)

		# Make an instantiated object of this class immutable.
		self._C.freeze()

	def dump(self, file_path: str):
		self._C.dump(stream=open(file_path, "w"))

	def __getattr__(self, attr: str):
		if attr == "__setstate__":
			raise AttributeError(attr)
		return self._C.__getattr__(attr)

	def __str__(self):
		return _config_str(self)

	def __repr__(self):
		return self._C.__repr__()


def _config_str(config: Config) -> str:
    r"""
    Collect a subset of config in sensible order (not alphabetical) according to phase. Used by
    :func:`Config.__str__()`.

    Parameters
    ----------
    config: Config
        A :class:`Config` object which is to be printed.
    """
    _C = config

    __C: CN = CN({"RANDOM_SEED": _C.random_seed})
    common_string: str = str(__C) + "\n"

    return common_string