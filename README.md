# flant5

<strong>Env Installation</strong>

conda create --name flant5 python==3.8
conda init
source ~/.bashrc
conda activate flant5


pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers==4.18.0
pip install deepspeed==0.6.3
pip install h5py==3.6.0
pip install scikit-learn==1.0.2
pip install yacs
pip install datasets
pip install nltk
pip install seqeval
pip install fast_bleu
pip install edit_distance


<strong>Fully Finetuning</strong>

conda init
source ~/.bashrc
conda activate flant5


python run_t5.py --config configuration/rte/nlg_meta.yml --serialization-dir nlg_da --config-override num_training_steps 100000 running_task rte_document_generation enable_small_tune False tokenizer_type google/flan-t5-xxl lm_type google/flan-t5-xxl batch_size 1 in_context_instance_count 0 checkpoint_every_step 1000 max_length 2300 enable_new_task_embeddings False enable_pretrain_task_embeddings False task_embed_count 0 val_batch_size 4 random_seed 0 train_path dataset/rte/conv_train.jsonl dev_path  dataset/rte/conv_dev.jsonl gradient_accumulation_steps 1 --train --multi-gpu

Prefix Tuning 

conda init
source ~/.bashrc
conda activate flant5



python run_t5.py --config configuration/nlg_meta.yml --serialization-dir nlg_da --config-override num_training_steps 100000 enable_small_tune True tokenizer_type google/flan-t5-xxl lm_type google/flan-t5-xxl batch_size 1  checkpoint_every_step 1000 max_length 2300 enable_new_task_embeddings False enable_pretrain_task_embeddings False task_embed_count 0 prefix_length 3 prefix_set_number 1 enable_layer_wise_prefix True  val_batch_size 4 random_seed 0 train_path dataset/conv_train.jsonl dev_path  dataset/conv_dev.jsonl gradient_accumulation_steps 1 --train --multi-gpu



<strong>Prompt Tuning</strong>

conda init
source ~/.bashrc
conda activate flant5

python run_t5.py --config configuration/nlg_meta.yml --serialization-dir nlg_da --config-override num_training_steps 100000 enable_small_tune True tokenizer_type google/flan-t5-xxl lm_type google/flan-t5-xxl batch_size 1  checkpoint_every_step 1000 max_length 2300 enable_new_task_embeddings False enable_pretrain_task_embeddings False task_embed_count 0 prefix_length 3 prefix_set_number 1   val_batch_size 4 random_seed 0 train_path dataset/conv_train.jsonl dev_path  dataset/conv_dev.jsonl gradient_accumulation_steps 1 --train --multi-gpu
