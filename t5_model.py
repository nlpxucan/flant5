from transformers.models.t5.modeling_t5 import *
from transformers.generation_utils import *
from transformers import T5TokenizerFast
from transformers import RobertaTokenizerFast, RobertaForTokenClassification
from transformers import BertTokenizerFast, BertForTokenClassification, BertForSequenceClassification, BertForNextSentencePrediction
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification

def copyParams(module_src, module_dest):
    params_src = module_src.named_parameters()
    params_dest = module_dest.named_parameters()

    dict_dest = dict(params_dest)

    for name, param in params_src:
        if name in dict_dest:
            dict_dest[name].data.copy_(param.data)

class T5LayerFF(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.feed_forward_proj == "relu":
            self.DenseReluDense = T5DenseReluDense(config)
        elif config.feed_forward_proj == "gated-gelu":
            self.DenseReluDense = T5DenseGatedGeluDense(config)
        else:
            raise ValueError(
                f"{self.config.feed_forward_proj} is not supported. Choose between `relu` and `gated-gelu`"
            )

        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states

class T5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.config = config
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))

        self.layer.append(T5LayerFF(config))

        self.enable_seperated_cross_attention = False
        self.cross_attention_type = 0

    def seperate_cross_attention(self):
        self.enable_seperated_cross_attention = True
        cross_attn = self.layer[1]
        dual_cross_attn = nn.ModuleList()
        dual_cross_attn.append(cross_attn)
        another_cross_attn = T5LayerCrossAttention(self.config)
        copyParams(cross_attn, another_cross_attn)
        dual_cross_attn.append(another_cross_attn)
        self.layer[1] = dual_cross_attn

    def set_cross_attention(self, _type):
        self.cross_attention_type = _type

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):

        if past_key_value is not None:
            assert self.is_decoder, "Only decoder can use `past_key_values`"
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            if self.enable_seperated_cross_attention:
                cross_attention_module_list = self.layer[1]
                cross_attention_module = cross_attention_module_list[self.cross_attention_type]
            else:
                cross_attention_module = self.layer[1]

            cross_attention_outputs = cross_attention_module(
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs

class T5Stack(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None, prefix_length=0, prefix_set_number=0, task_embed_count=0, task_type_vector_count_per_layer=0, enable_layer_wise_prefix=True, factorization_dim=-1, load_from_pretrained=False, delta_prompt_ids=None, delta_prompt_z_dim=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        self.enable_layer_wise_prefix = enable_layer_wise_prefix
        self.enable_prefix_factorization = factorization_dim > 0 and (not self.is_decoder)
        self.use_task_prefix_embeds = task_embed_count > 0
        self.use_task_type_embed = task_type_vector_count_per_layer > 0
        self.factorization_dim = factorization_dim
        self.enable_delta_prompt_for_embeddings = delta_prompt_ids is not None
        self.delta_prompt_z_dim = delta_prompt_z_dim
        if self.enable_delta_prompt_for_embeddings:
            self.delta_prompt_length = delta_prompt_ids.size(0)

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.use_input_prefix_embeds = prefix_length > 0 and prefix_set_number > 0

        self.prefix_embedding = None
        if not self.is_decoder:
            if self.use_input_prefix_embeds:
                prefix_dim = self.factorization_dim if self.enable_prefix_factorization else config.d_model
                if self.enable_layer_wise_prefix:
                    self.prefix_embedding = nn.Embedding(prefix_set_number, prefix_length * prefix_dim * config.num_layers)
                else:
                    self.prefix_embedding = nn.Embedding(prefix_set_number, prefix_length * prefix_dim)

                if self.enable_prefix_factorization:
                    self.factorization_layer = nn.Linear(self.factorization_dim, config.d_model)
            
            if self.enable_delta_prompt_for_embeddings:    
                self.fixed_prompt_embeddings = torch.nn.parameter.Parameter(self.embed_tokens(delta_prompt_ids))
                self.random_linear_layer = nn.Linear(self.delta_prompt_z_dim, self.delta_prompt_length * config.d_model)
                self.delta_prompt_z = nn.Embedding(1, self.delta_prompt_z_dim)
                
            if self.use_task_prefix_embeds:
                self.task_embedding = nn.Embedding(task_embed_count, config.d_model)

            if self.use_task_type_embed:
                self.task_type_embedding = nn.Embedding(2, task_type_vector_count_per_layer * config.d_model * config.num_layers)

        self.prefix_length = prefix_length
        self.prefix_set_number = prefix_set_number
        self.d_task_embedding = config.d_model
        self.task_type_vector_count_per_layer = task_type_vector_count_per_layer

        self.init_weights()
        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def seperate_cross_attention(self):
        for t5_block in self.block:
            t5_block.seperate_cross_attention()

    def set_cross_attention(self, _type):
        for t5_block in self.block:
            t5_block.set_cross_attention(_type)

    def update_prefix_embedding(self, prefix_set_number):
        pre_trained_embedding = self.prefix_embedding.weight.detach().data[0]
        prefix_dim = self.factorization_dim if self.enable_prefix_factorization else self.config.d_model
        self.prefix_embedding = nn.Embedding(prefix_set_number, self.prefix_length * prefix_dim * self.config.num_layers)
        for i in range(prefix_set_number):
            self.prefix_embedding.weight.data[i].copy_(pre_trained_embedding)

    def update_task_embedding(self, new_task_num):
        device = next(self.parameters()).device
        self.use_task_prefix_embeds = new_task_num > 0
        if new_task_num > 0:
            self.task_embedding = nn.Embedding(new_task_num, self.d_task_embedding)
            self.task_embedding = self.task_embedding.to(device)
        else:
            self.task_embedding = None

    def parallelize(self, device_map=None):
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.block), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.block))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # Load onto devices
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)

        # Set embed_tokens to first layer
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        if self.prefix_embedding is not None:
            self.prefix_embedding = self.prefix_embedding.to(self.first_device)
        if not self.is_decoder:
            if self.use_task_prefix_embeds:
                self.task_embedding = self.task_embedding.to(self.first_device)
            if self.use_task_type_embed:
                self.task_type_embedding = self.task_type_embedding.to(self.first_device)
            if self.enable_prefix_factorization:
                self.factorization_layer = self.factorization_layer.to(self.first_device)

        # Set final layer norm to last device
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        if self.prefix_embedding is not None:
            self.prefix_embedding = self.prefix_embedding.to("cpu")
        if not self.is_decoder:
            if self.use_task_type_embed:
                self.task_type_embedding = self.task_type_embedding.to("cpu")
            if self.enable_prefix_factorization:
                self.factorization_layer = self.factorization_layer.to("cpu")
            if self.enable_prefix_factorization:
                self.factorization_layer = self.factorization_layer.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_ids=None,
        task_type_ids=None,
        prefix_ids=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f":obj:`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        if self.enable_delta_prompt_for_embeddings:
            if not self.is_decoder:
                z_index = torch.zeros((attention_mask.size(0),), dtype=torch.long).to(inputs_embeds.device)
                z_variable = self.delta_prompt_z(z_index)
                delta_prompt = self.random_linear_layer(z_variable).view(attention_mask.size(0), self.delta_prompt_length, -1) 
                inputs_embeds = torch.cat([delta_prompt + self.fixed_prompt_embeddings, inputs_embeds], dim=1)
                seq_length = seq_length + self.delta_prompt_length

                task_prefix_mask = torch.ones((attention_mask.size(0), self.delta_prompt_length), dtype=attention_mask.dtype).to(inputs_embeds.device)
                attention_mask = torch.cat([task_prefix_mask, attention_mask], axis=1)
            else:
                task_prefix_mask = torch.ones((attention_mask.size(0), self.delta_prompt_length), dtype=encoder_attention_mask.dtype).to(encoder_attention_mask.device)
                encoder_attention_mask = torch.cat([task_prefix_mask, encoder_attention_mask], axis=1)

        if self.use_task_prefix_embeds:
            if not self.is_decoder:
                assert task_ids is not None
                task_prefix_embs = self.task_embedding(task_ids).view(task_ids.size(0), 1, -1)
                inputs_embeds = torch.cat([task_prefix_embs, inputs_embeds], dim=1)
                seq_length = seq_length + 1
                
                task_prefix_mask = torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype).to(inputs_embeds.device)
                attention_mask = torch.cat([task_prefix_mask, attention_mask], axis=1)
            else:
                task_prefix_mask = torch.ones((attention_mask.size(0), 1), dtype=encoder_attention_mask.dtype).to(encoder_attention_mask.device)
                encoder_attention_mask = torch.cat([task_prefix_mask, encoder_attention_mask], axis=1)


        if self.use_task_type_embed:
            if not self.is_decoder:
                assert task_type_ids is not None
                task_type_embs = self.task_type_embedding(task_type_ids).view(inputs_embeds.size(0), self.config.num_layers, self.task_type_vector_count_per_layer, -1)
                inputs_embeds = torch.cat([task_type_embs[:, 0], inputs_embeds], dim=1)

                task_type_mask = torch.ones((attention_mask.size(0), self.task_type_vector_count_per_layer), dtype=attention_mask.dtype).to(inputs_embeds.device)
                attention_mask = torch.cat([task_type_mask, attention_mask], axis=1)
                seq_length = seq_length + self.task_type_vector_count_per_layer
            else:
                task_type_mask = torch.ones((attention_mask.size(0), self.task_type_vector_count_per_layer), dtype=encoder_attention_mask.dtype).to(encoder_attention_mask.device)
                encoder_attention_mask = torch.cat([task_type_mask, encoder_attention_mask], axis=1)

        input_shape = (batch_size, seq_length)

        if self.use_input_prefix_embeds: 
            if not self.is_decoder:
                if prefix_ids is None:
                    prefix_ids = torch.zeros((inputs_embeds.size(0),)).long().to(inputs_embeds.device)
                
                if self.enable_layer_wise_prefix:
                    prefix_embs = self.prefix_embedding(prefix_ids).view(inputs_embeds.size(0), self.config.num_layers, self.prefix_length, -1)
                    if self.enable_prefix_factorization:
                        first_prefix_embs = self.factorization_layer(prefix_embs[:, 0])
                    else:
                        first_prefix_embs = prefix_embs[:, 0]
                    inputs_embeds = torch.cat([first_prefix_embs, inputs_embeds], dim=1)
                else:
                    prefix_embs = self.prefix_embedding(prefix_ids).view(inputs_embeds.size(0), self.prefix_length, -1)
                    if self.enable_prefix_factorization:
                        first_prefix_embs = self.factorization_layer(prefix_embs)
                    else:
                        first_prefix_embs = prefix_embs
                    inputs_embeds = torch.cat([first_prefix_embs, inputs_embeds], dim=1)

                prefix_mask = torch.ones((attention_mask.size(0), self.prefix_length), dtype=attention_mask.dtype).to(inputs_embeds.device)
                attention_mask = torch.cat([prefix_mask, attention_mask], axis=1)
                input_shape = (batch_size, seq_length + self.prefix_length)
            else:
                prefix_mask = torch.ones((attention_mask.size(0), self.prefix_length), dtype=encoder_attention_mask.dtype).to(encoder_attention_mask.device)
                encoder_attention_mask = torch.cat([prefix_mask, encoder_attention_mask], axis=1)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]

            if i > 0 and (not self.is_decoder):
                enable_layer_wise_prefix = self.use_input_prefix_embeds and self.enable_layer_wise_prefix
                if enable_layer_wise_prefix and (not self.use_task_type_embed):
                    prefix, real_hidden_states = torch.split(hidden_states, [self.prefix_length, hidden_states.size(1) - self.prefix_length], dim=1)
                    layer_wise_prefix_embs = prefix_embs[:, i]
                    if self.enable_prefix_factorization:
                        layer_wise_prefix_embs = self.factorization_layer(layer_wise_prefix_embs)
                    hidden_states = torch.cat([layer_wise_prefix_embs.to(real_hidden_states.device), real_hidden_states], dim=1)
                elif enable_layer_wise_prefix and self.use_task_type_embed:
                    prefix, task_type_prefix, real_hidden_states = torch.split(hidden_states, [self.prefix_length, self.task_type_vector_count_per_layer, hidden_states.size(1) - self.prefix_length - self.task_type_vector_count_per_layer], dim=1)
                    layer_wise_prefix_embs = prefix_embs[:, i]
                    layer_wise_task_type_embs = task_type_embs[:, i]
                    if self.enable_prefix_factorization:
                        layer_wise_prefix_embs = self.factorization_layer(layer_wise_prefix_embs)
                    hidden_states = torch.cat([layer_wise_prefix_embs.to(real_hidden_states.device), layer_wise_task_type_embs.to(real_hidden_states.device), real_hidden_states], dim=1)
                elif (not enable_layer_wise_prefix) and self.use_task_type_embed:
                    task_type_prefix, real_hidden_states = torch.split(hidden_states, [self.task_type_vector_count_per_layer, hidden_states.size(1) - self.task_type_vector_count_per_layer], dim=1)
                    layer_wise_task_type_embs = task_type_embs[:, i]
                    hidden_states = torch.cat([layer_wise_task_type_embs.to(real_hidden_states.device), real_hidden_states], dim=1)

            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )

class T5ForPT(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config, **model_args):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.local_config = model_args['local_config']

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, 
            self.shared, 
            prefix_length=self.local_config.prefix_length, 
            prefix_set_number=self.local_config.prefix_set_number, 
            task_embed_count=self.local_config.task_embed_count, 
            task_type_vector_count_per_layer=self.local_config.task_type_vector_count_per_layer, 
            enable_layer_wise_prefix=self.local_config.enable_layer_wise_prefix, 
            factorization_dim=self.local_config.factorization_dim, 
            load_from_pretrained=self.local_config.load_from_pretrained,
            delta_prompt_ids=model_args['delta_prompt_ids'] if 'delta_prompt_ids' in model_args else None, 
            delta_prompt_z_dim=self.local_config.delta_prompt_z_dim
        )

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, 
            self.shared, 
            prefix_length=self.local_config.prefix_length, 
            prefix_set_number=self.local_config.prefix_set_number, 
            task_embed_count=self.local_config.task_embed_count, 
            task_type_vector_count_per_layer=self.local_config.task_type_vector_count_per_layer, 
            enable_layer_wise_prefix=self.local_config.enable_layer_wise_prefix, 
            factorization_dim=self.local_config.factorization_dim, 
            load_from_pretrained=self.local_config.load_from_pretrained,
            delta_prompt_ids=model_args['delta_prompt_ids'] if 'delta_prompt_ids' in model_args else None, 
            delta_prompt_z_dim=self.local_config.delta_prompt_z_dim
        )

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def update_prefix_embedding(self, prefix_set_number):
        self.encoder.update_prefix_embedding(prefix_set_number)

    def update_task_embedding(self, new_task_num):
        self.encoder.update_task_embedding(new_task_num)

    def seperate_cross_attention(self):
        self.decoder.seperate_cross_attention()

    def set_cross_attention(self, _type):
        self.decoder.set_cross_attention(_type)

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True


    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()


    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids=None,
        task_ids=None,
        task_type_ids=None,
        prefix_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``

        Returns:

        Examples::

            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')

            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits

            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                task_ids=task_ids,
                task_type_ids=task_type_ids,
                prefix_ids=prefix_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        
        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100, reduction=('none' if self.local_config.enable_flip_filtering else 'mean'))
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            if self.local_config.enable_flip_filtering:
                B, N = labels.size()
                loss = loss.view(B, N)
                mask = (labels > -100).float()
                loss = torch.sum(loss * mask, dim=1) / torch.sum(mask, dim=1)
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return_dict = {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

        for additional_parameter in ['task_ids', 'task_type_ids', 'prefix_ids']:
            if additional_parameter in kwargs:
                return_dict[additional_parameter] = kwargs[additional_parameter]

        return return_dict

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs: ModelOutput = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        for additional_parameter in ['task_ids', 'task_type_ids', 'prefix_ids']:
            if additional_parameter in model_kwargs and model_kwargs[additional_parameter] is not None:
                model_kwargs[additional_parameter] = model_kwargs[additional_parameter].index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            assert encoder_outputs is not None
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
            )
            model_kwargs["encoder_outputs"] = encoder_outputs
        return input_ids, model_kwargs

def get_t5_model(config):
    tokenizer = T5TokenizerFast.from_pretrained(config.tokenizer_type)

    if len(config.delta_prompt_text) > 0:
        delta_prompt_ids = tokenizer(config.delta_prompt_text, return_tensors="pt").input_ids[0, :-1]
    else:
        delta_prompt_ids = None

    model = T5ForPT.from_pretrained(config.lm_type, local_config=config, delta_prompt_ids=delta_prompt_ids)

    for p in model.parameters():
        p.requires_grad = False

    if config.prefix_length > 0 and config.prefix_set_number > 0:
        for p in model.encoder.prefix_embedding.parameters():
            p.requires_grad = True
        if config.factorization_dim > 0:
            for p in model.encoder.factorization_layer.parameters():
                p.requires_grad = True

    if delta_prompt_ids is not None:
        for p in model.encoder.delta_prompt_z.parameters():
            p.requires_grad = True

    if config.task_embed_count > 0:
        for p in model.encoder.task_embedding.parameters():
            p.requires_grad = True

    return tokenizer, model

def get_full_finetune_t5_model(config):
    tokenizer = T5TokenizerFast.from_pretrained(config.tokenizer_type)
    model = T5ForPT.from_pretrained(config.lm_type, local_config=config, dropout_rate=config.t5_dropout)

    if config.enable_small_tune:
        for p in model.parameters():
            p.requires_grad = False

        if config.task_embed_count > 0:
            for p in model.encoder.task_embedding.parameters():
                p.requires_grad = True

        if config.task_type_vector_count_per_layer > 0:
            for p in model.encoder.task_type_embedding.parameters():
                p.requires_grad = True

        if config.prefix_length > 0 and config.prefix_set_number > 0:
            for p in model.encoder.prefix_embedding.parameters():
                p.requires_grad = True

    return tokenizer, model

def get_bert_model(config, output_size):
    tokenizer = BertTokenizerFast.from_pretrained(config.lm_type)
    model = BertForTokenClassification.from_pretrained(config.lm_type, num_labels=output_size, attention_probs_dropout_prob=0.3, hidden_dropout_prob=0.3)

    return tokenizer, model

def get_bert_pair_model(config, output_size):
    tokenizer = BertTokenizerFast.from_pretrained(config.lm_type)
    model = BertForNextSentencePrediction.from_pretrained(config.lm_type, num_labels=output_size, attention_probs_dropout_prob=0.3, hidden_dropout_prob=0.3)
    return tokenizer, model

def get_bert_sen_classification_model(config, output_size):
    tokenizer = BertTokenizerFast.from_pretrained(config.lm_type)
    model = BertForSequenceClassification.from_pretrained(config.lm_type, num_labels=output_size)

    return tokenizer, model

def get_debertav2_sen_classification_model(config, output_size):
    tokenizer = DebertaV2Tokenizer.from_pretrained(config.lm_type)
    model = DebertaV2ForSequenceClassification.from_pretrained(config.lm_type, num_labels=output_size)

    return tokenizer, model

