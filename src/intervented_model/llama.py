import torch
from transformers import LlamaModel, LlamaConfig, LlamaForCausalLM
import torch.nn as nn
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from typing import List, Optional, Tuple, Union
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from torch.nn import CrossEntropyLoss
import torch.utils.checkpoint
from transformers.cache_utils import Cache, DynamicCache, StaticCache
import torch.nn.functional as F

logger = logging.get_logger(__name__)


class Intervented_LlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            value_model: Optional[nn.Module] = None,
            lr: Optional[float] = None,
            epochs: Optional[int] = None,
            target_score: Optional[torch.Tensor]= None,
            sequence_unfinished_flag: Optional[torch.Tensor] = None,
            original_hidden_states: Optional[List[torch.Tensor]] = None,
            patience: Optional[int] = None
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        #### use the value model to optimize the hidden states

        if value_model is not None:
            value_model.eval()
            hidden_states_opt = hidden_states.detach().clone()  # also need to set the hidden states to be requires_grad=True
            
            # Ensure hidden_states_opt is on the right device
            device = hidden_states.device
            hidden_states_opt = hidden_states_opt.to(device)
            hidden_states_opt.requires_grad = True

            # Initialize optimizer
            optimizer = torch.optim.AdamW([hidden_states_opt], lr=lr, weight_decay=0.01)
            

            best_loss = None
            best_hidden_states = None
            no_improve = 0
            with torch.enable_grad():
                for iteration in range(epochs):  # Run for a number of iterations
                    if no_improve >= patience:
                        break
                    optimizer.zero_grad()  # Clear previous gradients

                    # Reshape hidden states for value model if needed
                    # Make sure both hidden_states_opt and value_model are on the same device
                    preds = value_model(hidden_states_opt[:, -1, :])  # only compute on the last token; shape (batch_size, 5)
                    
                    # filter the preds by keeping only the sequences that are unfinished
                    if sequence_unfinished_flag is not None:
                        preds = preds[sequence_unfinished_flag]
                    
            
                    # loss = F.mse_loss(preds, target_score, reduction='none')
                    loss = F.smooth_l1_loss(preds, target_score, reduction='none', beta=0.5)

                    target_mask = (target_score != 5)
                    loss = loss * target_mask

                    # Weighted per_sample loss for regularization
                    per_sample_loss = loss.sum(dim=1)
                    sample_weights = 1.0 / (per_sample_loss.detach() + 1e-6)
                    sample_weights = sample_weights / sample_weights.sum()
                    loss = (sample_weights * per_sample_loss).sum()

                    loss.backward()  # Compute gradients with respect to x
                    
                    torch.nn.utils.clip_grad_norm_(parameters=[hidden_states_opt], max_norm=1.0)
                    optimizer.step()  # Update x

                    # Check if the loss has improved
                    if best_loss is None or (best_loss - loss.item()) > 1e-4:
                        best_loss = loss.detach().clone()
                        best_hidden_states = hidden_states_opt.detach().clone()
                        no_improve = 0
                    else:
                        no_improve += 1

            hidden_states = best_hidden_states.detach().clone().to(device)  # Update the hidden states with the optimized x

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Intervented_LlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = Intervented_LlamaModel(config)
        self.value_model = None
        self.lr = None
        self.epochs = None
        # intended for filtering out hidden states corresponding to finished sequences,
        #   see `Intervented_LlamaModel.forward` above
        self.sequence_unfinished_flag = None
        self.original_hidden_states: Optional[List[torch.Tensor]] = None
        # Track number of generated tokens per sample
        self.generated_token_counts = None
        self.target_score = None
        self.patience = None

    def set_value_model(self, value_model):
        self.value_model = value_model

    def set_lr_and_epochs(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs
    
    def set_patience(self, patience):
        self.patience = patience

    def get_generated_token_counts(self):
        """
        Returns a tensor of shape (batch_size,) containing the number of tokens 
        generated so far for each sample in the batch.
        
        Returns:
            torch.Tensor: Tensor of shape (batch_size,) with token counts
                          or None if generation hasn't started
        """
        return self.generated_token_counts

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            num_logits_to_keep: int = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            value_model=self.value_model,
            epochs=self.epochs,
            lr=self.lr,
            sequence_unfinished_flag=self.sequence_unfinished_flag,
            original_hidden_states=self.original_hidden_states,
            target_score=self.target_score,
            patience = self.patience
        )


        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            from transformers.utils import is_torchdynamo_compiling
            if labels is None and not is_torchdynamo_compiling():
                logger.warning_once(
                    "Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)"
                )
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            # TODO: remove the float() operation in v4.46
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :]).float()

        # check if the last token generated is the eos token
        #   TODO: `model.generate()` would compute this operation again so it's not efficient
        next_tokens = torch.argmax(logits.clone()[:, -1, :].float(), dim=-1)
        self.sequence_unfinished_flag &= (next_tokens != self.config.eos_token_id)
        
        # Update token count for each sample in the batch
        if self.generated_token_counts is None:
            # Initialize counter if this is the first forward pass
            self.generated_token_counts = torch.ones(input_ids.shape[0], device=self.device, dtype=torch.long)
        else:
            # Increment counter for sequences that are still being generated
            self.generated_token_counts[self.sequence_unfinished_flag] += 1

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
