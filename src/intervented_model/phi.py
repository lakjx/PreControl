from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from transformers import Phi3Config, Phi3Model, Phi3ForCausalLM
from transformers.utils import logging
from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
import torch.nn.functional as F


logger = logging.get_logger(__name__)

class Intervented_PhiModel(Phi3Model):
    def __init__(self, config: Phi3Config):
        super().__init__(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
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
        patience: Optional[int] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

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

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
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
                    position_embeddings,
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
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

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

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()
    

class Intervented_PhiForCausalLM(Phi3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = Intervented_PhiModel(config)
        self.value_model = None
        self.lr = None
        self.epochs = None
        # intended for filtering out hidden states corresponding to finished sequences,
        #   see `Intervented_PhiModel.forward` above
        self.sequence_unfinished_flag = None
        # Track number of generated tokens per sample
        self.generated_token_counts = None
        self.target_score = None
        self.patience = None
    
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
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,

    ) -> Union[Tuple, CausalLMOutputWithPast]:
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
            target_score=self.target_score,
            patience = self.patience
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # Update token count for each sample in the batch
        if self.generated_token_counts is None:
            # Initialize counter if this is the first forward pass
            self.generated_token_counts = torch.ones(input_ids.shape[0], device=self.device, dtype=torch.long)
        else:
            # Increment counter for sequences that are still being generated
            self.generated_token_counts[self.sequence_unfinished_flag] += 1

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)

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
