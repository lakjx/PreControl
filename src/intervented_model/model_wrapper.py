import torch
import torch.nn as nn
from transformers import PreTrainedModel
from src.preference_router import PreferenceRouter

class PreferenceModelWrapper(nn.Module):
    def __init__(self, base_model: PreTrainedModel, router: PreferenceRouter, layer_idx: int = -1):
        """
        Wraps a huggingface PreTrainedModel to inject a PreferenceRouter at a specific layer via forward hook.
        
        Args:
            base_model: The LLM to wrap (e.g. Phi3ForCausalLM). Expected to have a `model.layers` attribute.
            router: The PreferenceRouter instance.
            layer_idx: The index of the layer to attach the hook to. (-1 for the last layer).
        """
        super().__init__()
        self.base_model = base_model
        self.router = router
        
        # Freeze the base model to save memory during DPO
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # The router is the only trainable component
        for param in self.router.parameters():
            param.requires_grad = True
            
        # Find the target layer
        if not hasattr(self.base_model, 'model') or not hasattr(self.base_model.model, 'layers'):
            raise ValueError("The provided base_model does not have a `model.layers` attribute. "
                             "This wrapper assumes a standard Llama/Mistral/Phi architecture.")
            
        self.layers = self.base_model.model.layers
        self.layer_idx = layer_idx if layer_idx >= 0 else len(self.layers) + layer_idx
        self.target_layer = self.layers[self.layer_idx]
        
        self.current_alpha = None
        self._hook_handle = self.target_layer.register_forward_hook(self._forward_hook)
        
    def _forward_hook(self, module, inputs, output):
        """
        The hook modifies the hidden states right after the target transformer layer computations.
        """
        if self.current_alpha is None:
            return output
            
        # For standard transformer layers, output is a tuple (hidden_states, optional_caches, ...)
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
            
        # Check devices
        device = hidden_states.device
        dtype = hidden_states.dtype
        alpha_dev = self.current_alpha.to(device)
        
        # Ensure alpha batch size matches hidden_states (important during DPO where batch=batch_size*2)
        batch_size = hidden_states.shape[0]
        if alpha_dev.shape[0] != batch_size and alpha_dev.dim() == 1:
            alpha_dev = alpha_dev.unsqueeze(0).expand(batch_size, -1)
        elif alpha_dev.shape[0] != batch_size:
            # Handle cases where alpha is passed as batched but mismatch
            # E.g. generate() might duplicate batch for beam search
            # We simply assume the first few map or duplicate
            # In a strict implementation, we would require exact matching or handle it upstream.
            if alpha_dev.shape[0] == 1:
                alpha_dev = alpha_dev.expand(batch_size, -1)
            else:
                # Just take the first one and expand as fallback
                alpha_dev = alpha_dev[0:1].expand(batch_size, -1)
            
        # Route logic: h_edited = Router(hidden_states, alpha)
        # Avoid overriding dtype since base model uses bfloat16 and router might be float32/bf16
        edited_hidden_states = self.router(hidden_states.to(self.router.W.dtype), alpha_dev).to(dtype)
        
        if isinstance(output, tuple):
            return (edited_hidden_states,) + output[1:]
        return edited_hidden_states
        
    def set_alpha(self, alpha: torch.Tensor):
        """
        Set the preference vector to condition the generation.
        alpha: tensor of shape [batch_size, num_attributes] or [num_attributes]
        """
        self.current_alpha = alpha
        
    def forward(self, *args, alpha=None, **kwargs):
        """
        Forward pass. If alpha is provided, it replaces the current_alpha.
        """
        # Always set alpha (so if it's None, it correctly disables the router)
        self.set_alpha(alpha)
            
        return self.base_model(*args, **kwargs)
        
    def generate(self, *args, alpha=None, **kwargs):
        """
        Wrapper around base_model.generate().
        """
        if alpha is not None:
            self.set_alpha(alpha)
        return self.base_model.generate(*args, **kwargs)
        
    def remove_hook(self):
        """Clean up the hook if this wrapper is discarded."""
        self._hook_handle.remove()
