import torch

class ResidualSteerer:
    """Attaches additive unit refusal vectors to Cohere2ForCausalLM decoder layers."""
    def __init__(self, model, layer_indices: list[int], direction: torch.Tensor, scale: float):
        self.model = model
        self.layer_indices = layer_indices
        self.direction = direction
        self.scale = scale
        self.handles = []

    def _make_hook(self):
        def hook_fn(module, inputs, output):
            if isinstance(output, tuple):
                h = output[0]
                vec = self.direction.to(device=h.device, dtype=h.dtype)
                h_steered = h + (self.scale * vec)
                return (h_steered,) + output[1:]
            else:
                vec = self.direction.to(device=output.device, dtype=output.dtype)
                return output + (self.scale * vec)
        return hook_fn

    def __enter__(self):
        layers = self.model.model.layers
        for idx in self.layer_indices:
            self.handles.append(layers[idx].register_forward_hook(self._make_hook()))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self.handles:
            h.remove()
        self.handles.clear()
