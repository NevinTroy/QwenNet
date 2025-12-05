import torch


def load_model(model, model_path):
    """
    Load model from saved weights.
    Handles both UER models and Qwen models (wrapped in QwenWrapper).
    """
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Check if this is a Qwen model (wrapped)
    is_qwen = hasattr(model, 'is_qwen') and model.is_qwen
    
    if is_qwen:
        # For Qwen models, we need to load into the underlying qwen_model
        # Handle both wrapped and unwrapped state dictss
        if hasattr(model, "qwen_model"):
            # Try loading directly into qwen_model
            try:
                model.qwen_model.load_state_dict(state_dict, strict=False)
            except Exception:
                # If that fails, try loading the full wrapper state dict
                if hasattr(model, "module"):
                    model.module.load_state_dict(state_dict, strict=False)
                else:
                    model.load_state_dict(state_dict, strict=False)
        else:
            # Fallback to standard loading
            if hasattr(model, "module"):
                model.module.load_state_dict(state_dict, strict=False)
            else:
                model.load_state_dict(state_dict, strict=False)
    else:
        # Standard UER model loading
        if hasattr(model, "module"):
            model.module.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)
    
    return model
