from uer.embeddings import *
from uer.encoders import *
from uer.decoders import *
from uer.targets import *
from uer.models.model import Model
import torch
import torch.nn as nn


class QwenWrapper(nn.Module):
    """
    Wrapper class to make Qwen models compatible with UER's interface.
    """
    def __init__(self, qwen_model, args):
        super(QwenWrapper, self).__init__()
        self.qwen_model = qwen_model
        self.args = args
        self.is_qwen = True
        
    def forward(self, src, tgt, seg, tgt_in=None, tgt_seg=None):
        """
        Convert UER's forward signature to Qwen's expected format.
        
        Args:
            src: [batch_size x seq_length] - input token ids
            tgt: [batch_size x seq_length] - target token ids (for language modeling)
                  In UER, tgt[i] is the next token after src[i]
            seg: [batch_size x seq_length] - segment/attention mask
        Returns:
            loss_info: (loss, correct, denominator) for LM
        """
        # Convert seg to attention_mask (1 for valid tokens, 0 for padding)
        attention_mask = (seg > 0).long()
        
        # For UER's LM format: src = [t0, t1, t2], tgt = [t1, t2, t3]
        # Qwen expects: input_ids = [t0, t1, t2, t3], labels = [-100, t1, t2, t3]
        # We concatenate the first token of src with all of tgt to form the full sequence
        batch_size, seq_len = src.size()
        
        # Get the first token from each sequence (handling padding)
        first_tokens = []
        for i in range(batch_size):
            valid_positions = (seg[i] > 0).nonzero(as_tuple=True)[0]
            if len(valid_positions) > 0:
                first_tokens.append(src[i, valid_positions[0]].item())
            else:
                first_tokens.append(0)  # padding token
        
        first_tokens_tensor = torch.tensor(first_tokens, device=src.device).unsqueeze(1)
        
        # Concatenate: [first_token] + tgt = [t0, t1, t2, t3]
        input_ids = torch.cat([first_tokens_tensor, tgt], dim=1)
        
        # Create labels: [-100, t1, t2, t3] (don't predict first token, predict rest)
        labels = torch.full_like(input_ids, -100)
        labels[:, 1:] = tgt  # All tgt tokens should be predicted
        
        # Update attention mask: [1] + original mask
        extended_attention_mask = torch.cat([torch.ones(batch_size, 1, device=src.device, dtype=torch.long), attention_mask], dim=1)
        
        # Forward pass through Qwen
        outputs = self.qwen_model(
            input_ids=input_ids,
            attention_mask=extended_attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        
        # Calculate accuracy for compatibility with UER's loss_info format
        logits = outputs.logits
        # Qwen shifts internally: logits[i] predicts labels[i+1]
        # We compare logits[:-1] with labels[1:]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten for accuracy calculation
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        # Only calculate accuracy on non-ignored tokens
        valid_mask = (shift_labels != -100)
        if valid_mask.sum() > 0:
            pred_ids = shift_logits.argmax(dim=-1)
            correct = (pred_ids[valid_mask] == shift_labels[valid_mask]).sum().float()
            denominator = valid_mask.sum().float()
        else:
            correct = torch.tensor(0.0, device=loss.device)
            denominator = torch.tensor(1.0, device=loss.device)
        
        return loss, correct, denominator


def build_model(args):
    """
    Build universial encoder representations models.
    The combinations of different embedding, encoder,
    and target layers yield pretrained models of different
    properties.
    We could select suitable one for downstream tasks.
    """
    
    # Check if using Qwen model from Hugging Face
    use_qwen = getattr(args, 'use_qwen', False) or getattr(args, 'qwen_model_name', None) is not None
    
    if use_qwen:
        try:
            from transformers import AutoModelForCausalLM, AutoConfig
            
            qwen_model_name = getattr(args, 'qwen_model_name', 'Qwen/Qwen2.5-0.5B')
            
            # Load Qwen model
            config = AutoConfig.from_pretrained(qwen_model_name)
            # Use CPU if CUDA is not available
            if not torch.cuda.is_available():
                model = AutoModelForCausalLM.from_pretrained(
                    qwen_model_name,
                    config=config,
                    torch_dtype=torch.float32
                )
                model = model.to('cpu')
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    qwen_model_name,
                    config=config,
                    torch_dtype=torch.float32
                )
            
            # Resize token embeddings to match custom vocab size
            vocab_size = len(args.tokenizer.vocab)
            if model.get_input_embeddings().num_embeddings != vocab_size:
                model.resize_token_embeddings(vocab_size)
                print(f"Resized Qwen embeddings from {config.vocab_size} to {vocab_size}")
            
            # Wrap Qwen model to be compatible with UER interface
            wrapped_model = QwenWrapper(model, args)
            return wrapped_model
            
        except ImportError:
            raise ImportError("transformers library is required for Qwen models. Install with: pip install transformers>=4.35.0")
        except Exception as e:
            raise RuntimeError(f"Failed to load Qwen model: {str(e)}")

    # Original UER model building code
    embedding = Embedding(args)
    for embedding_name in args.embedding:
        tmp_emb = str2embedding[embedding_name](args, len(args.tokenizer.vocab))
        embedding.update(tmp_emb, embedding_name)

    encoder = str2encoder[args.encoder](args)

    if args.decoder is not None:
        if args.data_processor == "mt":
            tgt_vocab_size = len(args.tgt_tokenizer.vocab)
        else:
            tgt_vocab_size = len(args.tokenizer.vocab)

        tgt_embedding = Embedding(args)
        for embedding_name in args.tgt_embedding:
            tmp_emb = str2embedding[embedding_name](args, tgt_vocab_size)
            tgt_embedding.update(tmp_emb, embedding_name)

        decoder = str2decoder[args.decoder](args)
    else:
        tgt_embedding = None
        decoder = None

    target = Target()
    for target_name in args.target:
        if args.data_processor == "mt":
            tmp_target = str2target[target_name](args, len(args.tgt_tokenizer.vocab))
        else:
            tmp_target = str2target[target_name](args, len(args.tokenizer.vocab))
        target.update(tmp_target, target_name)
    model = Model(args, embedding, encoder, tgt_embedding, decoder, target)

    return model
