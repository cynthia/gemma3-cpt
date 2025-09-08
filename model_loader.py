from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoProcessor
import torch
import torch.nn as nn


def load_model_and_tokenizer(
    model_id="google/gemma-3-12b-it",
    vision_model_id="google/siglip-so400m-patch14-384",
    device_map="auto"
):
    """Load Gemma3 with vision encoder for multimodal training."""
    
    # Load language model
    language_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device_map,
    )
    language_model.config.use_cache = False
    
    # Load vision encoder
    vision_model = AutoModel.from_pretrained(
        vision_model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    
    # Load processors
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    vision_processor = AutoProcessor.from_pretrained(vision_model_id, use_fast=True)
    
    # Simple projection layer to align vision and language dimensions
    vision_dim = vision_model.config.vision_config.hidden_size
    language_dim = language_model.config.text_config.hidden_size
    projection = nn.Linear(vision_dim, language_dim, dtype=torch.bfloat16)
    
    if device_map == "auto":
        projection = projection.cuda()
    
    return {
        "language_model": language_model,
        "vision_model": vision_model,
        "projection": projection,
        "tokenizer": tokenizer,
        "vision_processor": vision_processor
    }