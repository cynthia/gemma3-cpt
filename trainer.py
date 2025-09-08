from transformers import Trainer, TrainingArguments
from accelerate import Accelerator
import torch
import torch.nn as nn
from typing import Dict, Optional


class MultimodalTrainer(Trainer):
    """Custom trainer for multimodal continual pre-training."""
    
    def __init__(self, vision_model=None, projection=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vision_model = vision_model
        self.projection = projection
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Extract inputs
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")
        pixel_values = inputs.get("pixel_values")
        
        # Process vision inputs if present
        if pixel_values is not None and self.vision_model is not None:
            # Encode images
            with torch.no_grad():
                vision_outputs = self.vision_model(pixel_values)
                image_features = vision_outputs.last_hidden_state
            
            # Project to language model dimension
            image_embeds = self.projection(image_features)
            
            # Get text embeddings
            inputs_embeds = model.get_input_embeddings()(input_ids)
            
            # Simple fusion: prepend image embeddings to text
            batch_size = inputs_embeds.shape[0]
            seq_length = inputs_embeds.shape[1]
            
            # Flatten image embeddings to sequence dimension
            image_embeds = image_embeds.view(batch_size, -1, image_embeds.shape[-1])
            
            # Concatenate image and text embeddings
            combined_embeds = torch.cat([image_embeds, inputs_embeds], dim=1)
            
            # Adjust attention mask
            image_attention = torch.ones(
                batch_size, image_embeds.shape[1],
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            combined_attention = torch.cat([image_attention, attention_mask], dim=1)
            
            # Adjust labels (no loss on image tokens)
            image_labels = torch.full(
                (batch_size, image_embeds.shape[1]),
                -100,
                dtype=labels.dtype,
                device=labels.device
            )
            combined_labels = torch.cat([image_labels, labels], dim=1)
            
            # Forward pass with combined embeddings
            outputs = model(
                inputs_embeds=combined_embeds,
                attention_mask=combined_attention,
                labels=combined_labels,
            )
        else:
            # Text-only forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def create_training_args(
    output_dir="./outputs",
    num_train_epochs=1,
    per_device_train_batch_size=4096,
    gradient_accumulation_steps=1,
    learning_rate=2e-5,
    warmup_steps=100,
    logging_steps=10,
    save_steps=500,
    max_steps=10000,
    fp16=False,
    bf16=True,
    gradient_checkpointing=False,
    deepspeed_config="deepspeed_config.json",
    local_rank=-1,
    auto_find_batch_size=True,
):
    """Create training arguments for distributed training."""
    
    return TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_strategy="steps",
        eval_strategy="no",
        fp16=fp16,
        bf16=bf16,
        gradient_checkpointing=gradient_checkpointing,
        deepspeed=deepspeed_config if local_rank != -1 else None,
        local_rank=local_rank,
        ddp_find_unused_parameters=False,
        report_to=["wandb"],
        run_name="gemma3-cpt-omnicorpus",
        push_to_hub=False,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        auto_find_batch_size=auto_find_batch_size,
        optim="adamw_torch",
        gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
    )