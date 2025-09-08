from datasets import load_dataset, IterableDataset
from torch.utils.data import DataLoader
from transformers import default_data_collator
import torch
import torch.nn.functional
import gzip
import json
from huggingface_hub import hf_hub_download, list_repo_files


class MultimodalDataCollator:
    """Simple collator for multimodal data."""
    
    def __init__(self, tokenizer, vision_processor):
        self.tokenizer = tokenizer
        self.vision_processor = vision_processor
    
    def __call__(self, examples):
        # Process text
        texts = []
        for ex in examples:
            text = ex.get("text", "")
            # Ensure text is not empty
            if not text or text.strip() == "":
                text = "This is a placeholder text for training."
            texts.append(text)
        
        text_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        )
        
        # Process images if present
        images = [ex.get("image") for ex in examples]
        if any(img is not None for img in images):
            vision_inputs = self.vision_processor(
                images=[img for img in images if img is not None],
                return_tensors="pt"
            )
        else:
            vision_inputs = None
        
        # Create labels for language modeling
        labels = text_inputs["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # Ensure minimum sequence length
        if labels.shape[1] < 2:
            # Pad to at least 2 tokens
            padding_size = 2 - labels.shape[1]
            labels = torch.nn.functional.pad(labels, (0, padding_size), value=-100)
            text_inputs["input_ids"] = torch.nn.functional.pad(
                text_inputs["input_ids"], (0, padding_size), value=self.tokenizer.pad_token_id
            )
            text_inputs["attention_mask"] = torch.nn.functional.pad(
                text_inputs["attention_mask"], (0, padding_size), value=0
            )
        
        batch = {
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "labels": labels,
        }
        
        if vision_inputs is not None:
            batch["pixel_values"] = vision_inputs["pixel_values"]
        
        return batch


def create_dataloaders(
    tokenizer,
    vision_processor,
    batch_size=1,
    num_workers=4,
    dataset_name="OpenGVLab/OmniCorpus-CC-210M",
    streaming=True
):
    """Create DataLoader for OmniCorpus dataset."""
    
    # Load OmniCorpus dataset
    dataset = load_dataset(
        dataset_name,
        streaming=streaming,
        split="train",
    )
    
    # Create collator
    collator = MultimodalDataCollator(tokenizer, vision_processor)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return dataloader