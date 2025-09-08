import os
import torch
from transformers import set_seed
from model_loader import load_model_and_tokenizer
from data_loader import create_dataloaders
from trainer import MultimodalTrainer, create_training_args
import argparse


def main():
    parser = argparse.ArgumentParser(description="Gemma3 Continual Pre-training")
    parser.add_argument("--model_id", type=str, default="google/gemma-3-12b-it")
    parser.add_argument("--vision_model_id", type=str, default="google/siglip-so400m-patch14-384")
    parser.add_argument("--dataset_name", type=str, default="OpenGVLab/OmniCorpus-CC-210M")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--use_deepspeed", action="store_true")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load models and processors
    print("Loading models and tokenizers...")
    components = load_model_and_tokenizer(
        model_id=args.model_id,
        vision_model_id=args.vision_model_id,
        device_map="auto" if args.local_rank == -1 else None
    )
    
    language_model = components["language_model"]
    vision_model = components["vision_model"]
    projection = components["projection"]
    tokenizer = components["tokenizer"]
    vision_processor = components["vision_processor"]
    
    # Freeze vision model
    for param in vision_model.parameters():
        param.requires_grad = False
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_dataloader = create_dataloaders(
        tokenizer=tokenizer,
        vision_processor=vision_processor,
        batch_size=args.batch_size,
        dataset_name=args.dataset_name,
        streaming=True
    )
    
    # Create training arguments
    training_args = create_training_args(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        gradient_checkpointing=False,
        deepspeed_config="deepspeed_config.json" if args.use_deepspeed else None,
        local_rank=args.local_rank,
    )
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = MultimodalTrainer(
        model=language_model,
        vision_model=vision_model,
        projection=projection,
        args=training_args,
        train_dataset=train_dataloader.dataset,
        tokenizer=tokenizer,
        data_collator=train_dataloader.collate_fn,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Save projection layer separately
    torch.save(projection.state_dict(), os.path.join(args.output_dir, "projection.pt"))
    
    print("Training complete!")


if __name__ == "__main__":
    main()