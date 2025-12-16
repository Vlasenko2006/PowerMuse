"""
Pretrain transformer as autoencoder on encoder outputs.
Goal: Learn to reconstruct encoder outputs with RMS loss < 0.004
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from pathlib import Path
import argparse
from tqdm import tqdm

from model_fusion_continuation import MaskedMultiPatternFusion
from dataset_continuation import StructuredAudioDataset


def rms_loss(output, target):
    """RMS (Root Mean Square) loss"""
    return torch.sqrt(torch.mean((output - target) ** 2))


def pretrain_epoch(model, dataloader, optimizer, device, rank, epoch):
    """Pretrain transformer for one epoch"""
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Pretrain]")
    else:
        pbar = dataloader
    
    for batch_idx, (masked_inputs, targets, masks) in enumerate(pbar):
        masked_inputs = masked_inputs.to(device)
        targets = targets.to(device)
        masks = masks.to(device)
        
        batch_size, num_patterns = masked_inputs.shape[0], masked_inputs.shape[1]
        
        # Get encoder outputs (frozen)
        with torch.no_grad():
            encodec_model = model.module.encodec if hasattr(model, 'module') else model.encodec
            
            # Encode all patterns
            emb = encodec_model.encoder(masked_inputs.view(-1, 2, masked_inputs.shape[-1]))
            codes = encodec_model.quantizer.encode(emb)
            encoded = encodec_model.quantizer.decode(codes)
            
            # Reshape: [B*3, D, T_enc] -> [B, 3, D, T_enc]
            encoded = encoded.view(batch_size, num_patterns, encoded.size(1), encoded.size(2))
            # [B, 3, D, T_enc] -> [B*3, T_enc, D]
            encoded = encoded.transpose(2, 3).flatten(0, 1)
            
            # Store input RMS for normalization
            input_rms = torch.sqrt((encoded ** 2).mean() + 1e-8)
        
        # Separate into 3 patterns
        encoded_list = []
        for i in range(num_patterns):
            enc = encoded[i::num_patterns]  # [B, T_enc, D]
            encoded_list.append(enc)
        
        # Concatenate patterns
        concat_encoded = torch.cat(encoded_list, dim=-1)  # [B, T_enc, 3*D]
        
        # Normalize input to RMS=1.0
        concat_rms = torch.sqrt((concat_encoded ** 2).mean() + 1e-8)
        normalized_input = concat_encoded / concat_rms
        
        # Forward through transformer (trainable)
        model_to_use = model.module if hasattr(model, 'module') else model
        transformer_output = model_to_use.transformer(normalized_input)
        
        # Normalize transformer output to RMS=1.0
        transformer_rms = torch.sqrt((transformer_output ** 2).mean() + 1e-8)
        transformer_output = transformer_output / transformer_rms
        
        # Target: normalized input (identity mapping in normalized space)
        target_normalized = normalized_input
        
        # Compute RMS loss
        loss = rms_loss(transformer_output, target_normalized)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        # Accumulate metrics
        total_loss += loss.item()
        num_batches += 1
        
        if rank == 0:
            pbar.set_postfix({'rms_loss': f'{loss.item():.6f}'})
        
        # Debug first batch
        if batch_idx == 0 and rank == 0:
            print(f"\n📊 First Batch Debug:")
            print(f"  Input RMS: {concat_rms.item():.6f}")
            print(f"  Transformer output RMS (before norm): {transformer_rms.item():.6f}")
            print(f"  RMS loss: {loss.item():.6f}")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate_epoch(model, dataloader, device, rank):
    """Validate transformer pretraining"""
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for masked_inputs, targets, masks in dataloader:
            masked_inputs = masked_inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            
            batch_size, num_patterns = masked_inputs.shape[0], masked_inputs.shape[1]
            
            # Get encoder outputs
            encodec_model = model.module.encodec if hasattr(model, 'module') else model.encodec
            emb = encodec_model.encoder(masked_inputs.view(-1, 2, masked_inputs.shape[-1]))
            codes = encodec_model.quantizer.encode(emb)
            encoded = encodec_model.quantizer.decode(codes)
            encoded = encoded.view(batch_size, num_patterns, encoded.size(1), encoded.size(2))
            encoded = encoded.transpose(2, 3).flatten(0, 1)
            
            # Separate and concatenate
            encoded_list = []
            for i in range(num_patterns):
                enc = encoded[i::num_patterns]
                encoded_list.append(enc)
            concat_encoded = torch.cat(encoded_list, dim=-1)
            
            # Normalize
            concat_rms = torch.sqrt((concat_encoded ** 2).mean() + 1e-8)
            normalized_input = concat_encoded / concat_rms
            
            # Forward through transformer
            model_to_use = model.module if hasattr(model, 'module') else model
            transformer_output = model_to_use.transformer(normalized_input)
            
            # Normalize output
            transformer_rms = torch.sqrt((transformer_output ** 2).mean() + 1e-8)
            transformer_output = transformer_output / transformer_rms
            
            # Compute loss
            loss = rms_loss(transformer_output, normalized_input)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Pretrain Transformer Autoencoder')
    parser.add_argument('--dataset_folder', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_patterns', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--target_rms', type=float, default=0.004)
    parser.add_argument('--checkpoint_folder', type=str, default='checkpoints_pretrain')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    
    args = parser.parse_args()
    
    # Setup DDP
    rank = args.local_rank
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', init_method='env://', 
                           world_size=args.world_size, rank=rank)
    device = torch.device(f'cuda:{rank}')
    
    if rank == 0:
        print("="*80)
        print("TRANSFORMER PRETRAINING")
        print("="*80)
        print(f"Target RMS loss: {args.target_rms}")
        print(f"Max epochs: {args.max_epochs}")
    
    # Create datasets
    train_dataset = StructuredAudioDataset(
        folder=args.dataset_folder,
        split='train',
        num_patterns=args.num_patterns
    )
    
    val_dataset = StructuredAudioDataset(
        folder=args.dataset_folder,
        split='val',
        num_patterns=args.num_patterns
    )
    
    # Create samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, 
                                       rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=args.world_size, 
                                     rank=rank, shuffle=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = MaskedMultiPatternFusion(
        d_model=args.d_model,
        nhead=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        mask_type='zero'
    ).to(device)
    
    model = DDP(model, device_ids=[rank])
    
    # Only train transformer parameters
    for name, param in model.named_parameters():
        if 'transformer' not in name:
            param.requires_grad = False
    
    # Optimizer - only transformer parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    
    if rank == 0:
        num_trainable = sum(p.numel() for p in trainable_params)
        print(f"\nTrainable parameters: {num_trainable:,}")
        Path(args.checkpoint_folder).mkdir(exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, args.max_epochs + 1):
        if rank == 0:
            print(f"\n{'='*80}")
            print(f"Epoch {epoch}/{args.max_epochs}")
            print(f"{'='*80}")
        
        train_sampler.set_epoch(epoch)
        
        # Train
        train_loss = pretrain_epoch(model, train_loader, optimizer, device, rank, epoch)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, device, rank)
        
        if rank == 0:
            print(f"\nEpoch {epoch} Results:")
            print(f"  Train RMS loss: {train_loss:.6f}")
            print(f"  Val RMS loss:   {val_loss:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'args': vars(args)
                }
                torch.save(checkpoint, Path(args.checkpoint_folder) / 'pretrained_transformer.pt')
                print(f"  ✅ Saved best model (val_loss: {val_loss:.6f})")
            
            # Check if target reached
            if val_loss < args.target_rms:
                print(f"\n🎉 Target RMS loss {args.target_rms} reached!")
                print(f"   Final val loss: {val_loss:.6f}")
                print(f"   Training complete after {epoch} epochs")
                break
    
    if rank == 0:
        print(f"\n{'='*80}")
        print("Pretraining complete!")
        print(f"Best validation RMS loss: {best_val_loss:.6f}")
        print(f"{'='*80}")
    
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
