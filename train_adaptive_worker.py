#!/usr/bin/env python3
"""
DDP Worker for Adaptive Window Selection Training

Combines adaptive window selection with full DDP training features.
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from encodec import EncodecModel
from tqdm import tqdm
import time

# Import custom modules
from dataset_wav_pairs_24sec import AudioPairsDataset24sec, collate_fn
from adaptive_window_agent import AdaptiveWindowCreativeAgent


def setup_ddp(rank, world_size):
    """Initialize DDP"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Cleanup DDP"""
    dist.destroy_process_group()


def encode_audio_batch(audio_batch, encodec_model, target_frames=1200):
    """
    Encode batch of audio with EnCodec.
    
    Args:
        audio_batch: [B, 1, samples] - mono audio
        encodec_model: EnCodec model
        target_frames: Target number of frames (1200 = 24 sec at 50 fps)
    
    Returns:
        encoded: [B, 128, target_frames]
    """
    with torch.no_grad():
        latents = encodec_model.encoder(audio_batch)  # [B, 128, T]
        
        # Resample to target frames if needed
        if latents.shape[-1] != target_frames:
            latents = torch.nn.functional.interpolate(
                latents,
                size=target_frames,
                mode='linear',
                align_corners=False
            )
    
    return latents


def train_epoch(model, dataloader, encodec_model, optimizer, rank, world_size, args, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    total_novelty = 0.0
    num_batches = 0
    
    # Only show progress bar on rank 0
    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    else:
        pbar = dataloader
    
    for batch_idx, (audio_inputs, audio_targets) in enumerate(pbar):
        # Move to device
        audio_inputs = audio_inputs.cuda(rank)
        audio_targets = audio_targets.cuda(rank)
        
        # Unity test: replace target with input
        if args.unity_test:
            audio_targets = audio_inputs.clone()
        
        # Shuffle targets: random pairing
        if args.shuffle_targets:
            indices = torch.randperm(audio_targets.size(0), device=audio_targets.device)
            audio_targets = audio_targets[indices]
        
        # Encode audio
        encoded_inputs = encode_audio_batch(audio_inputs, encodec_model)  # [B, 128, 1200]
        encoded_targets = encode_audio_batch(audio_targets, encodec_model)
        
        # Forward pass through adaptive window agent
        outputs, losses, metadata = model(encoded_inputs, encoded_targets)
        # outputs: List of 3 tensors [B, 128, 800]
        # losses: List of 3 scalars (novelty losses)
        
        # Compute mean novelty loss
        mean_novelty_loss = torch.mean(torch.stack(losses))
        
        # Total loss
        loss = args.mask_reg_weight * mean_novelty_loss
        
        # Add spectral loss if requested
        if args.loss_weight_spectral > 0:
            # Simple MSE on outputs as spectral proxy
            spectral_loss = sum([torch.nn.functional.mse_loss(out, encoded_targets[:, :, :800]) 
                                for out in outputs]) / len(outputs)
            loss = loss + args.loss_weight_spectral * spectral_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # Stats
        total_loss += loss.item()
        total_novelty += mean_novelty_loss.item()
        num_batches += 1
        
        # Update progress bar (rank 0 only)
        if rank == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'novelty': f'{mean_novelty_loss.item():.4f}',
                'pair0_start': f'{metadata["pairs"][0]["start_input_mean"]:.1f}',
                'pair0_ratio': f'{metadata["pairs"][0]["ratio_input_mean"]:.2f}x'
            })
    
    # Synchronize losses across GPUs
    avg_loss_tensor = torch.tensor([total_loss / num_batches], device=rank)
    avg_novelty_tensor = torch.tensor([total_novelty / num_batches], device=rank)
    
    dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(avg_novelty_tensor, op=dist.ReduceOp.SUM)
    
    avg_loss = avg_loss_tensor.item() / world_size
    avg_novelty = avg_novelty_tensor.item() / world_size
    
    return avg_loss, avg_novelty


def validate(model, dataloader, encodec_model, rank, world_size, args):
    """Validate"""
    model.eval()
    
    total_loss = 0.0
    total_novelty = 0.0
    num_batches = 0
    
    with torch.no_grad():
        if rank == 0:
            pbar = tqdm(dataloader, desc="Validation")
        else:
            pbar = dataloader
        
        for audio_inputs, audio_targets in pbar:
            audio_inputs = audio_inputs.cuda(rank)
            audio_targets = audio_targets.cuda(rank)
            
            # Unity test
            if args.unity_test:
                audio_targets = audio_inputs.clone()
            
            # Encode
            encoded_inputs = encode_audio_batch(audio_inputs, encodec_model)
            encoded_targets = encode_audio_batch(audio_targets, encodec_model)
            
            # Forward
            outputs, losses, metadata = model(encoded_inputs, encoded_targets)
            mean_novelty_loss = torch.mean(torch.stack(losses))
            
            loss = args.mask_reg_weight * mean_novelty_loss
            
            if args.loss_weight_spectral > 0:
                spectral_loss = sum([torch.nn.functional.mse_loss(out, encoded_targets[:, :, :800]) 
                                    for out in outputs]) / len(outputs)
                loss = loss + args.loss_weight_spectral * spectral_loss
            
            total_loss += loss.item()
            total_novelty += mean_novelty_loss.item()
            num_batches += 1
    
    # Synchronize
    avg_loss_tensor = torch.tensor([total_loss / num_batches], device=rank)
    avg_novelty_tensor = torch.tensor([total_novelty / num_batches], device=rank)
    
    dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(avg_novelty_tensor, op=dist.ReduceOp.SUM)
    
    avg_loss = avg_loss_tensor.item() / world_size
    avg_novelty = avg_novelty_tensor.item() / world_size
    
    return avg_loss, avg_novelty


def train_adaptive_worker(rank, world_size, args):
    """
    DDP training worker for adaptive window selection
    
    Args:
        rank: GPU rank (0 to world_size-1)
        world_size: Total number of GPUs
        args: Training arguments
    """
    # Setup DDP
    setup_ddp(rank, world_size)
    
    if rank == 0:
        print(f"\n[Rank {rank}] Initializing training...")
    
    # Set random seed
    torch.manual_seed(args.seed + rank)
    
    # Load datasets
    train_dataset = AudioPairsDataset24sec(args.dataset_folder, split='train')
    val_dataset = AudioPairsDataset24sec(args.dataset_folder, split='val')
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    if rank == 0:
        print(f"[Rank {rank}] Train batches: {len(train_loader)}")
        print(f"[Rank {rank}] Val batches: {len(val_loader)}")
    
    # Load EnCodec model
    encodec_model = EncodecModel.encodec_model_24khz()
    encodec_model.set_target_bandwidth(args.encodec_bandwidth)
    encodec_model = encodec_model.cuda(rank)
    encodec_model.eval()
    for param in encodec_model.parameters():
        param.requires_grad = False
    
    if rank == 0:
        print(f"[Rank {rank}] EnCodec loaded and frozen")
    
    # Initialize adaptive window agent
    model = AdaptiveWindowCreativeAgent(
        encoding_dim=args.encoding_dim,
        num_pairs=args.num_pairs
    )
    model = model.cuda(rank)
    
    # Wrap with DDP
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Rank {rank}] Total parameters: {total_params/1e6:.1f}M")
        print(f"[Rank {rank}] Trainable parameters: {trainable_params/1e6:.1f}M")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Resume from checkpoint if specified
    start_epoch = 1
    best_val_loss = float('inf')
    
    if args.resume and rank == 0:
        if os.path.exists(args.resume):
            print(f"[Rank {rank}] Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=f'cuda:{rank}')
            model.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            print(f"[Rank {rank}] Resumed from epoch {checkpoint['epoch']}")
    
    # Broadcast start_epoch to all ranks
    start_epoch_tensor = torch.tensor([start_epoch], device=rank)
    dist.broadcast(start_epoch_tensor, src=0)
    start_epoch = start_epoch_tensor.item()
    
    if rank == 0:
        print(f"\n[Rank {rank}] Starting training from epoch {start_epoch}")
        print("="*80)
    
    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        # Set epoch for distributed sampler
        train_sampler.set_epoch(epoch)
        
        # Train
        train_loss, train_novelty = train_epoch(
            model, train_loader, encodec_model, optimizer,
            rank, world_size, args, epoch
        )
        
        # Validate
        val_loss, val_novelty = validate(
            model, val_loader, encodec_model, rank, world_size, args
        )
        
        # Print results (rank 0 only)
        if rank == 0:
            print(f"\nEpoch {epoch} Results:")
            print(f"  Train Loss: {train_loss:.4f} (novelty: {train_novelty:.4f})")
            print(f"  Val Loss: {val_loss:.4f} (novelty: {val_novelty:.4f})")
            
            # Save checkpoint
            if epoch % args.save_every == 0 or val_loss < best_val_loss:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'args': vars(args)
                }
                
                checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
                torch.save(checkpoint, checkpoint_path)
                print(f"  Saved checkpoint: {checkpoint_path}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
                    torch.save(checkpoint, best_path)
                    print(f"  ✓ New best model! Val loss: {val_loss:.4f}")
        
        # Synchronize before next epoch
        dist.barrier()
    
    if rank == 0:
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Checkpoints saved in: {args.checkpoint_dir}/")
    
    # Cleanup
    cleanup_ddp()
