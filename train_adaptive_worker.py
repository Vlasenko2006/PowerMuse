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
    total_spectral = 0.0
    num_batches = 0
    
    # Track window selection statistics
    total_pair0_start = 0.0
    total_pair1_start = 0.0
    total_pair2_start = 0.0
    total_pair0_ratio = 0.0
    total_pair1_ratio = 0.0
    total_pair2_ratio = 0.0
    total_pair0_tonality = 0.0
    total_pair1_tonality = 0.0
    total_pair2_tonality = 0.0
    
    # Only show progress bar on rank 0
    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", ncols=140)
    else:
        pbar = dataloader
    
    for batch_idx, (audio_inputs, audio_targets) in enumerate(pbar):
        # Move to device
        audio_inputs = audio_inputs.cuda(rank)
        audio_targets = audio_targets.cuda(rank)
        
        # Unity test: replace target with input
        if args.unity_test:
            audio_targets = audio_inputs.clone()
            if rank == 0 and batch_idx == 0:
                print(f"\n🔍 Unity Test ENABLED: target = input (sanity check)")
        
        # Shuffle targets: random pairing
        if args.shuffle_targets:
            indices = torch.randperm(audio_targets.size(0), device=audio_targets.device)
            audio_targets = audio_targets[indices]
            if rank == 0 and batch_idx == 0:
                print(f"🎲 Shuffle Targets ENABLED: random pairing for creativity")
        
        # Encode audio
        encoded_inputs = encode_audio_batch(audio_inputs, encodec_model)  # [B, 128, 1200]
        encoded_targets = encode_audio_batch(audio_targets, encodec_model)
        
        # Forward pass through adaptive window agent
        outputs, losses, metadata = model(encoded_inputs, encoded_targets)
        # outputs: List of 3 tensors [B, 128, 800]
        # losses: List of 3 scalars (novelty losses)
        # metadata: Dict with window selection info
        
        # Compute mean novelty loss
        mean_novelty_loss = torch.mean(torch.stack(losses))
        
        # Total loss
        loss = args.mask_reg_weight * mean_novelty_loss
        spectral_loss_value = 0.0
        
        # Add spectral loss if requested
        if args.loss_weight_spectral > 0:
            # Simple MSE on outputs as spectral proxy
            spectral_loss = sum([torch.nn.functional.mse_loss(out, encoded_targets[:, :, :800]) 
                                for out in outputs]) / len(outputs)
            spectral_loss_value = spectral_loss.item()
            loss = loss + args.loss_weight_spectral * spectral_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Debug: Check gradients on first batch of first epoch
        if batch_idx == 0 and epoch == 1 and rank == 0:
            print(f"\n{'='*80}")
            print(f"GRADIENT DEBUGGING (First batch, Epoch {epoch})")
            print(f"{'='*80}")
            total_grad_norm = 0.0
            grad_info = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm ** 2
                    grad_info.append((name, grad_norm, param.grad.abs().mean().item(), param.grad.abs().max().item()))
            
            total_grad_norm = total_grad_norm ** 0.5
            print(f"Total gradient norm (before clipping): {total_grad_norm:.6f}")
            
            # Show top 10 gradients by norm
            print("\nTop 10 parameters by gradient norm:")
            for name, norm, mean, max_val in sorted(grad_info, key=lambda x: -x[1])[:10]:
                print(f"  {name:50s}: norm={norm:.6f}, mean={mean:.6f}, max={max_val:.6f}")
            
            # Check for zero gradients
            zero_grad = [name for name, norm, _, _ in grad_info if norm < 1e-8]
            if zero_grad:
                print(f"\n⚠️  WARNING: {len(zero_grad)} parameters have near-zero gradients:")
                for name in zero_grad[:5]:
                    print(f"    {name}")
                if len(zero_grad) > 5:
                    print(f"    ... and {len(zero_grad)-5} more")
            
            print(f"{'='*80}\n")
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # Stats
        total_loss += loss.item()
        total_novelty += mean_novelty_loss.item()
        total_spectral += spectral_loss_value
        num_batches += 1
        
        # Track window selection statistics
        if 'pairs' in metadata and len(metadata['pairs']) >= 3:
            total_pair0_start += metadata['pairs'][0]['start_input_mean']
            total_pair1_start += metadata['pairs'][1]['start_input_mean']
            total_pair2_start += metadata['pairs'][2]['start_input_mean']
            total_pair0_ratio += metadata['pairs'][0]['ratio_input_mean']
            total_pair1_ratio += metadata['pairs'][1]['ratio_input_mean']
            total_pair2_ratio += metadata['pairs'][2]['ratio_input_mean']
            total_pair0_tonality += metadata['pairs'][0].get('tonality_input_mean', 0.0)
            total_pair1_tonality += metadata['pairs'][1].get('tonality_input_mean', 0.0)
            total_pair2_tonality += metadata['pairs'][2].get('tonality_input_mean', 0.0)
        
        # Update progress bar (rank 0 only)
        if rank == 0:
            postfix = {
                'loss': f'{loss.item():.4f}',
                'novelty': f'{mean_novelty_loss.item():.4f}',
            }
            
            if args.loss_weight_spectral > 0:
                postfix['spectral'] = f'{spectral_loss_value:.4f}'
            
            # Show window selection info for first pair
            if 'pairs' in metadata and len(metadata['pairs']) > 0:
                postfix['p0_start'] = f'{metadata["pairs"][0]["start_input_mean"]:.0f}'
                postfix['p0_ratio'] = f'{metadata["pairs"][0]["ratio_input_mean"]:.2f}x'
            
            pbar.set_postfix(postfix)
        
        # Debug printout for first batch of first epoch
        if rank == 0 and batch_idx == 0 and epoch == 1:
            print(f"\n🎯 First Batch Window Selection:")
            for i, pair in enumerate(metadata['pairs'][:3]):
                print(f"  Pair {i}: start={pair['start_input_mean']:.1f}f, "
                      f"ratio={pair['ratio_input_mean']:.2f}x, "
                      f"tonality={pair.get('tonality_input_mean', 0.0):.2f}")
    
    # Synchronize losses across GPUs
    avg_loss_tensor = torch.tensor([total_loss / num_batches], device=rank)
    avg_novelty_tensor = torch.tensor([total_novelty / num_batches], device=rank)
    avg_spectral_tensor = torch.tensor([total_spectral / num_batches], device=rank)
    
    dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(avg_novelty_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(avg_spectral_tensor, op=dist.ReduceOp.SUM)
    
    avg_loss = avg_loss_tensor.item() / world_size
    avg_novelty = avg_novelty_tensor.item() / world_size
    avg_spectral = avg_spectral_tensor.item() / world_size
    
    # Compute window statistics
    window_stats = None
    if num_batches > 0:
        window_stats = {
            'pair0_start': total_pair0_start / num_batches,
            'pair1_start': total_pair1_start / num_batches,
            'pair2_start': total_pair2_start / num_batches,
            'pair0_ratio': total_pair0_ratio / num_batches,
            'pair1_ratio': total_pair1_ratio / num_batches,
            'pair2_ratio': total_pair2_ratio / num_batches,
            'pair0_tonality': total_pair0_tonality / num_batches,
            'pair1_tonality': total_pair1_tonality / num_batches,
            'pair2_tonality': total_pair2_tonality / num_batches,
        }
    
    return avg_loss, avg_novelty, avg_spectral, window_stats


def _create_waveform_visualization(audio, label, num_segments=50):
    """
    Create a text-based waveform visualization showing loudness over time.
    
    Args:
        audio: 1D audio tensor [samples]
        label: Label for the waveform (e.g., "Input", "Target", "Output")
        num_segments: Number of time segments to display (default: 50)
    
    # Store first batch samples for waveform visualization
    first_input_audio = None
    first_target_audio = None
    first_output_audio = None
    output_input_corr_sum = 0.0
    output_target_corr_sum = 0.0
    
    with torch.no_grad():
        if rank == 0:
            pbar = tqdm(dataloader, desc="Validation", ncols=120)
        else:
            pbar = dataloader
        
        for batch_idx, (audio_inputs, audio_targets) in enumerate(pbar):
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
            spectral_loss_value = 0.0
            
            if args.loss_weight_spectral > 0:
                spectral_loss = sum([torch.nn.functional.mse_loss(out, encoded_targets[:, :, :800]) 
                                    for out in outputs]) / len(outputs)
                spectral_loss_value = spectral_loss.item()
                loss = loss + args.loss_weight_spectral * spectral_loss
            
            # Store first sample for visualization (use first output pair)
            if first_input_audio is None and rank == 0 and len(outputs) > 0:
                # Decode to get audio waveforms
                output_audio = encodec_model.decoder(outputs[0][:1])  # First sample, first pair [1, 1, samples]
                input_audio = audio_inputs[:1]  # [1, 1, samples]
                target_audio = audio_targets[:1]  # [1, 1, samples]
                
                first_input_audio = input_audio[0, 0].cpu()  # [samples]
                first_target_audio = target_audio[0, 0].cpu()
                first_output_audio = output_audio[0, 0].cpu()
                
                # Compute correlations for this sample
                import numpy as np
                out_np = first_output_audio.numpy()
                tgt_np = first_target_audio.numpy()
                in_np = first_input_audio.numpy()
                
                # Ensure same length for correlation
                min_len = min(len(out_np), len(tgt_np), len(in_np))
                out_np = out_np[:min_len]
                tgt_np = tgt_np[:min_len]
                in_np = in_np[:min_len]
                
                corr_out_tgt = np.corrcoef(out_np, tgt_np)[0, 1]
                corr_out_in = np.corrcoef(out_np, in_np)[0, 1]
                output_target_corr_sum = corr_out_tgt
                output_input_corr_sum = corr_out_in
            
            total_loss += loss.item()
            total_novelty += mean_novelty_loss.item()
            total_spectral += spectral_loss_value
            num_batches += 1
            
            if rank == 0:
                postfix = {
                    'loss': f'{loss.item():.4f}',
                    'novelty': f'{mean_novelty_loss.item():.4f}'
                }
                if args.loss_weight_spectral > 0:
                    postfix['spectral'] = f'{spectral_loss_value:.4f}'
                pbar.set_postfix(postfix)
    
    # Display waveform visualization on rank 0
    if rank == 0 and first_input_audio is not None:
        print()  # New line after validation progress bar
        
        # Compute RMS levels
        input_rms = torch.sqrt(torch.mean(first_input_audio ** 2))
        target_rms = torch.sqrt(torch.mean(first_target_audio ** 2))
        output_rms = torch.sqrt(torch.mean(first_output_audio ** 2))
        
        # Display waveforms
        print(f"\n  🎵 Waveform Visualization (First validation sample):")
        print(f"  {_create_waveform_visualization(first_input_audio, 'Input ')}")
        print(f"  {_create_waveform_visualization(first_target_audio, 'Target')}")
        print(f"  {_create_waveform_visualization(first_output_audio, 'Output')}")
        
        # Display statistics
        rms_ratio = output_rms.item() / input_rms.item() if input_rms.item() > 1e-8 else 0.0
        print(f"\n  📊 Audio Statistics:")
        print(f"     RMS: Input={input_rms.item():.4f}  Target={target_rms.item():.4f}  Output={output_rms.item():.4f}")
        print(f"     Out/In ratio: {rms_ratio:.2f}")
        print(f"     Correlation: Out→Target={output_target_corr_sum:.3f}  Out→Input={output_input_corr_sum:.3f}")
        
        if abs(output_target_corr_sum) > abs(output_input_corr_sum) * 1.5:
            print(f"     ⚠️  Output is highly correlated with TARGET (possible copying)")
        elif abs(output_input_corr_sum) > abs(output_target_corr_sum) * 1.5:
            print(f"     ⚠️  Output is highly correlated with INPUT (possible identity)")
        else:
            print(f"     ✓ Output appears to mix both sources")
        print(
    
    with torch.no_grad():
        if rank == 0:
            pbar = tqdm(dataloader, desc="Validation", ncols=120)
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
            spectral_loss_value = 0.0
            
            if args.loss_weight_spectral > 0:
                spectral_loss = sum([torch.nn.functional.mse_loss(out, encoded_targets[:, :, :800]) 
                                    for out in outputs]) / len(outputs)
                spectral_loss_value = spectral_loss.item()
                loss = loss + args.loss_weight_spectral * spectral_loss
            
            total_loss += loss.item()
            total_novelty += mean_novelty_loss.item()
            total_spectral += spectral_loss_value
            num_batches += 1
            
            if rank == 0:
                postfix = {
                    'loss': f'{loss.item():.4f}',
                    'novelty': f'{mean_novelty_loss.item():.4f}'
                }
                if args.loss_weight_spectral > 0:
                    postfix['spectral'] = f'{spectral_loss_value:.4f}'
                pbar.set_postfix(postfix)
    
    # Synchronize
    avg_loss_tensor = torch.tensor([total_loss / num_batches], device=rank)
    avg_novelty_tensor = torch.tensor([total_novelty / num_batches], device=rank)
    avg_spectral_tensor = torch.tensor([total_spectral / num_batches], device=rank)
    
    dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(avg_novelty_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(avg_spectral_tensor, op=dist.ReduceOp.SUM)
    
    avg_loss = avg_loss_tensor.item() / world_size
    avg_novelty = avg_novelty_tensor.item() / world_size
    avg_spectral = avg_spectral_tensor.item() / world_size
    
    return avg_loss, avg_novelty, avg_spectral


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
        train_loss, train_novelty, train_spectral, train_window_stats = train_epoch(
            model, train_loader, encodec_model, optimizer,
            rank, world_size, args, epoch
        )
        
        # Validate
        val_loss, val_novelty, val_spectral = validate(
            model, val_loader, encodec_model, rank, world_size, args
        )
        
        # Print results (rank 0 only)
        if rank == 0:
            print(f"\n{'='*80}")
            print(f"Epoch {epoch}/{args.epochs} Results:")
            print(f"{'='*80}")
            print(f"  📊 Train Metrics:")
            print(f"     Total Loss:    {train_loss:.4f}")
            print(f"     Novelty Loss:  {train_novelty:.4f} (weight: {args.mask_reg_weight})")
            if args.loss_weight_spectral > 0:
                print(f"     Spectral Loss: {train_spectral:.4f} (weight: {args.loss_weight_spectral})")
            
            print(f"\n  🎯 Adaptive Window Selection (Training):")
            if train_window_stats:
                print(f"     Pair 0: start={train_window_stats['pair0_start']:6.1f}f  ratio={train_window_stats['pair0_ratio']:.2f}x  tonality={train_window_stats['pair0_tonality']:.2f}")
                print(f"     Pair 1: start={train_window_stats['pair1_start']:6.1f}f  ratio={train_window_stats['pair1_ratio']:.2f}x  tonality={train_window_stats['pair1_tonality']:.2f}")
                print(f"     Pair 2: start={train_window_stats['pair2_start']:6.1f}f  ratio={train_window_stats['pair2_ratio']:.2f}x  tonality={train_window_stats['pair2_tonality']:.2f}")
                
                # Check window diversity
                start_std = torch.std(torch.tensor([
                    train_window_stats['pair0_start'],
                    train_window_stats['pair1_start'],
                    train_window_stats['pair2_start']
                ])).item()
                
                ratio_std = torch.std(torch.tensor([
                    train_window_stats['pair0_ratio'],
                    train_window_stats['pair1_ratio'],
                    train_window_stats['pair2_ratio']
                ])).item()
                
                print(f"     Diversity: start_std={start_std:.1f}f, ratio_std={ratio_std:.3f}x")
                
                if start_std < 10:
                    print(f"     ⚠️  WARNING: Low window diversity (all selecting similar positions)")
                elif start_std > 50:
                    print(f"     ✓ Good window diversity")
            
            print(f"\n  📈 Validation Metrics:")
            print(f"     Total Loss:    {val_loss:.4f}")
            print(f"     Novelty Loss:  {val_novelty:.4f}")
            if args.loss_weight_spectral > 0:
                print(f"     Spectral Loss: {val_spectral:.4f}")
            
            print(f"\n  ⚙️  Training Config:")
            print(f"     Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"     Batch Size:    {args.batch_size} × {world_size} GPUs = {args.batch_size * world_size}")
            print(f"     Num Pairs:     {args.num_pairs}")
            print(f"{'='*80}\n")
            
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
