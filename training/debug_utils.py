"""
Training debugging and tracking utilities.

Functions for:
- Gradient debugging and analysis
- Training progress printing
- Metric tracking and logging
"""

import torch


def debug_gradients(model, epoch, rank):
    """
    Analyze and print gradient statistics for debugging.
    
    Args:
        model: The model to analyze gradients for
        epoch: Current epoch number
        rank: Process rank (for distributed training)
    """
    if rank != 0:
        return
    
    print(f"\n{'='*80}")
    print(f"GRADIENT DEBUGGING (First batch, Epoch {epoch})")
    print(f"{'='*80}")
    
    total_grad_norm = 0.0
    grad_info = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm ** 2
            grad_info.append((
                name,
                grad_norm,
                param.grad.abs().mean().item(),
                param.grad.abs().max().item()
            ))
    
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


def print_training_progress(epoch, batch_idx, total_batches, loss, novelty_loss, 
                            rms_input, rms_target, rank, print_every=20):
    """
    Print training progress periodically.
    
    Args:
        epoch: Current epoch number
        batch_idx: Current batch index
        total_batches: Total number of batches
        loss: Current loss value
        novelty_loss: Current novelty loss value
        rms_input: RMS input loss value
        rms_target: RMS target loss value
        rank: Process rank (for distributed training)
        print_every: Print every N batches
    """
    if rank != 0:
        return
    
    if batch_idx % print_every == 0 or batch_idx == total_batches - 1:
        progress_pct = (batch_idx + 1) / total_batches * 100
        
        # Handle tensor vs float values
        loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
        novelty_val = novelty_loss.item() if isinstance(novelty_loss, torch.Tensor) else novelty_loss
        rms_in_val = rms_input.item() if isinstance(rms_input, torch.Tensor) else rms_input
        rms_tgt_val = rms_target.item() if isinstance(rms_target, torch.Tensor) else rms_target
        
        # Single-line update with carriage return
        # Use \r to return to start of line, end='' to prevent newline, flush=True to update immediately
        print(f"\rEpoch {epoch}: {batch_idx+1}/{total_batches} ({progress_pct:.0f}%) - "
              f"loss={loss_val:.4f}, novelty={novelty_val:.4f}, "
              f"rms_in={rms_in_val:.4f}, rms_tgt={rms_tgt_val:.4f}", 
              end='', flush=True)
        
        # Print newline only at the end of epoch
        if batch_idx == total_batches - 1:
            print()  # Final newline


def print_window_selection_debug(metadata, epoch, rank):
    """
    Print window selection parameters for the first batch.
    
    Args:
        metadata: Metadata dictionary containing pair information
        epoch: Current epoch number
        rank: Process rank (for distributed training)
    """
    if rank != 0 or epoch != 1:
        return
    
    print(f"\n🎯 First Batch Window Selection:")
    if 'pairs' in metadata:
        for i, pair in enumerate(metadata['pairs'][:3]):
            print(f"  Pair {i}:")
            print(f"    Input:  start={pair.get('start_input_mean', 0):.1f}f, "
                  f"ratio={pair.get('ratio_input_mean', 0):.2f}x (compression)")
            print(f"    Target: start={pair.get('start_target_mean', 0):.1f}f, "
                  f"ratio={pair.get('ratio_target_mean', 0):.2f}x (compression)")
            print(f"    Tonality strength: {pair.get('tonality_strength_mean', 0):.2f}")


def print_epoch_summary(epoch, metrics, window_stats, rank):
    """
    Print comprehensive epoch summary with all metrics.
    
    Args:
        epoch: Current epoch number
        metrics: Dictionary with all averaged metrics
        window_stats: Dictionary with window selection statistics (or None)
        rank: Process rank (for distributed training)
    """
    if rank != 0:
        return
    
    print(f"\n{'='*80}")
    print(f"EPOCH {epoch} SUMMARY")
    print(f"{'='*80}")
    
    # Core losses
    print(f"Loss:          {metrics['loss']:.6f}")
    print(f"RMS Input:     {metrics['rms_input']:.6f}")
    print(f"RMS Target:    {metrics['rms_target']:.6f}")
    print(f"Spectral:      {metrics['spectral']:.6f}")
    print(f"Mel:           {metrics['mel']:.6f}")
    print(f"Novelty:       {metrics['novelty']:.6f}")
    print(f"Corr Penalty:  {metrics['corr_penalty']:.6f}")
    
    # NEW: Ratio supervision losses
    if 'ratio_diversity' in metrics:
        print(f"\nRatio Supervision:")
        print(f"  Ratio Diversity:   {metrics['ratio_diversity']:.6f}")
        print(f"  Reconstruction:    {metrics['reconstruction']:.6f}")
    
    # Balance loss if present
    if 'balance_loss' in metrics and metrics['balance_loss'] > 0:
        print(f"Balance Loss:  {metrics['balance_loss']:.6f}")
    
    # GAN metrics if present
    if 'gan_loss' in metrics and metrics['gan_loss'] > 0:
        print(f"\nGAN Metrics:")
        print(f"  Generator Loss:      {metrics['gan_loss']:.6f}")
        print(f"  Discriminator Loss:  {metrics['disc_loss']:.6f}")
        print(f"  Disc Real Accuracy:  {metrics['disc_real_acc']:.3f}")
        print(f"  Disc Fake Accuracy:  {metrics['disc_fake_acc']:.3f}")
    
    # Correlation metrics
    if 'output_input_corr' in metrics:
        print(f"\nCorrelations:")
        print(f"  Output-Input:  {metrics['output_input_corr']:.4f}")
        print(f"  Output-Target: {metrics['output_target_corr']:.4f}")
    
    # Window statistics if present
    if window_stats is not None:
        print(f"\nWindow Selection Statistics (Averaged over Epoch):")
        print(f"  Pair 0:")
        print(f"    Input:  start={window_stats['pair0_start_in']:.1f}f, ratio={window_stats['pair0_ratio_in']:.2f}x")
        print(f"    Target: start={window_stats['pair0_start_tgt']:.1f}f, ratio={window_stats['pair0_ratio_tgt']:.2f}x")
        print(f"    Tonality: {window_stats['pair0_tonality']:.2f}")
        print(f"  Pair 1:")
        print(f"    Input:  start={window_stats['pair1_start_in']:.1f}f, ratio={window_stats['pair1_ratio_in']:.2f}x")
        print(f"    Target: start={window_stats['pair1_start_tgt']:.1f}f, ratio={window_stats['pair1_ratio_tgt']:.2f}x")
        print(f"    Tonality: {window_stats['pair1_tonality']:.2f}")
        print(f"  Pair 2:")
        print(f"    Input:  start={window_stats['pair2_start_in']:.1f}f, ratio={window_stats['pair2_ratio_in']:.2f}x")
        print(f"    Target: start={window_stats['pair2_start_tgt']:.1f}f, ratio={window_stats['pair2_ratio_tgt']:.2f}x")
        print(f"    Tonality: {window_stats['pair2_tonality']:.2f}")
    
    # Component weights if present
    if 'input_rhythm_w' in metrics:
        print(f"\nCompositional Component Weights:")
        print(f"  Input:  Rhythm={metrics['input_rhythm_w']:.3f}, "
              f"Harmony={metrics['input_harmony_w']:.3f}")
        print(f"  Target: Rhythm={metrics['target_rhythm_w']:.3f}, "
              f"Harmony={metrics['target_harmony_w']:.3f}")
    
    print(f"{'='*80}\n")


class MetricsAccumulator:
    """Helper class to accumulate and average training metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.total_loss = 0.0
        self.total_rms_input = 0.0
        self.total_rms_target = 0.0
        self.total_spectral = 0.0
        self.total_mel = 0.0
        self.total_corr_penalty = 0.0
        self.total_novelty = 0.0
        self.total_balance_loss_raw = 0.0
        self.total_gan_loss = 0.0
        self.total_disc_loss = 0.0
        self.total_disc_real_acc = 0.0
        self.total_disc_fake_acc = 0.0
        self.total_output_input_corr = 0.0
        self.total_output_target_corr = 0.0
        
        # NEW: Ratio supervision metrics
        self.total_ratio_diversity = 0.0
        self.total_reconstruction = 0.0
        
        # Window selection stats
        self.total_pair0_start = 0.0
        self.total_pair1_start = 0.0
        self.total_pair2_start = 0.0
        self.total_pair0_ratio = 0.0
        self.total_pair1_ratio = 0.0
        self.total_pair2_ratio = 0.0
        self.total_pair0_tonality = 0.0
        self.total_pair1_tonality = 0.0
        self.total_pair2_tonality = 0.0
        
        # Compositional agent stats
        self.total_input_rhythm_w = 0.0
        self.total_input_harmony_w = 0.0
        self.total_target_rhythm_w = 0.0
        self.total_target_harmony_w = 0.0
        
        self.num_batches = 0
    
    def update(self, loss, rms_input, rms_target, spectral, mel, corr_penalty,
               novelty, balance_loss_raw=0.0, gan_loss=0.0, disc_loss=0.0,
               disc_real_acc=0.0, disc_fake_acc=0.0, output_input_corr=0.0,
               output_target_corr=0.0, ratio_diversity=0.0, reconstruction=0.0,
               metadata=None):
        """
        Update accumulated metrics with current batch values.
        
        Args:
            loss: Total loss value
            rms_input: RMS input loss
            rms_target: RMS target loss
            spectral: Spectral loss
            mel: Mel loss
            corr_penalty: Correlation penalty
            novelty: Novelty loss
            balance_loss_raw: Balance loss (optional)
            gan_loss: GAN generator loss (optional)
            disc_loss: Discriminator loss (optional)
            disc_real_acc: Discriminator real accuracy (optional)
            disc_fake_acc: Discriminator fake accuracy (optional)
            output_input_corr: Output-input correlation (optional)
            output_target_corr: Output-target correlation (optional)
            ratio_diversity: Ratio diversity loss (optional)
            reconstruction: Reconstruction loss (optional)
            metadata: Metadata dict with window/component stats (optional)
        """
        # Convert tensors to floats
        self.total_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
        self.total_rms_input += rms_input.item() if isinstance(rms_input, torch.Tensor) else rms_input
        self.total_rms_target += rms_target.item() if isinstance(rms_target, torch.Tensor) else rms_target
        self.total_spectral += spectral.item() if isinstance(spectral, torch.Tensor) else spectral
        self.total_mel += mel.item() if isinstance(mel, torch.Tensor) else mel
        self.total_corr_penalty += corr_penalty.item() if isinstance(corr_penalty, torch.Tensor) else corr_penalty
        self.total_novelty += novelty.item() if isinstance(novelty, torch.Tensor) else novelty
        self.total_balance_loss_raw += balance_loss_raw if isinstance(balance_loss_raw, float) else balance_loss_raw.item()
        
        self.total_gan_loss += gan_loss.item() if isinstance(gan_loss, torch.Tensor) else gan_loss
        self.total_disc_loss += disc_loss.item() if isinstance(disc_loss, torch.Tensor) else disc_loss
        self.total_disc_real_acc += disc_real_acc
        self.total_disc_fake_acc += disc_fake_acc
        
        self.total_output_input_corr += output_input_corr.item() if isinstance(output_input_corr, torch.Tensor) else output_input_corr
        self.total_output_target_corr += output_target_corr.item() if isinstance(output_target_corr, torch.Tensor) else output_target_corr
        
        # NEW: Ratio supervision metrics
        self.total_ratio_diversity += ratio_diversity.item() if isinstance(ratio_diversity, torch.Tensor) else ratio_diversity
        self.total_reconstruction += reconstruction.item() if isinstance(reconstruction, torch.Tensor) else reconstruction
        
        # Update window statistics from metadata
        if metadata and 'pairs' in metadata and len(metadata['pairs']) >= 3:
            # Track separate input and target statistics
            self.total_pair0_start += metadata['pairs'][0].get('start_input_mean', 0.0)
            self.total_pair1_start += metadata['pairs'][1].get('start_input_mean', 0.0)
            self.total_pair2_start += metadata['pairs'][2].get('start_input_mean', 0.0)
            self.total_pair0_start_tgt = getattr(self, 'total_pair0_start_tgt', 0.0) + metadata['pairs'][0].get('start_target_mean', 0.0)
            self.total_pair1_start_tgt = getattr(self, 'total_pair1_start_tgt', 0.0) + metadata['pairs'][1].get('start_target_mean', 0.0)
            self.total_pair2_start_tgt = getattr(self, 'total_pair2_start_tgt', 0.0) + metadata['pairs'][2].get('start_target_mean', 0.0)
            
            self.total_pair0_ratio += metadata['pairs'][0].get('ratio_input_mean', 0.0)
            self.total_pair1_ratio += metadata['pairs'][1].get('ratio_input_mean', 0.0)
            self.total_pair2_ratio += metadata['pairs'][2].get('ratio_input_mean', 0.0)
            self.total_pair0_ratio_tgt = getattr(self, 'total_pair0_ratio_tgt', 0.0) + metadata['pairs'][0].get('ratio_target_mean', 0.0)
            self.total_pair1_ratio_tgt = getattr(self, 'total_pair1_ratio_tgt', 0.0) + metadata['pairs'][1].get('ratio_target_mean', 0.0)
            self.total_pair2_ratio_tgt = getattr(self, 'total_pair2_ratio_tgt', 0.0) + metadata['pairs'][2].get('ratio_target_mean', 0.0)
            
            self.total_pair0_tonality += metadata['pairs'][0].get('tonality_strength_mean', 0.0)
            self.total_pair1_tonality += metadata['pairs'][1].get('tonality_strength_mean', 0.0)
            self.total_pair2_tonality += metadata['pairs'][2].get('tonality_strength_mean', 0.0)
        
        # Update compositional agent stats from metadata
        if metadata and 'compositional_stats' in metadata:
            stats = metadata['compositional_stats']
            self.total_input_rhythm_w += stats.get('input_rhythm_weight', 0.0)
            self.total_input_harmony_w += stats.get('input_harmony_weight', 0.0)
            self.total_target_rhythm_w += stats.get('target_rhythm_weight', 0.0)
            self.total_target_harmony_w += stats.get('target_harmony_weight', 0.0)
        
        self.num_batches += 1
    
    def get_averages(self):
        """
        Get averaged metrics over all accumulated batches.
        
        Returns:
            Dictionary with averaged metrics
        """
        if self.num_batches == 0:
            return {}
        
        n = self.num_batches
        
        return {
            'loss': self.total_loss / n,
            'rms_input': self.total_rms_input / n,
            'rms_target': self.total_rms_target / n,
            'spectral': self.total_spectral / n,
            'mel': self.total_mel / n,
            'corr_penalty': self.total_corr_penalty / n,
            'novelty': self.total_novelty / n,
            'balance_loss': self.total_balance_loss_raw / n,
            'gan_loss': self.total_gan_loss / n,
            'disc_loss': self.total_disc_loss / n,
            'disc_real_acc': self.total_disc_real_acc / n,
            'disc_fake_acc': self.total_disc_fake_acc / n,
            'output_input_corr': self.total_output_input_corr / n,
            'output_target_corr': self.total_output_target_corr / n,
            'ratio_diversity': self.total_ratio_diversity / n,
            'reconstruction': self.total_reconstruction / n,
            'disc_fake_acc': self.total_disc_fake_acc / n,
            'output_input_corr': self.total_output_input_corr / n,
            'output_target_corr': self.total_output_target_corr / n,
            'input_rhythm_w': self.total_input_rhythm_w / n,
            'input_harmony_w': self.total_input_harmony_w / n,
            'target_rhythm_w': self.total_target_rhythm_w / n,
            'target_harmony_w': self.total_target_harmony_w / n,
        }
    
    def get_window_stats(self):
        """
        Get averaged window selection statistics.
        
        Returns:
            Dictionary with window stats or None if no batches
        """
        if self.num_batches == 0:
            return None
        
        n = self.num_batches
        
        return {
            'pair0_start_in': self.total_pair0_start / n,
            'pair1_start_in': self.total_pair1_start / n,
            'pair2_start_in': self.total_pair2_start / n,
            'pair0_start_tgt': getattr(self, 'total_pair0_start_tgt', 0.0) / n,
            'pair1_start_tgt': getattr(self, 'total_pair1_start_tgt', 0.0) / n,
            'pair2_start_tgt': getattr(self, 'total_pair2_start_tgt', 0.0) / n,
            'pair0_ratio_in': self.total_pair0_ratio / n,
            'pair1_ratio_in': self.total_pair1_ratio / n,
            'pair2_ratio_in': self.total_pair2_ratio / n,
            'pair0_ratio_tgt': getattr(self, 'total_pair0_ratio_tgt', 0.0) / n,
            'pair1_ratio_tgt': getattr(self, 'total_pair1_ratio_tgt', 0.0) / n,
            'pair2_ratio_tgt': getattr(self, 'total_pair2_ratio_tgt', 0.0) / n,
            'pair0_tonality': self.total_pair0_tonality / n,
            'pair1_tonality': self.total_pair1_tonality / n,
            'pair2_tonality': self.total_pair2_tonality / n,
        }
