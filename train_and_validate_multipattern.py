import torch
from tqdm import tqdm
from utilities import save_checkpoint, save_sample_as_numpy
from masking_utils import generate_batch_masks, apply_mask, create_fixed_validation_masks
from fusion_loss import multi_pattern_loss, reconstruction_only_loss


def freeze_parameters(module):
    """Freeze all parameters in a module."""
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_parameters(module):
    """Unfreeze all parameters in a module."""
    for param in module.parameters():
        param.requires_grad = True


def get_actual_model(model):
    """Get the actual model, unwrapping DDP if necessary."""
    return model.module if hasattr(model, 'module') else model


def train_and_validate_multipattern(model,
                                    train_loader,
                                    val_loader,
                                    start_epoch,
                                    epochs,
                                    criterion,
                                    optimizer,
                                    device,
                                    sample_rate,
                                    checkpoint_folder,
                                    music_out_folder,
                                    phase1_end=10,
                                    phase2_end=20,
                                    accumulation_steps=4,
                                    use_scheduler=True,
                                    num_patterns=3,
                                    rank=0,
                                    train_sampler=None):
    """
    Three-phase training for multi-pattern fusion model.
    
    Phase 1 (epochs 1-10): Train encoder-decoder without masking
        - Forward: 3 patterns → encode → decode → 3 reconstructions
        - Loss: reconstruction_only_loss (no masking, no transformer)
        - Frozen: transformer, fusion_layer
        
    Phase 2 (epochs 11-20): Train encoder-decoder with random masking
        - Forward: 3 masked patterns → encode → decode → 3 reconstructions
        - Loss: reconstruction_only_loss (random masks)
        - Frozen: transformer, fusion_layer
        
    Phase 3 (epochs 21+): Full model training with transformer fusion
        - Forward: 3 masked patterns → encode → transformer fusion → 1 fused output
        - Loss: multi_pattern_loss (chunk-wise MSE with min selection)
        - Frozen: none (all components trainable)
    
    Args:
        model: MultiPatternAttentionModel
        train_loader: DataLoader with triplet format
        val_loader: DataLoader with triplet format
        start_epoch: Starting epoch
        epochs: Total epochs
        criterion: Base loss function (MSE)
        optimizer: Optimizer
        device: cuda/cpu
        sample_rate: Audio sample rate (22050)
        checkpoint_folder: Checkpoint save path
        music_out_folder: Output music save path
        phase1_end: Last epoch of phase 1 (default 10)
        phase2_end: Last epoch of phase 2 (default 20)
        accumulation_steps: Gradient accumulation steps
        use_scheduler: Use learning rate scheduler
        num_patterns: Number of patterns per triplet (3)
    """
    model = model.to(device)
    
    # Initialize learning rate scheduler
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    
    # Generate fixed validation masks (reused every epoch)
    # Get seq_len from first batch
    first_batch = next(iter(val_loader))[0]
    val_seq_len = first_batch.shape[-1]
    val_masks = create_fixed_validation_masks(
        num_patterns=num_patterns,
        seq_len=val_seq_len,
        sample_rate=sample_rate
    )
    val_masks = val_masks.to(device)
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, epochs + 1):
        # Set epoch for distributed sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # Get actual model (unwrap DDP if necessary)
        actual_model = get_actual_model(model)
        
        # Determine current phase
        if epoch <= phase1_end:
            phase = 1
            phase_name = "Phase 1: Unmasked Encoder-Decoder"
            freeze_parameters(actual_model.transformer)
            freeze_parameters(actual_model.fusion_layer)
            unfreeze_parameters(actual_model.encoder_decoder)
        elif epoch <= phase2_end:
            phase = 2
            phase_name = "Phase 2: Masked Encoder-Decoder"
            freeze_parameters(actual_model.transformer)
            freeze_parameters(actual_model.fusion_layer)
            unfreeze_parameters(actual_model.encoder_decoder)
        else:
            phase = 3
            phase_name = "Phase 3: Full Model + Transformer Fusion"
            unfreeze_parameters(actual_model.transformer)
            unfreeze_parameters(actual_model.fusion_layer)
            unfreeze_parameters(actual_model.encoder_decoder)
        
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{epochs} - {phase_name}")
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
            # Print GPU memory usage
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.max_memory_allocated(device) / 1024**3
                print(f"GPU memory allocated: {gpu_mem:.2f} GB")
            print(f"{'='*60}")
        
        # Training
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
            # inputs: [batch, 3, 2, 352800]
            # targets: [batch, 3, 2, 352800]
            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_size = inputs.shape[0]
            
            # Generate masks based on phase
            if phase == 1:
                # Phase 1: No masking
                masks = torch.ones(batch_size, num_patterns, inputs.shape[-1], dtype=torch.bool, device=device)
                use_transformer = False
            else:
                # Phase 2 & 3: Random masking
                masks = generate_batch_masks(
                    batch_size=batch_size,
                    num_patterns=num_patterns,
                    seq_len=inputs.shape[-1],
                    sample_rate=sample_rate
                ).to(device)
                use_transformer = (phase == 3)
            
            # Apply masking to inputs
            masked_inputs = apply_mask(inputs, masks)
            
            # Forward pass
            if use_transformer:
                # Phase 3: Full forward with transformer fusion
                reconstructed, fused_output = model(masked_inputs, masks)
                
                # Compute multi-pattern loss (chunk-wise with min selection)
                loss, rec_loss, pred_loss = multi_pattern_loss(
                    reconstructed=reconstructed,
                    inputs=inputs,
                    output=fused_output,
                    targets=targets,
                    masks=masks,
                    criterion=criterion
                )
            else:
                # Phase 1 & 2: Only reconstruction loss
                reconstructed, _ = model(masked_inputs, masks)
                
                loss = reconstruction_only_loss(
                    reconstructed=reconstructed,
                    inputs=inputs,
                    criterion=criterion
                )
            
            # Scale loss for accumulation
            loss = loss / accumulation_steps
            loss.backward()
            
            # Update weights after accumulation
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * accumulation_steps
        
        avg_train_loss = train_loss / len(train_loader)
        if rank == 0:
            print(f"Training Loss: {avg_train_loss:.6f}")
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch}")):
                inputs = inputs.to(device)
                targets = targets.to(device)
                batch_size = inputs.shape[0]
                
                # Expand fixed validation masks to batch size
                # val_masks: [num_patterns, seq_len] -> [batch_size, num_patterns, seq_len]
                current_val_masks = val_masks.unsqueeze(0).expand(batch_size, -1, -1)
                
                # Apply masking
                masked_inputs = apply_mask(inputs, current_val_masks)
                
                # Forward pass
                if phase == 3:
                    reconstructed, fused_output = model(masked_inputs, current_val_masks)
                    loss, rec_loss, pred_loss = multi_pattern_loss(
                        reconstructed=reconstructed,
                        inputs=inputs,
                        output=fused_output,
                        targets=targets,
                        masks=current_val_masks,
                        criterion=criterion
                    )
                else:
                    reconstructed, _ = model(masked_inputs, current_val_masks)
                    loss = reconstruction_only_loss(
                        reconstructed=reconstructed,
                        inputs=inputs,
                        criterion=criterion
                    )
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        if rank == 0:
            print(f"Validation Loss: {avg_val_loss:.6f}")
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step(avg_val_loss)
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0 and rank == 0:
            save_checkpoint(model, optimizer, epoch, checkpoint_folder)
            # Note: save_sample_as_numpy may need adaptation for multi-pattern
            try:
                save_sample_as_numpy(model, val_loader, device, music_out_folder, epoch)
            except Exception as e:
                print(f"Warning: Could not save sample: {e}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if rank == 0:
                save_checkpoint(model, optimizer, epoch, checkpoint_folder, filename='model_best.pt')
                print(f"★ New best model saved! Validation loss: {best_val_loss:.6f}")
        
        if rank == 0:
            print(f"{'='*60}\n")
        
        # Clear GPU cache after each epoch to prevent OOM
        torch.cuda.empty_cache()
