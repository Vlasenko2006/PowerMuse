import torch
from tqdm import tqdm
from utilities import save_checkpoint, save_sample_as_numpy


# Helper function to freeze parameters
def freeze_parameters(module):
    for param in module.parameters():
        param.requires_grad = False


# Training and validation with improved features
def train_and_validate(model,
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
                       FREEZE_ENCODER_DECODER_AFTER=10,
                       accumulation_steps=4,
                       use_scheduler=True):
    """
    Train and validate the model with gradient accumulation and learning rate scheduling.

    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        start_epoch: Starting epoch number
        epochs: Total number of epochs
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use (cuda/cpu)
        sample_rate: Audio sample rate
        checkpoint_folder: Folder to save checkpoints
        music_out_folder: Folder to save music outputs
        FREEZE_ENCODER_DECODER_AFTER: Epoch after which to add transformer training
        accumulation_steps: Number of steps to accumulate gradients
        use_scheduler: Whether to use learning rate scheduler
    """
    # Move the model to the correct device
    model = model.to(device)
    
    # Initialize learning rate scheduler
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            verbose=True,
            min_lr=1e-6
        )
    
    # Track best validation loss for model saving
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Training
        model.train()
        train_loss = 0
        train_rec_loss = 0
        train_task_loss = 0
        optimizer.zero_grad()  # Initialize gradients before accumulation

        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc="Training")):
            # Move inputs and targets to the correct device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # No normalization shift - data already normalized to [-1, 1]
            
            # Forward pass
            reconstructed, outputs = model(inputs)
            rec_loss = criterion(inputs, reconstructed)

            # Add task-specific loss if applicable
            if epoch >= FREEZE_ENCODER_DECODER_AFTER:
                task_loss = criterion(outputs, targets)
                # Weighted loss: give more weight to prediction task
                total_loss = 0.3 * rec_loss + 0.7 * task_loss
            else:
                task_loss = torch.tensor(0.0)
                total_loss = rec_loss

            # Scale loss for accumulation
            total_loss = total_loss / accumulation_steps

            # Backward pass
            total_loss.backward()

            # Update weights after accumulating gradients
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()  # Reset gradients after the step

            # Accumulate the training loss
            train_loss += total_loss.item() * accumulation_steps
            train_rec_loss += rec_loss.item()
            if isinstance(task_loss, torch.Tensor):
                train_task_loss += task_loss.item()

        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        avg_train_rec_loss = train_rec_loss / len(train_loader)
        avg_train_task_loss = train_task_loss / len(train_loader) if train_task_loss > 0 else 0
        
        print(
            f"Training - Total Loss: {avg_train_loss:.6f}, "
            f"Reconstruction: {avg_train_rec_loss:.6f}, "
            f"Prediction: {avg_train_task_loss:.6f}"
        )

        # Validation
        model.eval()
        val_loss = 0
        val_rec_loss = 0
        val_task_loss = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation"):
                # Move inputs and targets to the correct device
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                reconstructed, outputs = model(inputs)
                rec_loss = criterion(inputs, reconstructed)

                # Add task-specific loss if applicable
                if epoch >= FREEZE_ENCODER_DECODER_AFTER:
                    task_loss = criterion(outputs, targets)
                    total_loss = 0.3 * rec_loss + 0.7 * task_loss
                else:
                    task_loss = torch.tensor(0.0)
                    total_loss = rec_loss

                # Accumulate the validation loss
                val_loss += total_loss.item()
                val_rec_loss += rec_loss.item()
                if isinstance(task_loss, torch.Tensor):
                    val_task_loss += task_loss.item()

        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        avg_val_rec_loss = val_rec_loss / len(val_loader)
        avg_val_task_loss = val_task_loss / len(val_loader) if val_task_loss > 0 else 0
        
        print(
            f"Validation - Total Loss: {avg_val_loss:.6f}, "
            f"Reconstruction: {avg_val_rec_loss:.6f}, "
            f"Prediction: {avg_val_task_loss:.6f}"
        )

        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step(avg_val_loss)

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, epoch, checkpoint_folder)
            save_sample_as_numpy(model, val_loader, device, music_out_folder, epoch)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(model, optimizer, epoch, checkpoint_folder, filename="model_best.pt")
            print(f"★ New best model saved! Validation loss: {best_val_loss:.6f}")
