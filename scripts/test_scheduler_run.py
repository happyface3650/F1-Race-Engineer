from torch.utils.data import random_split, DataLoader
import torch
import dataloader as dl
import training

# Create small train/val split from existing dataset
full_dataset = training.dataset
n = len(full_dataset)
train_len = int(n * 0.9)
val_len = n - train_len
train_ds, val_ds = random_split(full_dataset, [train_len, val_len])

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=dl.collate_fn)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=dl.collate_fn)

# Use a smaller patience and fewer epochs to trigger LR reduction for the test
training.train_and_save(training.model, train_loader, training.optimizer, training.criterion, training.device, num_epochs=6, save_path='tmp_test_scheduler.pth', use_plateau=True, plateau_kwargs={'patience':1, 'factor':0.1}, val_dataloader=val_loader)
print('Test run complete')
