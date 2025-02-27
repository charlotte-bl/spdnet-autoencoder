from spdnet.optimizers import RiemannianAdam
import torch
from data_preprocessing import is_data_with_noise
from tqdm import tqdm


def train(train_loader, val_loader, model, n_epochs, criterion, lr=0.001):
    optimizer = RiemannianAdam(model.parameters(), lr=lr)
    list_train_loss = []
    list_val_loss = []
    noised = is_data_with_noise(train_loader)

    # Barre de progression sur les epochs
    epoch_bar = tqdm(range(n_epochs), desc="Training", position=0)

    for epoch in epoch_bar:
        batch_train_loss = 0.0
        batch_val_loss = 0.0

        # Train step
        model.train()
        for data in train_loader:
            if noised:
                noisy_train, data_train, _ = data
                outputs_train = model(noisy_train)
            else:
                data_train, _ = data
                outputs_train = model(data_train)

            data_train_loss = criterion(outputs_train, data_train)
            batch_train_loss += data_train_loss.item() / data_train.size(0)

            data_train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Validation step
        model.eval()
        with torch.no_grad():
            for data in val_loader:
                if noised:
                    noisy_val, data_val, _ = data
                    outputs_val = model(noisy_val)
                else:
                    data_val, _ = data
                    outputs_val = model(data_val)
                data_val_loss = criterion(outputs_val, data_val)
                batch_val_loss += data_val_loss.item() / data_val.size(0)

        # Calcul des pertes moyennes
        epoch_train_loss = batch_train_loss / len(train_loader)
        epoch_val_loss = batch_val_loss / len(val_loader)
        delta = batch_val_loss * 0.05

        # Mettre à jour la barre de progression avec la loss
        epoch_bar.set_postfix(
            {"Train Loss": epoch_train_loss, "Val Loss": epoch_val_loss}
        )

        # Stocker les pertes
        list_train_loss.append(epoch_train_loss)
        list_val_loss.append(epoch_val_loss)

        # Early stopping
        if epoch_val_loss > min(list_val_loss) + delta:
            break
    result = [
        data_train,
        outputs_train,
        list_train_loss,
        data_val,
        outputs_val,
        list_val_loss,
    ]
    if noised:
        result.append(noisy_train)  # a la fin
        result.append(noisy_val)
    return result
