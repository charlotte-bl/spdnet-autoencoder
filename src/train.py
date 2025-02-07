from spdnet.optimizers import RiemannianAdam
import torch
from tqdm import tqdm
from data_preprocessing import is_data_with_noise

def train(train_loader,val_loader,model,n_epochs,criterion):
    optimizer = RiemannianAdam(model.parameters(), lr=0.001)  #optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    list_train_loss = []
    list_val_loss = []
    noised = is_data_with_noise(train_loader)
    for epoch in range(n_epochs):
        #initialization
        batch_train_loss = 0.0
        batch_val_loss = 0.0
        #train step
        model.train()
        for data in train_loader:
            if noised:
                noisy_train, data_train, _ = data
                outputs_train = model(noisy_train)
            else:
                data_train,_ = data
                outputs_train = model(data_train)
            data_train_loss = criterion(outputs_train, data_train)
            batch_train_loss += data_train_loss.item()/data_train.size(0)
            data_train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        #validation step
        model.eval()
        for data in val_loader:
            with torch.no_grad():
                if noised:
                    noisy_val, data_val, _ = data
                    outputs_val = model(noisy_val)
                else :
                    data_val,_ = data
                    outputs_val = model(data_val)
                data_val_loss = criterion(outputs_val, data_val)
                batch_val_loss += data_val_loss.item()/data_val.size(0)

        #loss
        epoch_train_loss = batch_train_loss/len(train_loader)
        epoch_val_loss = batch_val_loss/len(val_loader)
        delta = batch_val_loss*0.05

        #print losses
        print(f"Epoch : {epoch}")
        print(f"| Perte train moyenne du batch: {epoch_train_loss} ")
        print(f"| Perte val : {epoch_val_loss} ")

        list_train_loss.append(epoch_train_loss)
        list_val_loss.append(epoch_val_loss)

        #early stopping
        if epoch_val_loss>min(list_val_loss)+delta:
                break;
    
    result = [data_train,outputs_train,list_train_loss,data_val,outputs_val,list_val_loss]
    if noised:
        result.append(noisy_train) #a la fin
        result.append(noisy_val)
    return result