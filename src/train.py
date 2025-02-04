from spdnet.optimizers import RiemannianAdam
import torch
from tqdm import tqdm

def train(train_loader,val_loader,model,n_epochs,criterion,noise="none"):
    optimizer = RiemannianAdam(model.parameters(), lr=0.001)  #optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    list_train_loss = []
    list_val_loss = []
    for epoch in range(n_epochs):
        #initialization
        batch_train_loss = 0.0
        batch_val_loss = 0.0

        #train step
        model.train()
        if noise!="none":
            for noisy_train, data_train,_ in train_loader: #for data_train,_ in tqdm(train_loader):
                outputs_train = model(noisy_train)
                data_train_loss = criterion(outputs_train, data_train)
                batch_train_loss += data_train_loss.item()/data_train.size(0)
                data_train_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        else:
            for data_train,_ in train_loader: #for data_train,_ in tqdm(train_loader):
                outputs_train = model(data_train)
                data_train_loss = criterion(outputs_train, data_train)
                batch_train_loss += data_train_loss.item()/data_train.size(0)
                data_train_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        #validation step
        model.eval()
        if noise!="none":
             for noisy_val, data_val ,_ in val_loader:
                with torch.no_grad():
                        outputs_val = model(noisy_val)
                        data_val_loss = criterion(outputs_val, data_val)
                        batch_val_loss += data_val_loss.item()/data_val.size(0)
        else:
            for data_val,_ in val_loader:
                with torch.no_grad():
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
        #print(f"| Perte train dernier élément du batch : {data_train_loss.item()/data_train.size(0)} ")
        print(f"| Perte val : {epoch_val_loss} ")

        list_train_loss.append(epoch_train_loss)
        list_val_loss.append(epoch_val_loss)

        #early stopping
        if epoch_val_loss>min(list_val_loss)+delta:
                break;

    return data_train,outputs_train,list_train_loss,data_val,outputs_val,list_val_loss