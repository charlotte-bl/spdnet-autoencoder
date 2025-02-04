import torch
from metrics import trustworthiness,pairwise_euclidean_distances
from visualization import show_latent_dim_2

def test(test_loader,model,criterion,noise="none",show=False):
    class_right = []
    class_left = []
    batch_test_loss = 0.0
    trustworthiness_recomp = 0.0
    trustworthiness_latent = 0.0
    model.eval()
    #for data_test,labels in tqdm(test_loader):
    if noise!="none":
        for noisy_test, data_test,labels in test_loader:
            with torch.no_grad():                 
                z = model.encoder(noisy_test)
                #ajout point dans liste pour affichage
                for i in range(z.shape[0]):
                    if labels[i]=='right_hand':
                        class_right.append(z[i].numpy())
                    else:
                        class_left.append(z[i].numpy())
                outputs_test = model.decoder(z)
                data_test_loss = criterion(outputs_test, data_test)
                batch_test_loss += data_test_loss.item()/data_test.size(0)
                if isinstance(criterion, torch.nn.MSELoss):
                    trustworthiness_recomp += trustworthiness(data_test,outputs_test,pairwise_distance=pairwise_euclidean_distances)
                    trustworthiness_latent += trustworthiness(data_test,z,pairwise_distance=pairwise_euclidean_distances)
                else:
                    trustworthiness_recomp += trustworthiness(data_test,outputs_test)
                    trustworthiness_latent += trustworthiness(data_test,z)
                
    else:
        for data_test,labels in test_loader:
            with torch.no_grad():  
                z = model.encoder(data_test)
                #ajout point dans liste pour affichage
                for i in range(z.shape[0]):
                    if labels[i]=='right_hand':
                        class_right.append(z[i].numpy())
                    else:
                        class_left.append(z[i].numpy())
                outputs_test = model.decoder(z)
                data_test_loss = criterion(outputs_test, data_test)
                batch_test_loss += data_test_loss.item()/data_test.size(0)
                if isinstance(criterion, torch.nn.MSELoss):
                    trustworthiness_recomp += trustworthiness(data_test,outputs_test,pairwise_distance=pairwise_euclidean_distances)
                    trustworthiness_latent += trustworthiness(data_test,z,pairwise_distance=pairwise_euclidean_distances)
                else:
                    trustworthiness_recomp += trustworthiness(data_test,outputs_test)
                    trustworthiness_latent += trustworthiness(data_test,z)
            
    test_loss = batch_test_loss/len(test_loader)
    trustworthiness_recomp = trustworthiness_recomp/len(test_loader)
    trustworthiness_latent = trustworthiness_latent/len(test_loader)
    print("Test : ")
    print(f"| Perte : {test_loss}")
    print(f"| Trustworthiness origine/décodé : {trustworthiness_recomp}")
    print(f"| Trustworthiness origine/encodé : {trustworthiness_latent}")

    #affichage si matrice 2x2
    if z.shape[2]==2:
        show_latent_dim_2(class_right,class_left,show)
        

    return data_test,outputs_test,test_loss