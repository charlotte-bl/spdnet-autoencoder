import torch
from metrics import trustworthiness,pairwise_euclidean_distances
from visualization import show_encoding_dim_2
from data_preprocessing import is_data_with_noise

def test(test_loader,model,criterion,show=False,class_1_name=''):
    class_1 = []
    class_2 = []
    batch_test_loss = 0.0
    trustworthiness_recomp = 0.0
    trustworthiness_encoding = 0.0
    model.eval()
    noised = is_data_with_noise(test_loader)
    is_one_channel = (model.ho == 1)
    #for data_test,labels in tqdm(test_loader):
    for data in test_loader:
        with torch.no_grad():   
            if noised:
                noisy_test, data_test, labels = data
                outputs_test = model(noisy_test)
                z = model.encoder(noisy_test)
            else:
                data_test, labels = data
                outputs_test = model(data_test)
                z = model.encoder(data_test)
            #ajout point dans liste pour affichage
            for i in range(z.shape[0]):
                if labels[i]==class_1_name:
                    class_1.append(z[i].numpy())
                else:
                    class_2_name=labels[i]
                    class_2.append(z[i].numpy())
            outputs_test = model.decoder(z)
            data_test_loss = criterion(outputs_test, data_test)
            batch_test_loss += data_test_loss.item()/data_test.size(0)
            #trustworthiness metrics on batches
            if isinstance(criterion, torch.nn.MSELoss):
                trustworthiness_recomp += trustworthiness(data_test,outputs_test,pairwise_distance=pairwise_euclidean_distances)
                if is_one_channel: #on peut pas si ho diff hi car on compare ho channels à hi
                    trustworthiness_encoding += trustworthiness(data_test,z,pairwise_distance=pairwise_euclidean_distances)
            else:
                trustworthiness_recomp += trustworthiness(data_test,outputs_test)
                #if is_one_channel:
                #    trustworthiness_encoding += trustworthiness(data_test,z)
            
    test_loss = batch_test_loss/len(test_loader)
    trustworthiness_recomp = trustworthiness_recomp/len(test_loader)
    trustworthiness_encoding = trustworthiness_encoding/len(test_loader)
    print("Test : ")
    print(f"| Perte : {test_loss}")
    print(f"| Trustworthiness origine/décodé : {trustworthiness_recomp}")
    #if is_one_channel:
    #    print(f"| Trustworthiness origine/encodé : {trustworthiness_encoding}")

    #affichage si matrice 2x2
    if z.shape[2]==2:
        show_encoding_dim_2(class_1,class_2,show,class_1_name,class_2_name)
        
    result = [data_test, outputs_test, test_loss, trustworthiness_recomp]
    if noised:
        result.append(noisy_test) #a la fin
    if model.ho == 1:
        result.append(trustworthiness_encoding) #a la fin
    return result