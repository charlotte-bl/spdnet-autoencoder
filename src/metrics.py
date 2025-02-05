from spdnet.functional import dist_riemann_matrix
import torch

def pairwise_riemannian_distances(batch):
    batch_size,_,n,_ = batch.shape
    batch = batch.squeeze(1)
    dist_matrix = torch.zeros(batch_size,batch_size,device=batch.device)
    for i in range(batch_size):
        for j in range(batch_size):
            dist_matrix[i,j] = dist_riemann_matrix(batch[i],batch[j]) #distances entre matrice i du batch vs matrice j du batch
    return dist_matrix

def pairwise_euclidean_distances(batch):
    batch_size,_,n,_ = batch.shape
    batch = batch.squeeze(1)
    dist_matrix = torch.zeros(batch_size,batch_size,device=batch.device)
    for i in range(batch_size):
        for j in range(batch_size):
            dist_matrix[i,j] = torch.norm(batch[i] - batch[j],p='fro') #distances entre matrice i du batch vs matrice j du batch
    return dist_matrix

def diag_inf(dist_matrix):
    dist_matrix.fill_diagonal_(float('inf'))
    return dist_matrix

def trustworthiness(original,reconstructed,k=2,pairwise_distance=pairwise_riemannian_distances):
    batch_size=original.shape[0]
    orig_dist = diag_inf(pairwise_distance(original)) #diag inf pour qu'une matrice n'ait pas elle meme comme plus proche voisin
    recon_dist = diag_inf(pairwise_distance(reconstructed))

    orig_ranks = torch.argsort(orig_dist, dim=-1)
    recon_ranks = torch.argsort(recon_dist, dim=-1) #r(i,j)
    #pour chaque matrice, on trie les k plus proches voisins
    
    differences = torch.zeros(batch_size, dtype=torch.float32)

    # premier signe somme
    for i in range(batch_size):
        # deuxieme signe somme : iterations des voisins
        for j in range(batch_size):
            if recon_ranks[i, j] < k: # pour que ce soit sur l'ensemble des k nearest neighbours dans l'espace output
                orig_rank = orig_ranks[i, j]
                recon_rank = recon_ranks[i, j]
                differences[i] += max(orig_rank - recon_rank - k, 0)


    numerator = 2 * differences.sum()
    denominator = batch_size * k * (2 * batch_size - 3 * k - 1)
    trustworthiness_score = 1 - (numerator / denominator)

    return trustworthiness_score.mean()
