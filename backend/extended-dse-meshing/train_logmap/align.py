import torch

def align_pca(X):
    n_pc_points = X.size(1)
    C = torch.matmul(X.transpose(1, 2), X)  # C = X^T @ X
    s_v, u_v, v_v = torch.svd(C)
    v_v = torch.einsum('aij->aji', v_v)  # Rearrange dimensions
    R_opt = torch.einsum('aij,ajk->aik', u_v, v_v)
    concat_R_opt = R_opt.unsqueeze(1).repeat(1, n_pc_points, 1, 1)
    opt_labels = torch.einsum('abki,abi->abk', concat_R_opt, X)
    return opt_labels


def align(X, Y):
    X=X.permute(0,2,1)
    n_pc_points = X.size(1)
    centered_y = Y.unsqueeze(2)  # Y.shape: (B, N, 3) -> (B, N, 1, 3)
    centered_x = X.unsqueeze(2)  # X.shape: (B, N, 3) -> (B, N, 1, 3)
    # Transpose centered_y
    centered_y = torch.einsum('ijkl->ijlk', centered_y) # (B, N, 1, 3) -> (B, N, 3, 1)
    mult_xy = torch.einsum('abij,abjk->abik', centered_y, centered_x)
    C = torch.einsum('abij->aij', mult_xy)
    u,s, v = torch.svd(C)
    v = torch.einsum("aij->aji", v)
    R_opt = torch.einsum("aij,ajk->aik", u, v)
    concat_R_opt = R_opt.unsqueeze(1).repeat(1, n_pc_points, 1, 1)
    opt_labels = torch.einsum('abki,abi->abk', concat_R_opt, X)
    return opt_labels
