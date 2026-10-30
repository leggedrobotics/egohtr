import torch

# inpsiration from WHAM: https://github.com/yohanshin/WHAM/blob/2b54f7797391c94876848b905ed875b154c4a295/lib/eval/eval_utils.py#L92-L250
def batch_compute_similarity_transform_torch(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0,2,1)
        S2 = S2.permute(0,2,1)
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0,2,1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0],1,1)
    Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0,2,1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0,2,1)

    return S1_hat


def compute_similarity_transform_torch(S1, S2, frames=None, fixed_scale=False):
    '''
    Computes a single global similarity transform (sR, t) based on the first `frames`
    and applies it to the entire timeseries S1.
    
    Expected input shape for S1 and S2: (T, J, 3) 
    where T = Time/Frames, J = Joints, 3 = Spatial Dimensions
    '''
    # Ensure we are working with matching 3D tensors
    assert S1.dim() == 3 and S2.dim() == 3, "S1 and S2 must be 3D tensors of shape (Time, Joints, 3)"
    assert S1.shape == S2.shape, "S1 and S2 must have the same shape"
    
    T, J, C = S1.shape
    frames = frames if frames is not None else T
    assert frames <= T, "Number of frames for computing transform cannot exceed total frames in S"
    
    # 1. Extract the subset for computing the transform and flatten it.
    # S1[:frames] shape is (frames, J, C) -> reshape to (frames*J, C) -> transpose to (C, frames*J)
    S1_sub = S1[:frames].reshape(-1, C).T
    S2_sub = S2[:frames].reshape(-1, C).T

    # 2. Compute means ONLY on the subset
    mu1 = S1_sub.mean(axis=1, keepdims=True)
    mu2 = S2_sub.mean(axis=1, keepdims=True)
    
    X1 = S1_sub - mu1
    X2 = S2_sub - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1 ** 2)

    # 3. The outer product of X1 and X2.
    K = X1.mm(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)
    # V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[0], device=S1.device)
    Z[-1, -1] *= torch.sign(torch.det(U @ V.T))
    # Construct R.
    R = V.mm(Z.mm(U.T))

    # 5. Recover scale.
    if fixed_scale:
        scale = torch.ones(1, device=S1.device)  # No scaling, just use the original scale of S1)
    else:
        scale = torch.trace(R.mm(K)) / var1
        
    # 6. Recover translation.
    t = mu2 - scale * (R.mm(mu1))

    # 6. Apply the transformation to the ENTIRE sequence S1
    # Flatten whole S1 to (C, T*J)
    S1_full = S1.reshape(-1, C).T
    
    # Apply standard linear equation: s * R * S1 + t
    S1_hat_full = scale * R.mm(S1_full) + t
    
    # Reshape back to the original timeseries format (T, J, C)
    S1_hat = S1_hat_full.T.reshape(T, J, C)

    return S1_hat

    


    