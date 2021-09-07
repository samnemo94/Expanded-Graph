from typing import Tuple, List
import numpy as np

def cpu_tsvd(
        A: np.array,
        k: int,
        n_iter: int = 2,
        n_oversamples: int = 8
) -> Tuple[np.array, np.array, np.array]:
    """
    GPU Truncated SVD. Based on fbpca's version.

    Parameters
    ----------
    A : (M, N) torch.Tensor
    k : int
    n_iter : int
    n_oversamples : int

    Returns
    -------
    u : (M, k) torch.Tensor
    s : (k,) torch.Tensor
    vt : (k, N) torch.Tensor
    """
    m, n = A.shape
    Q = np.random.rand(n, k + n_oversamples)
    Q = A @ Q

    Q, _ = np.linalg.qr(Q)

    # Power iterations
    for _ in range(n_iter):
        Q = (Q.transpose() @ A).transpose()
        Q, _ = np.linalg.qr(Q)
        Q = A @ Q
        Q, _ = np.linalg.qr(Q)

    QA = Q.transpose() @ A
    # Transpose QA to make it tall-skinny as MAGMA has optimisations for this
    # (USVt)t = VStUt
    Va, s, R = np.linalg.svd(QA.transpose())
    U = Q @ R.transpose()

    return U[:, :k], s[:k], Va.transpose()[:k, :]


