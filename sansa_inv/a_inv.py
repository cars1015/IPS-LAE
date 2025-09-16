import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import sklearn.utils.sparsefuncs as spfuncs
import warnings
from .utils import (
    get_residual_matrix,
    substitute_columns,
    sparsify,
    sq_column_norms,
)


def s1(L: sp.csc_matrix) -> sp.csc_matrix:
    """Calculate approximate inverse of unit lower triangular matrix using 1 step of Schultz method."""
    M = L.copy()
    M.setdiag(M.diagonal() - 2)
    return -M


def umr(
    A: sp.csc_matrix, M_0: sp.csc_matrix, target_density: float, params: dict
) -> sp.csc_matrix:
    def _umr_scan(
        A: sp.csc_matrix,
        M: sp.csc_matrix,
        R: sp.csc_matrix,
        residuals: np.ndarray,
        n: int,
        target_density: float,
        ncols: int,
        nblocks: int,
        counter: int,
        log_norm_threshold: float,
    ) -> sp.csc_matrix:
        sncn = residuals / np.sqrt(R.shape[0])
        large_norm_indices = sncn > (10 ** max(-counter - 1, log_norm_threshold))

        # Iterate over blocks of columns
        for i in range(nblocks):
            left = i * ncols
            right = min((i + 1) * ncols, n)
            # Get indices of columns to be updated in this step
            col_indices = np.arange(left, right)
            # Only consider columns with sufficiently large norm
            col_indices = np.intersect1d(
                col_indices,
                np.where(large_norm_indices)[0],
                assume_unique=True,
            )  # this returns columns in sorted order
            if len(col_indices) == 0:
                # No columns to be updated in this step
                continue

            R_part = R[:, col_indices]
            M_part = M[:, col_indices]

            # Compute projection matrix
            P = A.dot(R_part)
            # scale columns of P by 1 / (norm of columns squared)
            with np.errstate(
                divide="ignore"
            ):  # can raise divide by zero warning when using intel MKL numpy. Most likely cause by endianness
                scale = 1 / sq_column_norms(P)
            spfuncs.inplace_column_scale(P, scale)

            # compute: alpha = diag(R_part^T @ P)
            alpha = np.asarray(R_part.multiply(P).sum(axis=0))[0]
            # garbage collection, since we don't need P anymore
            del P

            # scale columns of R_part by alpha
            spfuncs.inplace_column_scale(R_part, alpha)

            # compute update
            M_update = R_part + M_part

            # update M
            M = substitute_columns(M, col_indices, M_update)

            # Sparsify matrix M globally to target density
            M = sparsify(A=M, m=n, n=n, target_density=target_density)

        return M

    def _umr_finetune_step(
        A: sp.csc_matrix,
        M: sp.csc_matrix,
        R: sp.csc_matrix,
        residuals: np.ndarray,
        n: int,
        target_density: float,
        ncols: int,
    ) -> sp.csc_matrix:
        """Finetune M by updating the worst columns"""
        # Find columns with large length-normalized residuals (because L is lower triangular)
        # seems to converge faster than unnormalized residuals
        # for non-lower-triangular, no not normalize.
        residuals = residuals / np.sqrt(np.arange(1, n + 1)[::-1])

        # select ncols columns with largest residuals
        col_indices = np.argpartition(residuals, -ncols)[-ncols:]
        col_indices = np.sort(col_indices)

        R_part = R[:, col_indices]
        M_part = M[:, col_indices]

        # compute projection matrix
        P = A.dot(R_part)

        # scale columns of P by 1 / (norm of columns squared)
        with np.errstate(
            divide="ignore"
        ):  # can raise divide by zero warning when using intel MKL numpy. Most likely cause by endianness
            scale = 1 / sq_column_norms(P)
        spfuncs.inplace_column_scale(P, scale)

        # compute: alpha = diag(R^T @ P)
        alpha = np.asarray(R_part.multiply(P).sum(axis=0))[0]
        # garbage collection, since we don't need P anymore
        del P

        # scale columns of R by alpha
        spfuncs.inplace_column_scale(R_part, alpha)

        with warnings.catch_warnings():  # ignore warning about changing sparsity pattern
            warnings.simplefilter("ignore")
            M_update = R_part + M_part

        # update M
        M = substitute_columns(M, col_indices, M_update)

        # Sparsify matrix M globally to target density
        M = sparsify(A=M, m=n, n=n, target_density=target_density)

        return M

    # Initialize parameters
    n = A.shape[0]
    num_scans = params.get("umr_scans", 1)
    num_finetune_steps = params.get("umr_finetune_steps", 1)
    log_norm_threshold = params.get(
        "umr_log_norm_threshold", -7
    )  # 10**-7 is circa machine precision for float32
    loss_threshold = params.get("umr_loss_threshold", 1e-4)
    # corresponds to ||I - A @ M||_F / ||I||_F < 10^{-2}, i.e. 1% error

    # Compute number of columns to be updated in each iteration inside scan and finetune step
    # We want to utilize dense addition, so we choose ncols such that the dense matrix of size n x ncols
    # has the same number of values as the sparse matrix A of size n x n with target density.
    ncols = np.ceil(n * target_density).astype(int)
    nblocks = np.ceil(n / ncols).astype(int)

    # Initialize M
    M = M_0
    # Initialize arrays to log computation times

    # Perform given number of scans
    for i in range(1, num_scans + 1):
        # Compute residual matrix
        R = get_residual_matrix(A, M)
        # Compute column norms of R
        sq_norm = sq_column_norms(R)
        # Compute maximum residual and mean squared column norm for logging
        # n * MSE = mean (column norm)^2 = (relative Frobenius norm)^2
        residuals = np.sqrt(sq_norm)  # column norms
        max_res = np.max(residuals)
        loss = np.mean(sq_norm)
        # Stop if mean squared column norm is sufficiently small
        if loss < loss_threshold:
            print("Reached stopping criterion.")
            return M
        # Perform UMR scan
        print(f"Performing UMR scan {i}...")
        M= _umr_scan(
            A=A,
            M=M,
            R=R,
            residuals=residuals,
            n=n,
            target_density=target_density,
            ncols=ncols,
            nblocks=nblocks,
            counter=i,
            log_norm_threshold=log_norm_threshold,
        )
        
    # Perform given number of finetune steps
    for i in range(1, num_finetune_steps + 1):
        # Compute residual matrix
        R = get_residual_matrix(A, M)
        # Log its density for debugging - it will be denser than A and M, but should be manageable
        # In case R is too dense, we may adjust this algorithm to compute R in chunks and proceed with the following steps
        # on each chunk separately.
        print(f"Density of residual matrix: {R.nnz / (n**2):.6%}")

        # Compute column norms of R
        sq_norm = sq_column_norms(R)
        # Compute maximum residual and mean squared column norm for logging
        # Loss = n * MSE = mean (column norm)^2 = (relative Frobenius norm)^2
        residuals = np.sqrt(sq_norm)  # column norms
        max_res = np.max(residuals)
        loss = np.mean(sq_norm)
        # Stop if mean column norm is sufficiently small
        if loss < loss_threshold:
            print("Reached stopping criterion.")
            return M

        # Perform finetune step
        print(f"Performing UMR finetune step {i}...")
        M= _umr_finetune_step(
            A=A,
            M=M,
            R=R,
            residuals=residuals,
            n=n,
            target_density=target_density,
            ncols=ncols,
        )
    return M


def ainv_L(L,
    target_density,
    method_params: dict,
) -> tuple[sp.csr_matrix, list[float]]:
    """Calculate approximate inverse of L using UMR method."""
    print("Calculate approximate inverse of L using UMR method.")
    M_0 = s1(L)  # initial guess
    ainv = umr(
        A=L,
        M_0=M_0,
        target_density=target_density,
        params=method_params,
        )
    # Final residual norm evaluation
    # Compute residual matrix
    R = get_residual_matrix(L, ainv)
    # Compute column norms of R
    sq_norm = sq_column_norms(R)
    # Compute maximum residual and mean squared column norm for logging
    # Loss = n * MSE = mean (column norm)^2 = (relative Frobenius norm)^2
    residuals = np.sqrt(sq_norm)  # column norms
    max_res = np.max(residuals)
    loss = np.mean(sq_norm)
    print(
            f"Current maximum residual: {max_res:.8f}, relative Frobenius norm squared: {loss:.8f}"
        )
    return ainv.tocsr()