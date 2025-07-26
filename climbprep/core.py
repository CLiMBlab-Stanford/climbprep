import time
import numpy as np
from scipy import sparse
from nilearn import image
import pickle



def get_sparse_adjacency(faces, adjacency_depth=None):
    """
    Get a sparse adjacency matrix from the faces of a mesh.
    :param faces: Faces of the mesh as a numpy array of shape (n_faces, 3).
    :param adjacency_depth: Depth of adjacency to compute (1 for direct neighbors, 2 for neighbors of neighbors, etc.).
    :return: Sparse adjacency matrix as a scipy CSR matrix.
    """
    n_vertices = np.max(faces) + 1
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    data = np.ones_like(rows, dtype=np.uint8)
    adjacency_matrix = sparse.coo_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices)) \
        .astype(bool).tocsr()

    adjacency_matrix_ = adjacency_matrix
    for i in range(adjacency_depth - 1):
        adjacency_matrix_ += adjacency_matrix_ @ adjacency_matrix

    return adjacency_matrix_


def get_sparse_squared_diff(x, adjacency_matrix):
    n = len(x)
    X = sparse.coo_matrix((x, (np.arange(n), np.arange(n))), shape=(n, n))
    X = X.tocsr()
    D = X @ adjacency_matrix - adjacency_matrix @ X
    D2 = D.multiply(D)

    return D2


def get_geodesic_smoothing_weights(faces, coordinates=None, fwhm=None, adjacency_depth=None):
    if not fwhm:
        return sparse.eye(len(coordinates))

    if not adjacency_depth:
        # Infer adjacency depth from FWHM assuming ~1mm per edge
        ratio = 6 / 2.3548  # We want to cover 6 (+-3) sds with a fwhm to sd ratio of 2.3548
        adjacency_depth = int(np.ceil(fwhm * ratio))

    M = get_sparse_adjacency(faces, adjacency_depth=adjacency_depth)

    if coordinates is not None and fwhm is not None:
        X2 = get_sparse_squared_diff(coordinates[:, 0], M)
        Y2 = get_sparse_squared_diff(coordinates[:, 1], M)
        Z2 = get_sparse_squared_diff(coordinates[:, 2], M)

        s = (fwhm / 2.3548)  # Convert FWHM to standard deviation

        D = (X2 + Y2 + Z2).sqrt()

        W = (-(D / (2 * s ** 2))).expm1() + M
        W /= np.sqrt(2 * np.pi) * s
        W = W.astype(np.float32)
    else:
        W = M.astype(np.float32)  # If no coordinates or fwhm, use adjacency matrix as weights

    return W


def apply_geodesic_smoothing_weights(metric, W):
    """
    Apply geodesic smoothing weights to a metric.
    :param metric: Metric to smooth as a numpy array of shape (n_vertices,) or (n_vertices, n_timepoints).
    :param W: Geodesic smoothing weights as a sparse matrix of shape (n_vertices, n_vertices).
    :return: Smoothed metric as a numpy array of shape (n_vertices,) or (n_vertices, n_timepoints).
    """
    assert len(metric.shape) < 3, 'Metric must be a 1D or 2D array.'
    assert metric.shape[0] == W.shape[0], \
        f'Mesh had {W.shape[0]} vertices, so the first dimension of metric must be {W.shape[0]}, ' \
        f'but it was {metric.shape[0]}.'

    denom = np.array(W.sum(axis=1))[:, 0]
    if len(metric.shape) > 1:
        denom = denom[:, None]

    return W @ metric / denom


def smooth_metric_on_surface(metric, faces, coordinates=None, fwhm=None, adjacency_depth=None):
    """
    Smooth a metric on a surface using geodesic smoothing.
    :param metric: Metric to smooth as a numpy array of shape (n_vertices,) or (n_vertices, n_timepoints).
    :param faces: Faces of the mesh as a numpy array of shape (n_faces, 3).
    :param coordinates: Coordinates of the vertices as a numpy array of shape (n_vertices, 3).
    :param fwhm: Full-width at half-maximum for smoothing. If None, no smoothing is applied.
    :param adjacency_depth: Depth of adjacency to compute for smoothing. If None, inferred from fwhm.
    :return: Smoothed metric as a numpy array of shape (n_vertices,) or (n_vertices, n_timepoints).
    """
    if not fwhm:
        smoothed_metric = metric
    else:
        assert len(metric.shape) < 3, 'Metric must be a 1D or 2D array.'
        t0 = time.time()
        W = get_geodesic_smoothing_weights(faces, coordinates=coordinates, fwhm=fwhm, adjacency_depth=adjacency_depth)
        t1 = time.time()
        print(f'Computed geodesic smoothing weights in {t1 - t0:.2f} seconds.')
        t0 = time.time()
        smoothed_metric = apply_geodesic_smoothing_weights(metric, W)
        t1 = time.time()
        print(f'Applied geodesic smoothing weights in {t1 - t0:.2f} seconds.')

    return smoothed_metric


def extract_timecourses(
        functional_paths,
        functional_labels=None,
        mask_path=None,
        parcel_paths=None,
        localizer_path=None,
        criterion='x > np.quantile(x, 0.9)',
        voxelwise=False,
        as_csv=False
):
    raise NotImplementedError
    if isinstance(functional_paths, str):
        functional_paths = [functional_paths]
    assert isinstance(functional_paths, list) or isinstance(functional_paths, tuple), \
        "Functional paths must be a list of file paths."
    assert len(functional_paths) > 0, "No functional data provided."

    if functional_labels is None:
        functional_labels = [f'run_{i+1}' for i in range(len(functional_paths))]
    elif isinstance(functional_labels, str):
        functional_labels = [functional_labels]
    assert isinstance(functional_labels, list) or isinstance(functional_labels, tuple), \
        "Functional labels must be a list of labels."
    assert len(functional_labels) == len(functional_paths), \
        "Functional labels must match the number of functional paths."

    functionals = [
        image.load_img(path) for path in functional_paths
    ]
    nii_ref = functionals[0]
    functionals = [nii_ref] + [image.resample_to_img(x, nii_ref) for x in functionals[1:]]

    # TODO: Implement the rest