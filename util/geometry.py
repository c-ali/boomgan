import numpy as np

def orthogonalize(normal, non_ortho):
    h = normal * non_ortho
    return non_ortho - normal * h


def make_orthonormal_vector(normal, dims=512):
    # random unit vector
    rand_dir = np.random.randn(dims)

    # make orthonormal
    result = orthogonalize(normal, rand_dir)
    return result / np.linalg.norm(result)


def random_circle(radius, ndim):
    '''Given a radius, parametrizes a random circle'''
    n1 = np.random.randn(ndim)
    n1 /= np.linalg.norm(n1)
    n2 = make_orthonormal_vector(n1, ndim)

    def circle(theta):
        return np.repeat(n1[None, :], theta.shape[0], axis=0) * np.cos(theta)[:, None] * radius + np.repeat(n2[None, :], theta.shape[0], axis=0) * np.sin(theta)[:, None] * radius
    return circle
