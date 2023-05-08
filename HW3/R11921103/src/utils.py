import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A
    A = []
    for i in range(N):
        A.append([u[i][0], u[i][1], 1, 0, 0, 0, -u[i][0] * v[i][0], -u[i][1] * v[i][0], -v[i][0]])
        A.append([0, 0, 0, u[i][0], u[i][1], 1, -u[i][0] * v[i][1], -u[i][1] * v[i][1], -v[i][1]])
                 
    # TODO: 2.solve H with A
    res = np.linalg.svd(A, full_matrices=True)
    H = np.reshape(res[2][-1], (3,3))

    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """
    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    x, y = np.meshgrid(range(xmin,xmax),range(ymin,ymax))
    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    before = np.array([x, y, np.ones(x.shape)]).reshape((3, -1))

    if direction == 'b':
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        after = np.matmul(H_inv, before)
        after_x, after_y = (after[:2,] / after[2]).astype(int)
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        mask = np.logical_and(np.logical_and(after_x >= 0, after_x < w_src), np.logical_and(after_y >= 0, after_y < h_src))
        after_x, after_y = after_x[mask], after_y[mask]
        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        sample = src[after_y, after_x]
        # TODO: 6. assign to destination image with proper masking
        dst[ymin:ymax, xmin:xmax][np.reshape(mask, (ymax-ymin, xmax-xmin))] = sample

    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        after = np.matmul(H, before)
        after_x, after_y = (after[:2,] / after[2]).astype(int)
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        mask = np.logical_and(np.logical_and(after_x >= 0, after_x < w_dst), np.logical_and(after_y >= 0, after_y < h_dst))
        # TODO: 5.filter the valid coordinates using previous obtained mask
        after_x, after_y = after_x[mask], after_y[mask]
        # TODO: 6. assign to destination image using advanced array indicing
        dst[after_y, after_x] = np.reshape(src, (-1, 3))

    return dst
