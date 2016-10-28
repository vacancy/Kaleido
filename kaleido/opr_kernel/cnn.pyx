# -*- coding:utf8 -*-
# File   : cnn.pyx
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/27/16 19:57
# 
# This file is part of Kaleido
# (c) 2016 vccy.xyz


import numpy as np
cimport numpy as np

dtype = np.float32
ctypedef np.float32_t dtype_t


def conv2d_forward(np.ndarray[dtype_t, ndim=4] src, np.ndarray[dtype_t, ndim=4] kernel,
                   int ph, int pw, int sh, int sw):
    assert src.shape[1] == kernel.shape[1], 'input channel mismatch'
    assert src.shape[2] >= kernel.shape[2] and src.shape[3] >= kernel.shape[3], 'input size too small'

    cdef int n = src.shape[0]
    cdef int ih = src.shape[2], iw = src.shape[3], ic = src.shape[1]
    cdef int kh = kernel.shape[2], kw = kernel.shape[3]
    cdef int oc = kernel.shape[0]
    cdef int oh = (ih + 2 * ph - kh) / sh + 1
    cdef int ow = (iw + 2 * pw - kw) / sw + 1

    src_padded = np.pad(src, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode='constant')
    out = np.zeros([n, oc, oh, ow], dtype=src.dtype)

    for i in range(n):
        for j in range(oc):
            tmp = np.zeros([oh, ow], dtype=src.dtype)
            for k in range(ic):
                sub_kernel = kernel[j, k, :, :]
                sub_img = src_padded[j, k, :, :]
                for oy in range(oh):
                    y0 = oy * sh
                    y1 = y0 + kh
                    for ox in range(ow):
                        x0 = ox * sw
                        x1 = x0 + kw
                        tmp[oy, ox] += np.sum(sub_img[y0:y1, x0:x1] * sub_kernel)
            out[i, j] = tmp

    return out


def pooling2d_forward(np.ndarray[dtype_t, ndim=4] src, int kh, int kw, int ph, int pw, int sh, int sw, method):
    cdef int n = src.shape[0]
    cdef int ih = src.shape[2], iw = src.shape[3], ic = src.shape[1]
    cdef int oh = (ih + 2 * ph - kh) / sh + 1
    cdef int ow = (iw + 2 * pw - kw) / sw + 1

    src_padded = np.pad(src, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode='constant')
    out = np.zeros([n, ic, oh, ow], dtype=src.dtype)

    for i in range(n):
        for c in range(ic):
            for oy in range(oh):
                y0 = oy * sh
                y1 = y0 + kh
                for ox in range(ow):
                    x0 = ox * sw
                    x1 = x0 + kw
                    sub_src = src_padded[i, c, y0:y1, x0:x1]
                    if method == 'MAX':
                        out[i, c, oy, ox] = sub_src.max()
                    else:
                        out[i, c, oy, ox] = sub_src.mean()
    return out


def conv2d_backward_data(np.ndarray[dtype_t, ndim=4] grad,
                         np.ndarray[dtype_t, ndim=4] src, np.ndarray[dtype_t, ndim=4] kernel,
                         int ph, int pw, int sh, int sw):

    cdef int n = src.shape[0]
    cdef int ih = src.shape[2], iw = src.shape[3], ic = src.shape[1]
    cdef int kh = kernel.shape[2], kw = kernel.shape[3]
    cdef int oc = kernel.shape[0]
    cdef int oh = (ih + 2 * ph - kh) / sh + 1
    cdef int ow = (iw + 2 * pw - kw) / sw + 1

    out = np.zeros([n, ic, ih + 2 * ph, iw + 2 * pw])

    for i in range(n):
        for j in range(ic):
            for k in range(oc):
                x_hat = np.dot(grad[i, j, :, :].reshape(-1, 1), kernel[k, j, :, :].reshape(1, -1))
                x_hat = x_hat.reshape(oh, ow, kh, kw)

                for oy in range(oh):
                    y0 = oy * sh
                    y1 = y0 + kh
                    for ox in range(ow):
                        x0 = ox * sw
                        x1 = x0 + kw
                        out[i, j, y0:y1, x0:x1] += x_hat[oy, ox, :, :]
    return out[:, :, ph:-ph, pw:-pw]


def conv2d_backward_kernel(np.ndarray[dtype_t, ndim=4] grad,
                           np.ndarray[dtype_t, ndim=4] src, np.ndarray[dtype_t, ndim=4] kernel,
                           int ph, int pw, int sh, int sw):
    cdef int n = src.shape[0]
    cdef int ih = src.shape[2], iw = src.shape[3], ic = src.shape[1]
    cdef int kh = kernel.shape[2], kw = kernel.shape[3]
    cdef int oc = kernel.shape[0]
    cdef int oh = (ih + 2 * ph - kh) / sh + 1
    cdef int ow = (iw + 2 * pw - kw) / sw + 1

    src_padded = np.pad(src, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode='constant')
    out = np.zeros_like(kernel)

    for k in range(oc):
        for j in range(ic):
            w_hat = np.zeros((kh, kw), dtype=kernel.dtype)
            for i in range(n):
                sub_src = src_padded[i, j, :, :]
                sub_grad = grad[i, k, :, :]

                for ky in range(kh):
                    for kx in range(kw):
                        ss_src = sub_src[ky:ky+oh*sh:sh, kx:kx+ow*sw:sw]
                        w_hat[ky, kx] += np.sum(ss_src * sub_grad)
            out[k, j, :, :] += w_hat


def pooling2d_backward(np.ndarray[dtype_t, ndim=4] grad, np.ndarray[dtype_t, ndim=4] src,
                       int kh, int kw, int ph, int pw, int sh, int sw, method):

    cdef int n = src.shape[0]
    cdef int ih = src.shape[2], iw = src.shape[3], ic = src.shape[1]
    cdef int oh = (ih + 2 * ph - kh) / sh + 1
    cdef int ow = (iw + 2 * pw - kw) / sw + 1
    cdef int ks = kw * kh

    src_padded = np.pad(src, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode='constant')
    out = np.zeros([n, ic, ih + 2 * ph, iw + 2 * pw], dtype=src.dtype)

    for i in range(n):
        for c in range(ic):
            for oy in range(oh):
                y0 = oy * sh
                y1 = y0 + kh
                for ox in range(ow):
                    x0 = ox * sw
                    x1 = x0 + kw
                    sub_src = src_padded[i, c, y0:y1, x0:x1]
                    if method == 'MAX':
                        ind = np.unravel_index(sub_src.argmax(), sub_src.shape)
                        out[i, c, y0+ind[0], x0+ind[1]] += grad[i, c, oy, ox]
                    else:
                        out[i, c, y0:y1, x0:x1] += grad[i, c, oy, ox] / float(ks)

    return out[:, :, ph:-ph, pw:-pw]
