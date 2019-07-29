import tensorflow as tf
import os, sys, time


def cg_pseudoinverse(forward_gramian,rhs, eta):
    """
    Implementation of conjugate gradient iterations in Tensorflow, from MoDL.
    See their paper, code, and repo at: https://github.com/hkaggarwal/modl
    Requires function handle to gramian of forward operator.
    """
    n_iterations = 10
    cond=lambda i,rTr,*_: tf.logical_and( tf.less(i,n_iterations), rTr>1e-10)
    def body(i,rTr,x,r,p):
        with tf.name_scope('cgBody'):
            Ap=forward_gramian(p) + eta*p
            alpha = rTr / tf.to_float(tf.reduce_sum(tf.conj(p)*Ap))
            # alpha=tf.complex(alpha,0.)
            x = x + alpha * p
            r = r - alpha * Ap
            rTrNew = tf.to_float( tf.reduce_sum(r*r))
            beta = rTrNew / rTr
            p = r + beta * p
        return i+1,rTrNew,x,r,p

    x=tf.zeros_like(rhs)
    i,r,p=0,rhs,rhs
    rTr = tf.to_float( tf.reduce_sum(r*r),)
    loopVar=i,rTr,x,r,p
    out=tf.while_loop(cond,body,loopVar,name='CGwhile',parallel_iterations=1)[2]
    return out
