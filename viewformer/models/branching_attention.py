import tensorflow as tf
from viewformer.utils.tensorflow import shape_list


def compute_attention(k, v, q, attention_mask=None, attn_dropout=None, training=False):
    with tf.name_scope('attention'):
        w = tf.matmul(q, k, transpose_b=True)

        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        *_other_shape, nd, ns = shape_list(w)
        if attention_mask is not None:
            attention_mask = tf.reshape(attention_mask, ([1] * len(_other_shape)) + [nd, ns])
            w = w * attention_mask - 1e4 * (1 - attention_mask)

        w = tf.nn.softmax(w, axis=-1)
        if attn_dropout is not None:
            w = attn_dropout(w, training=training)
        return tf.matmul(w, v)


def compute_causal_attention(k, v, q, attn_dropout=None, training=False):
    '''
    Warning: this attention attends to its own tokens!
    '''
    with tf.name_scope('causal_attention'):
        # q has shape [B, H, T, d_model]
        # v has shape [B, H, T, d_model]
        # k_i has shape [B, H, T, d_model]
        # T is the sequence length, d_model is the size of the model and B is the batch size
        *_other_shape, ns, _ = shape_list(k)
        nd = tf.shape(q)[-2]
        i = tf.range(nd)[:, None]
        j = tf.range(ns)
        m = i >= j - ns + nd
        causal_mask = tf.cast(m, k.dtype)
        return compute_attention(
            k, v, q,
            attention_mask=causal_mask, attn_dropout=attn_dropout, training=training)


def compute_causal_block_attention(k, v, q, attn_dropout=None, training=False):
    with tf.name_scope('causal_block_attention'):
        # NOTE: this allows attending to itself!!!
        # q has shape [B, H, T, L, d_model]
        # v has shape [B, H, T, L, d_model]
        # k_i has shape [B, H, T, L, d_model]
        # Where L is the block size, T is the sequence length, d_model is the size of the model and B is the batch size
        b, h, ns, l, _ = shape_list(k)
        nd = tf.shape(q)[-3]
        block_size = tf.shape(k)[-2]
        i = tf.repeat(tf.range(nd), block_size)[:, None]
        j = tf.repeat(tf.range(ns), block_size)
        m = i >= j - ns + nd
        causal_mask = tf.cast(m, k.dtype)
        attention = compute_attention(
            tf.reshape(k, [b, h, ns * l, -1]),
            tf.reshape(v, [b, h, ns * l, -1]),
            tf.reshape(q, [b, h, nd * l, -1]),
            attention_mask=causal_mask, attn_dropout=attn_dropout, training=training)
        attention = tf.reshape(attention, [b, h, nd, l, -1])
        return attention


def compute_block_attention(k, v, q, attn_dropout=None, training=False):
    with tf.name_scope('block_attention'):
        # NOTE: this allows attending to itself!!!
        # q has shape [B, H, T, L, d_model]
        # v has shape [B, H, T, L, d_model]
        # k_i has shape [B, H, T, L, d_model]
        # Where L is the block size, T is the sequence length, d_model is the size of the model and B is the batch size
        b, h, ns, l, _ = shape_list(k)
        nd = tf.shape(q)[-3]
        attention = compute_attention(
            tf.reshape(k, [b, h, ns * l, -1]),
            tf.reshape(v, [b, h, ns * l, -1]),
            tf.reshape(q, [b, h, nd * l, -1]),
            attn_dropout=attn_dropout, training=training)
        attention = tf.reshape(attention, [b, h, nd, l, -1])
        return attention


def compute_causal_block_multiend_attention(kset, vset, qset, attn_dropout=None, training=False):
    with tf.name_scope('causal_block_multiend_attention'):
        # q has shape [B, H, T, L, d_model]
        # v has shape [B, H, T, L, d_model]
        # k_i has shape [B, H, T, L, d_model]
        # Where L is the block size, T is the sequence length, d_model is the size of the model and B is the batch size

        # First, we compute the normal causal block attention
        k = kset[0]
        v = vset[0]
        outputs = compute_causal_block_attention(k, v, qset[0], attn_dropout=attn_dropout, training=training),

        # Now, we compute the attentions for each end separatedly
        b, h, ns, l, _ = shape_list(k)
        k_flat = tf.reshape(k[:, :, :-1], [b, h, (ns - 1) * l, -1])
        v_flat = tf.reshape(v[:, :, :-1], [b, h, (ns - 1) * l, -1])
        nd = tf.shape(qset[0])[-3]
        i = tf.repeat(tf.range(nd), l)[:, None]
        j = tf.repeat(tf.range(ns - 1), l)
        m = i >= j - ns + nd + 1
        m = tf.cast(m, k.dtype)
        m = tf.reshape(m, [1, 1, nd * l, (ns - 1) * l])
        for k_new, v_new, q in zip(kset[1:], vset[1:], qset[1:]):
            nd = tf.shape(q)[-3]

            # Compute attn scores wrt old values
            q_flat = tf.reshape(q, [b, h, nd * l, -1])
            w_old = tf.matmul(q_flat, k_flat, transpose_b=True)
            w_old = w_old * m - 1e4 * (1 - m)

            # Compute attn scores wrt new values
            w_new = tf.reshape(tf.matmul(q, k_new, transpose_b=True), [b, h, -1, l])

            # Compute attention values
            w = tf.concat([w_old, w_new], -1)
            w = tf.nn.softmax(w, axis=-1)
            if attn_dropout is not None:
                w = attn_dropout(w, training=training)
            attn_old = tf.matmul(w[:, :, :, :(ns - 1) * l], v_flat)
            attn_old = tf.reshape(attn_old, [b, h, nd, l, -1])
            w_new = tf.reshape(w[:, :, :, (ns - 1) * l:], [b, h, nd, l, l])
            attn_new = tf.einsum('ijklm,ijkmv->ijklv', w_new, v_new)
            attn = attn_old + attn_new
            outputs = outputs + (attn,)
        return outputs


def compute_block_multiend_attention(kset, vset, qset, attn_dropout=None, training=False):
    with tf.name_scope('block_multiend_attention'):
        # q has shape [B, H, T, L, d_model]
        # v has shape [B, H, T, L, d_model]
        # k_i has shape [B, H, T, L, d_model]
        # Where L is the block size, T is the sequence length, d_model is the size of the model and B is the batch size

        # First, we compute the normal causal block attention
        k = kset[0]
        v = vset[0]
        outputs = compute_block_attention(k, v, qset[0], attn_dropout=attn_dropout, training=training),

        # Now, we compute the attentions for each end separatedly
        b, h, ns, l, _ = shape_list(k)
        k_flat = tf.reshape(k, [b, h, ns * l, -1])
        v_flat = tf.reshape(v, [b, h, ns * l, -1])
        nd = tf.shape(qset[0])[-3]
        i = tf.repeat(tf.range(nd), l)[:, None]
        j = tf.repeat(tf.range(ns), l)
        m = i != j
        m = tf.cast(m, k.dtype)
        m = tf.reshape(m, [1, 1, nd * l, ns * l])
        for k_new, v_new, q in zip(kset[1:], vset[1:], qset[1:]):
            nd = tf.shape(q)[-3]

            # Compute attn scores wrt old values
            q_flat = tf.reshape(q, [b, h, nd * l, -1])
            w_old = tf.matmul(q_flat, k_flat, transpose_b=True)
            w_old = w_old * m - 1e4 * (1 - m)

            # Compute attn scores wrt new values
            w_new = tf.reshape(tf.matmul(q, k_new, transpose_b=True), [b, h, -1, l])

            # Compute attention values
            w = tf.concat([w_old, w_new], -1)
            w = tf.nn.softmax(w, axis=-1)
            if attn_dropout is not None:
                w = attn_dropout(w, training=training)
            attn_old = tf.matmul(w[:, :, :, :ns * l], v_flat)
            attn_old = tf.reshape(attn_old, [b, h, nd, l, -1])
            w_new = tf.reshape(w[:, :, :, ns * l:], [b, h, nd, l, l])
            attn_new = tf.einsum('ijklm,ijkmv->ijklv', w_new, v_new)
            attn = attn_old + attn_new
            outputs = outputs + (attn,)
        return outputs


def compute_causal_multiend_attention(kset, vset, qset, attn_dropout=None, training=False):
    with tf.name_scope('causal_multiend_attention'):
        # q has shape [B, H, T, d_model]
        # v has shape [B, H, T, d_model]
        # k_i has shape [B, H, T, d_model]
        # T is the sequence length, d_model is the size of the model and B is the batch size

        # First, we compute the normal causal block attention
        k = kset[0]
        v = vset[0]
        outputs = compute_causal_attention(k, v, qset[0], attn_dropout=attn_dropout, training=training),

        # Now, we compute the attentions for each end separatedly
        *_other_shape, ns, _ = shape_list(k)
        k_flat = k[..., :-1, :]
        v_flat = v[..., :-1, :]
        nd = tf.shape(qset[0])[-2]
        i = tf.range(nd)[:, None]
        j = tf.range(ns - 1)
        m = i >= j - ns + nd + 1
        m = tf.cast(m, k.dtype)
        for k_new, v_new, q in zip(kset[1:], vset[1:], qset[1:]):
            nd = tf.shape(q)[-2]

            # Compute attn scores wrt old values
            w_old = tf.matmul(q, k_flat, transpose_b=True)
            w_old = w_old * m - 1e4 * (1 - m)

            # Compute attn scores wrt new values
            w_new = tf.reduce_sum(q * k_new, -1, keepdims=True)

            # Compute attention values
            w = tf.concat([w_old, w_new], -1)
            w = tf.nn.softmax(w, axis=-1)
            if attn_dropout is not None:
                w = attn_dropout(w, training=training)
            attn_old = tf.matmul(w[..., :ns - 1], v_flat)
            w_new = w[..., ns - 1:]
            attn_new = w_new * v_new
            attn = attn_old + attn_new
            outputs = outputs + (attn,)
        return outputs
