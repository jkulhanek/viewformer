from itertools import chain
import torch
import tensorflow as tf
from .convert import zip_th_tf_parameters, convert_weights_th_to_tf


def assert_weights_same(torch_module, tf_module, atol=1e-6, rtol=1e-5):
    for th_weight, tf_variable in zip_th_tf_parameters(torch_module, tf_module, permute_torch=True):
        tf_weight = torch.tensor(tf_variable.numpy())
        torch.testing.assert_allclose(tf_weight, th_weight, atol=atol, rtol=rtol)


def _assert_torch_modules_same(m1, m2, input_shape, atol, rtol, transform_weights=None):
    flat_torch_inputs = []

    def generate_input(input_shape):
        if isinstance(input_shape[0], tuple):
            return tuple(map(generate_input, input_shape))
        else:
            inp = torch.randn(input_shape, dtype=torch.float32)
            inp.requires_grad = True
            flat_torch_inputs.append(inp)
            return inp

    inp = generate_input(input_shape)
    if not isinstance(inp, tuple):
        inp = (inp,)
    state_dict = m1.state_dict()
    if transform_weights is not None:
        state_dict = dict(map(transform_weights, state_dict.items()))
    m2.load_state_dict(state_dict)

    m1.train()
    m1.zero_grad()
    m2.train()
    m2.zero_grad()
    o1 = m1(*inp)
    o2 = m2(*inp)

    def assert_weights_same(m1, m2, atol=atol, rtol=rtol):
        s1 = m1.state_dict()
        if transform_weights is not None:
            s1 = dict(map(transform_weights, s1.items()))
        s2 = m2.state_dict()
        for v1, v2 in zip(s1.values(), s2.values()):
            torch.testing.assert_allclose(v1, v2, atol=atol, rtol=rtol)

    assert_weights_same(m1, m2, atol=atol, rtol=rtol)

    def assert_same(o1, o2):
        if isinstance(o1, torch.Tensor):
            assert o1.shape == o2.shape
            torch.testing.assert_allclose(o1, o2, atol=atol, rtol=rtol)
        elif isinstance(o1, (tuple, list)):
            for x1, x2 in zip(o1, o2):
                assert_same(x1, x2)
    assert_same(o1, o2)

    # Assert loss is same
    def generate_losses(o1, o2):
        if isinstance(o1, tuple):
            weights = torch.randn((len(o1),), dtype=torch.float32)
            l1, l2 = 0, 0
            for w, m_o1, m_o2 in zip(weights, o1, o2):
                m_o1, m_o2 = generate_losses(m_o1, m_o2)
                l1 += w * m_o1.float()
                l2 += w * m_o2.float()
            return l1, l2
        elif len(o1.shape) == 0 or len(o2) == 1:
            return o1.view(tuple()), o2.view(tuple())
        else:
            weights = torch.randn(o1.shape, dtype=torch.float32)
            l1 = (o1 * weights).mean()
            l2 = (o2 * weights).mean()
            return l1, l2

    l1, l2 = generate_losses(o1, o2)
    assert abs(l1.item() - l2.item()) < atol

    # Assert weights are the same after backprop
    l1.backward()
    grad1_input = [x.grad.clone().detach() for x in flat_torch_inputs]
    if any(True for x in m1.parameters()):
        torch.optim.SGD(m1.parameters(), 0.01, 0.0, nesterov=False).step()
    for p in flat_torch_inputs:
        p.grad = None
    l2.backward()
    grad2_input = [x.grad.clone().detach() for x in flat_torch_inputs]
    if any(True for x in m2.parameters()):
        torch.optim.SGD(m2.parameters(), 0.01, 0.0, nesterov=False).step()
    assert_weights_same(m1, m2)

    # Assert gradient wrt. input is the same
    for g1, g2 in zip(grad1_input, grad2_input):
        torch.testing.assert_allclose(g1, g2, atol=atol, rtol=rtol)


def assert_modules_same(torch_module, tf_module, input_shape, atol=1e-5, transpose=True, transform_weights=None, rtol=1e-5):
    if isinstance(tf_module, torch.nn.Module):
        return _assert_torch_modules_same(torch_module, tf_module, input_shape, atol=atol, transform_weights=transform_weights, rtol=rtol)
    # We will start by copying weights
    flat_tf_inputs = []
    flat_torch_inputs = []

    def generate_input(input_shape):
        if isinstance(input_shape[0], tuple):
            return tuple(zip(*map(generate_input, input_shape)))
        else:
            inp = torch.randn(input_shape, dtype=torch.float32)
            inp.requires_grad = True
            flat_torch_inputs.append(inp)
            if len(input_shape) == 4:
                tf_inp = inp.permute(0, 2, 3, 1)
            else:
                tf_inp = inp
            tf_inp = tf_inp.detach().clone().numpy()
            tf_inp = tf.Variable(tf_inp, trainable=True)
            flat_tf_inputs.append(tf_inp)
            return inp, tf_inp

    inp, tf_inp = generate_input(input_shape)
    if not isinstance(inp, tuple):
        inp = (inp,)
        tf_inp = (tf_inp,)
    tf_module(*tf_inp)
    convert_weights_th_to_tf(torch_module, tf_module)
    # tf_module(tf_inp)

    torch_module.train()
    torch_module.zero_grad()
    torch_output = torch_module(*inp)
    with tf.GradientTape() as tape:
        tf_output = tf_module(*tf_inp, training=True)
        assert_weights_same(torch_module, tf_module)

        def assert_same(o1, o2):
            if isinstance(o1, torch.Tensor):
                o2 = torch.tensor(o2.numpy())
                if len(o2.shape) == 4:
                    o2 = o2.permute(0, 3, 1, 2)
                assert o1.shape == o2.shape
                torch.testing.assert_allclose(o1, o2, atol=atol, rtol=rtol)
            elif isinstance(o1, (tuple, list)):
                for x1, x2 in zip(o1, o2):
                    assert_same(x1, x2)
        assert_same(torch_output, tf_output)

        # Assert loss is same
        def generate_losses(th_output, tf_output):
            if isinstance(th_output, tuple):
                weights = torch.randn((len(th_output),), dtype=torch.float32)
                tf_loss, th_loss = 0, 0
                for w, th_o, tf_o in zip(weights, th_output, tf_output):
                    th_o, tf_o = generate_losses(th_o, tf_o)
                    tf_loss += tf.cast(tf_o, tf.float32) * w
                    th_loss += w * th_o.float()
                return th_loss, tf_loss
            elif len(th_output.shape) == 0 or len(th_output) == 1:
                return th_output.view(tuple()), tf.reshape(tf_output, [])
            else:
                if len(tf_output.shape) == 4:
                    tf_output = tf.transpose(tf_output, [0, 3, 1, 2])

                weights = torch.randn(th_output.shape, dtype=torch.float32)
                th_loss = (th_output * weights).mean()
                tf_loss = tf.reduce_mean(tf.cast(tf_output, tf.float32) * weights.numpy())
                return th_loss, tf_loss

        th_loss, tf_loss = generate_losses(torch_output, tf_output)
        assert abs(th_loss.item() - tf_loss.numpy()) < atol

    # Assert weights are the same after backprop
    tf_grads = tape.gradient(tf_loss, list(chain(tf_module.trainable_variables, flat_tf_inputs)))
    tf.keras.optimizers.SGD(0.01).apply_gradients(zip(tf_grads, tf_module.trainable_variables))
    th_loss.backward()
    if any(True for x in torch_module.parameters()):
        torch.optim.SGD(torch_module.parameters(), 0.01, 0.0, nesterov=False).step()
    assert_weights_same(torch_module, tf_module, atol=atol, rtol=rtol)

    # Assert gradient wrt. input is the same
    tf_grads = tf_grads[-len(flat_tf_inputs):]
    for th_var, tf_var in zip(flat_torch_inputs, tf_grads):
        if len(tf_var.shape) == 4:
            tf_var = tf.transpose(tf_var, [0, 3, 1, 2])
        torch.testing.assert_allclose(th_var.grad, torch.tensor(tf_var.numpy()), atol=atol, rtol=rtol)
