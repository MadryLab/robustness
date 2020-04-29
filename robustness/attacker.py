"""
**For most use cases, this can just be considered an internal class and
ignored.**

This module houses the :class:`robustness.attacker.Attacker` and
:class:`robustness.attacker.AttackerModel` classes. 

:class:`~robustness.attacker.Attacker` is an internal class that should not be
imported/called from outside the library.
:class:`~robustness.attacker.AttackerModel` is a "wrapper" class which is fed a
model and adds to it adversarial attack functionalities as well as other useful
options. See :meth:`robustness.attacker.AttackerModel.forward` for documentation
on which arguments AttackerModel supports, and see
:meth:`robustness.attacker.Attacker.forward` for the arguments pertaining to
adversarial examples specifically.

For a demonstration of this module in action, see the walkthrough
":doc:`../example_usage/input_space_manipulation`"

**Note 1**: :samp:`.forward()` should never be called directly but instead the
AttackerModel object itself should be called, just like with any
:samp:`nn.Module` subclass.

**Note 2**: Even though the adversarial example arguments are documented in
:meth:`robustness.attacker.Attacker.forward`, this function should never be
called directly---instead, these arguments are passed along from
:meth:`robustness.attacker.AttackerModel.forward`.
"""

import torch as ch
import dill
import os
if int(os.environ.get("NOTEBOOK_MODE", 0)) == 1:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

from .tools import helpers
from . import attack_steps

STEPS = {
    'inf': attack_steps.LinfStep,
    '2': attack_steps.L2Step,
    'unconstrained': attack_steps.UnconstrainedStep,
    'fourier': attack_steps.FourierStep,
    'random_smooth': attack_steps.RandomStep
}

class Attacker(ch.nn.Module):
    """
    Attacker class, used to make adversarial examples.

    This is primarily an internal class, you probably want to be looking at
    :class:`robustness.attacker.AttackerModel`, which is how models are actually
    served (AttackerModel uses this Attacker class).

    However, the :meth:`robustness.Attacker.forward` function below
    documents the arguments supported for adversarial attacks specifically.
    """
    def __init__(self, model, dataset):
        """
        Initialize the Attacker

        Args:
            nn.Module model : the PyTorch model to attack
            Dataset dataset : dataset the model is trained on, only used to get mean and std for normalization
        """
        super(Attacker, self).__init__()
        self.normalize = helpers.InputNormalize(dataset.mean, dataset.std)
        self.model = model

    def forward(self, x, target, *_, constraint, eps, step_size, iterations,
                random_start=False, random_restarts=False, do_tqdm=False,
                targeted=False, custom_loss=None, should_normalize=True,
                orig_input=None, use_best=True, return_image=True,
                est_grad=None, mixed_precision=False):
        """
        Implementation of forward (finds adversarial examples). Note that
        this does **not** perform inference and should not be called
        directly; refer to :meth:`robustness.attacker.AttackerModel.forward`
        for the function you should actually be calling.

        Args:
            x, target (ch.tensor) : see :meth:`robustness.attacker.AttackerModel.forward`
            constraint
                ("2"|"inf"|"unconstrained"|"fourier"|:class:`~robustness.attack_steps.AttackerStep`)
                : threat model for adversarial attacks (:math:`\ell_2` ball,
                :math:`\ell_\infty` ball, :math:`[0, 1]^n`, Fourier basis, or
                custom AttackerStep subclass).
            eps (float) : radius for threat model.
            step_size (float) : step size for adversarial attacks.
            iterations (int): number of steps for adversarial attacks.
            random_start (bool) : if True, start the attack with a random step.
            random_restarts (bool) : if True, do many random restarts and
                take the worst attack (in terms of loss) per input.
            do_tqdm (bool) : if True, show a tqdm progress bar for the attack.
            targeted (bool) : if True (False), minimize (maximize) the loss.
            custom_loss (function|None) : if provided, used instead of the
                criterion as the loss to maximize/minimize during
                adversarial attack. The function should take in
                :samp:`model, x, target` and return a tuple of the form
                :samp:`loss, None`, where loss is a tensor of size N
                (per-element loss).
            should_normalize (bool) : If False, don't normalize the input
                (not recommended unless normalization is done in the
                custom_loss instead).
            orig_input (ch.tensor|None) : If not None, use this as the
                center of the perturbation set, rather than :samp:`x`.
            use_best (bool) : If True, use the best (in terms of loss)
                iterate of the attack process instead of just the last one.
            return_image (bool) : If True (default), then return the adversarial
                example as an image, otherwise return it in its parameterization
                (for example, the Fourier coefficients if 'constraint' is
                'fourier')
            est_grad (tuple|None) : If not None (default), then these are
                :samp:`(query_radius [R], num_queries [N])` to use for estimating the
                gradient instead of autograd. We use the spherical gradient
                estimator, shown below, along with antithetic sampling [#f1]_
                to reduce variance:
                :math:`\\nabla_x f(x) \\approx \\sum_{i=0}^N f(x + R\\cdot
                \\vec{\\delta_i})\\cdot \\vec{\\delta_i}`, where
                :math:`\delta_i` are randomly sampled from the unit ball.
            mixed_precision (bool) : if True, use mixed-precision calculations
                to compute the adversarial examples / do the inference.
        Returns:
            An adversarial example for x (i.e. within a feasible set
            determined by `eps` and `constraint`, but classified as:

            * `target` (if `targeted == True`)
            *  not `target` (if `targeted == False`)

        .. [#f1] This means that we actually draw :math:`N/2` random vectors
            from the unit ball, and then use :math:`\delta_{N/2+i} =
            -\delta_{i}`.
        """
        # Can provide a different input to make the feasible set around
        # instead of the initial point
        if orig_input is None: orig_input = x.detach()
        orig_input = orig_input.cuda()

        # Multiplier for gradient ascent [untargeted] or descent [targeted]
        m = -1 if targeted else 1

        # Initialize step class and attacker criterion
        criterion = ch.nn.CrossEntropyLoss(reduction='none')
        step_class = STEPS[constraint] if isinstance(constraint, str) else constraint
        step = step_class(eps=eps, orig_input=orig_input, step_size=step_size) 

        def calc_loss(inp, target):
            '''
            Calculates the loss of an input with respect to target labels
            Uses custom loss (if provided) otherwise the criterion
            '''
            if should_normalize:
                inp = self.normalize(inp)
            output = self.model(inp)
            if custom_loss:
                return custom_loss(self.model, inp, target)

            return criterion(output, target), output

        # Main function for making adversarial examples
        def get_adv_examples(x):
            # Random start (to escape certain types of gradient masking)
            if random_start:
                x = step.random_perturb(x)

            iterator = range(iterations)
            if do_tqdm: iterator = tqdm(iterator)

            # Keep track of the "best" (worst-case) loss and its
            # corresponding input
            best_loss = None
            best_x = None

            # A function that updates the best loss and best input
            def replace_best(loss, bloss, x, bx):
                if bloss is None:
                    bx = x.clone().detach()
                    bloss = loss.clone().detach()
                else:
                    replace = m * bloss < m * loss
                    bx[replace] = x[replace].clone().detach()
                    bloss[replace] = loss[replace]

                return bloss, bx

            # PGD iterates
            for _ in iterator:
                x = x.clone().detach().requires_grad_(True)
                losses, out = calc_loss(step.to_image(x), target)
                assert losses.shape[0] == x.shape[0], \
                        'Shape of losses must match input!'

                loss = ch.mean(losses)

                if step.use_grad:
                    if (est_grad is None) and mixed_precision:
                        with amp.scale_loss(loss, []) as sl:
                            sl.backward()
                        grad = x.grad.detach()
                        x.grad.zero_()
                    elif (est_grad is None):
                        grad, = ch.autograd.grad(m * loss, [x])
                    else:
                        f = lambda _x, _y: m * calc_loss(step.to_image(_x), _y)[0]
                        grad = helpers.calc_est_grad(f, x, target, *est_grad)
                else:
                    grad = None

                with ch.no_grad():
                    args = [losses, best_loss, x, best_x]
                    best_loss, best_x = replace_best(*args) if use_best else (losses, x)

                    x = step.step(x, grad)
                    x = step.project(x)
                    if do_tqdm: iterator.set_description("Current loss: {l}".format(l=loss))

            # Save computation (don't compute last loss) if not use_best
            if not use_best: 
                ret = x.clone().detach()
                return step.to_image(ret) if return_image else ret

            losses, _ = calc_loss(step.to_image(x), target)
            args = [losses, best_loss, x, best_x]
            best_loss, best_x = replace_best(*args)
            return step.to_image(best_x) if return_image else best_x

        # Random restarts: repeat the attack and find the worst-case
        # example for each input in the batch
        if random_restarts:
            to_ret = None

            orig_cpy = x.clone().detach()
            for _ in range(random_restarts):
                adv = get_adv_examples(orig_cpy)

                if to_ret is None:
                    to_ret = adv.detach()

                _, output = calc_loss(adv, target)
                corr, = helpers.accuracy(output, target, topk=(1,), exact=True)
                corr = corr.byte()
                misclass = ~corr
                to_ret[misclass] = adv[misclass]

            adv_ret = to_ret
        else:
            adv_ret = get_adv_examples(x)

        return adv_ret

class AttackerModel(ch.nn.Module):
    """
    Wrapper class for adversarial attacks on models. Given any normal
    model (a ``ch.nn.Module`` instance), wrapping it in AttackerModel allows
    for convenient access to adversarial attacks and other applications.::

        model = ResNet50()
        model = AttackerModel(model)
        x = ch.rand(10, 3, 32, 32) # random images
        y = ch.zeros(10) # label 0
        out, new_im = model(x, y, make_adv=True) # adversarial attack
        out, new_im = model(x, y, make_adv=True, targeted=True) # targeted attack
        out = model(x) # normal inference (no label needed)

    More code examples available in the documentation for `forward`.
    For a more comprehensive overview of this class, see 
    :doc:`our detailed walkthrough <../example_usage/input_space_manipulation>`.
    """
    def __init__(self, model, dataset):
        super(AttackerModel, self).__init__()
        self.normalizer = helpers.InputNormalize(dataset.mean, dataset.std)
        self.model = model
        self.attacker = Attacker(model, dataset)

    def forward(self, inp, target=None, make_adv=False, with_latent=False,
                fake_relu=False, no_relu=False, with_image=True, **attacker_kwargs):
        """
        Main function for running inference and generating adversarial
        examples for a model.

        Parameters:
            inp (ch.tensor) : input to do inference on [N x input_shape] (e.g. NCHW)
            target (ch.tensor) : ignored if `make_adv == False`. Otherwise,
                labels for adversarial attack.
            make_adv (bool) : whether to make an adversarial example for
                the model. If true, returns a tuple of the form
                :samp:`(model_prediction, adv_input)` where
                :samp:`model_prediction` is a tensor with the *logits* from
                the network.
            with_latent (bool) : also return the second-last layer along
                with the logits. Output becomes of the form
                :samp:`((model_logits, model_layer), adv_input)` if
                :samp:`make_adv==True`, otherwise :samp:`(model_logits, model_layer)`.
            fake_relu (bool) : useful for activation maximization. If
                :samp:`True`, replace the ReLUs in the last layer with
                "fake ReLUs," which are ReLUs in the forwards pass but
                identity in the backwards pass (otherwise, maximizing a
                ReLU which is dead is impossible as there is no gradient).
            no_relu (bool) : If :samp:`True`, return the latent output with
                the (pre-ReLU) output of the second-last layer, instead of the
                post-ReLU output. Requires :samp:`fake_relu=False`, and has no
                visible effect without :samp:`with_latent=True`.
            with_image (bool) : if :samp:`False`, only return the model output
                (even if :samp:`make_adv == True`).

        """
        if make_adv:
            assert target is not None
            prev_training = bool(self.training)
            self.eval()
            adv = self.attacker(inp, target, **attacker_kwargs)
            if prev_training:
                self.train()

            inp = adv

        normalized_inp = self.normalizer(inp)

        if no_relu and (not with_latent):
            print("WARNING: 'no_relu' has no visible effect if 'with_latent is False.")
        if no_relu and fake_relu:
            raise ValueError("Options 'no_relu' and 'fake_relu' are exclusive")

        output = self.model(normalized_inp, with_latent=with_latent,
                                fake_relu=fake_relu, no_relu=no_relu)
        if with_image:
            return (output, inp)
        return output
