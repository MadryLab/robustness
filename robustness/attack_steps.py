"""
**For most use cases, this can just be considered an internal class and
ignored.**

This module contains the abstract class AttackerStep as well as a few subclasses. 

AttackerStep is a generic way to implement optimizers specifically for use with
:class:`robustness.attacker.AttackerModel`. In general, except for when you want
to :ref:`create a custom optimization method <adding-custom-steps>`, you probably do not need to
import or edit this module and can just think of it as internal.
"""

import torch as ch

class AttackerStep:
    '''
    Generic class for attacker steps, under perturbation constraints
    specified by an "origin input" and a perturbation magnitude.
    Must implement project, step, and random_perturb
    '''
    def __init__(self, orig_input, eps, step_size, 
                instance_attack_function, use_grad=True):
        '''
        Initialize the attacker step with a given perturbation magnitude.

        Args:
            eps (float): the perturbation magnitude
            orig_input (ch.tensor): the original input
        '''
        self.orig_input = orig_input
        self.eps = eps
        self.step_size = step_size
        self.use_grad = use_grad
        self.instance_attack_function = instance_attack_function

    def project(self, x, target=None, prediction=None):
        '''
        Given an input x, project it back into the feasible set

        Args:
            ch.tensor x : the input to project back into the feasible set.

        Returns:
            A `ch.tensor` that is the input projected back into
            the feasible set, that is,
        .. math:: \min_{x' \in S} \|x' - x\|_2
        '''
        raise NotImplementedError

    def step(self, x, g, target=None, prediction=None):
        '''
        Given a gradient, make the appropriate step according to the
        perturbation constraint (e.g. dual norm maximization for :math:`\ell_p`
        norms).

        Parameters:
            g (ch.tensor): the raw gradient

        Returns:
            The new input, a ch.tensor for the next step.
        '''
        raise NotImplementedError

    def random_perturb(self, x, target=None, prediction=None):
        '''
        Given a starting input, take a random step within the feasible set
        '''
        raise NotImplementedError

    def to_image(self, x):
        '''
        Given an input (which may be in an alternative parameterization),
        convert it to a valid image (this is implemented as the identity
        function by default as most of the time we use the pixel
        parameterization, but for alternative parameterizations this functino
        must be overriden).
        '''
        return x

### Instantiations of the AttackerStep class

# L-infinity threat model
class LinfStep(AttackerStep):
    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:

    .. math:: S = \{x | \|x - x_0\|_\infty \leq \epsilon\}
    """
    def project(self, x):
        """
        """
        diff = x - self.orig_input
        diff = ch.clamp(diff, -self.eps, self.eps)
        return ch.clamp(diff + self.orig_input, 0, 1)

    def step(self, x, g):
        """
        """
        step = ch.sign(g) * self.step_size
        return x + step

    def random_perturb(self, x):
        """
        """
        new_x = x + 2 * (ch.rand_like(x) - 0.5) * self.eps
        return ch.clamp(new_x, 0, 1)

# L2 threat model
class L2Step(AttackerStep):
    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:

    .. math:: S = \{x | \|x - x_0\|_2 \leq \epsilon\}
    """
    def project(self, x, target=None, prediction=None):
        """
        """
        diff = x - self.orig_input

        # Allow for per instance eps
        if self.instance_attack_function is None: 
            diff = diff.renorm(p=2, dim=0, maxnorm=self.eps)
        else:
            eps_use, _ = self.instance_attack_function(x, target, prediction)
            diff = ch.cat([diff[idx:idx+1].renorm(p=2, dim=0, maxnorm=self.eps * eps_use[idx]) \
                          for idx in range(len(diff))])
        return ch.clamp(self.orig_input + diff, 0, 1)

    def step(self, x, g, target=None, prediction=None):
        """
        """
        # Scale g so that each element of the batch is at least norm 1
        l = len(x.shape) - 1
        g_norm = ch.norm(g.view(g.shape[0], -1), dim=1).view(-1, *([1]*l))
        scaled_g = g / (g_norm + 1e-10)

        if self.instance_attack_function is None: 
            return x + scaled_g * self.step_size
        else:
            _, step_use = self.instance_attack_function(x, target, prediction)
            step_size = ch.tensor(self.step_size * step_use)
            step_size = step_size[:, None, None, None].to(scaled_g.device,
                                                          dtype=scaled_g.dtype)
            return x + scaled_g * step_size

    def random_perturb(self, x, target=None, prediction=None):
        """
        """
        if self.instance_attack_function is None: 
            new_x = x + (ch.rand_like(x) - 0.5).renorm(p=2, dim=0, maxnorm=self.eps)
        else:
            eps_use, _ = self.instance_attack_function(x, target, prediction)
            rand_x = (ch.rand_like(x) - 0.5)
            rand_x = ch.cat([rand_x[idx:idx+1].renorm(p=2, dim=0, maxnorm=self.eps * eps_use[idx]) \
                          for idx in range(len(rand_x))])
            new_x = x + rand_x
        return ch.clamp(new_x, 0, 1)

# Unconstrained threat model
class UnconstrainedStep(AttackerStep):
    """
    Unconstrained threat model, :math:`S = [0, 1]^n`.
    """
    def project(self, x):
        """
        """
        return ch.clamp(x, 0, 1)

    def step(self, x, g):
        """
        """
        return x + g * self.step_size

    def random_perturb(self, x):
        """
        """
        new_x = x + (ch.rand_like(x) - 0.5).renorm(p=2, dim=0, maxnorm=step_size)
        return ch.clamp(new_x, 0, 1)

class FourierStep(AttackerStep):
    """
    Step under the Fourier (decorrelated) parameterization of an image.

    See https://distill.pub/2017/feature-visualization/#preconditioning for more information.
    """
    def project(self, x):
        """
        """
        return x

    def step(self, x, g):
        """
        """
        return x + g * self.step_size

    def random_perturb(self, x):
        """
        """
        return x

    def to_image(self, x):
        """
        """
        return ch.sigmoid(ch.irfft(x, 2, normalized=True, onesided=False))