import abc
from typing import Union, Tuple, Optional
import torch
import re

__all__ = ["Euclidean"]

import itertools
from typing import Tuple, Any, Union, List
import torch.jit
import functools
import operator
from . import functional

__all__ = [
    "strip_tuple",
    "size2shape",
    "make_tuple",
    "broadcast_shapes",
    "ismanifold",
    "canonical_manifold",
    "list_range",
    "idx2sign",
    "drop_dims",
    "canonical_dims",
    "sign",
    "prod",
    "clamp_abs",
    "sabs",
]

COMPLEX_DTYPES = {torch.complex64, torch.complex128}
if hasattr(torch, "complex32"):
    COMPLEX_DTYPES.add(torch.complex32)


def strip_tuple(tup: Tuple) -> Union[Tuple, Any]:
    if len(tup) == 1:
        return tup[0]
    else:
        return tup


def make_tuple(obj: Union[Tuple, List, Any]) -> Tuple:
    if isinstance(obj, list):
        obj = tuple(obj)
    if not isinstance(obj, tuple):
        return (obj,)
    else:
        return obj


def prod(items):
    return functools.reduce(operator.mul, items, 1)


@torch.jit.script
def sign(x):
    return torch.sign(x.sign() + 0.5)


@torch.jit.script
def sabs(x, eps: float = 1e-15):
    return x.abs().add_(eps)


@torch.jit.script
def clamp_abs(x, eps: float = 1e-15):
    s = sign(x)
    return s * sabs(x, eps=eps)


@torch.jit.script
def idx2sign(idx: int, dim: int, neg: bool = True):
    """
    Unify idx to be negative or positive, that helps in cases of broadcasting.

    Parameters
    ----------
    idx : int
        current index
    dim : int
        maximum dimension
    neg : bool
        indicate we need negative index

    Returns
    -------
    int
    """
    if neg:
        if idx < 0:
            return idx
        else:
            return (idx + 1) % -(dim + 1)
    else:
        return idx % dim


@torch.jit.script
def drop_dims(tensor: torch.Tensor, dims: List[int]):
    # Workaround to drop several dims in :func:`torch.squeeze`.
    seen: int = 0
    for d in dims:
        tensor = tensor.squeeze(d - seen)
        seen += 1
    return tensor


@torch.jit.script
def list_range(end: int):
    res: List[int] = []
    for d in range(end):
        res.append(d)
    return res


@torch.jit.script
def canonical_dims(dims: List[int], maxdim: int):
    result: List[int] = []
    for idx in dims:
        result.append(idx2sign(idx, maxdim, neg=False))
    return result


def size2shape(*size: Union[Tuple[int], int]) -> Tuple[int]:
    return make_tuple(strip_tuple(size))


def broadcast_shapes(*shapes: Tuple[int]) -> Tuple[int]:
    """Apply numpy broadcasting rules to shapes."""
    result = []
    for dims in itertools.zip_longest(*map(reversed, shapes), fillvalue=1):
        dim: int = 1
        for d in dims:
            if dim != 1 and d != 1 and d != dim:
                raise ValueError("Shapes can't be broadcasted")
            elif d > dim:
                dim = d
        result.append(dim)
    return tuple(reversed(result))


def ismanifold(instance, cls):
    """
    Check if interface of an instance is compatible with given class.

    Parameters
    ----------
    instance : geoopt.Manifold
        check if a given manifold is compatible with cls API
    cls : type
        manifold type

    Returns
    -------
    bool
        comparison result
    """
    if not issubclass(cls, Manifold):
        raise TypeError("`cls` should be a subclass of geoopt.manifolds.Manifold")
    if not isinstance(instance, Manifold):
        return False
    else:
        # this is the case to care about, Scaled class is a proxy, but fails instance checks
        while isinstance(instance, Scaled):
            instance = instance.base
        return isinstance(instance, cls)


def canonical_manifold(manifold: "Manifold"):
    """
    Get a canonical manifold.

    If a manifold is wrapped with Scaled. Some attributes may not be available. This should help if you really need them.

    Parameters
    ----------
    manifold : geoopt.Manifold

    Returns
    -------
    geoopt.Maniflold
        an unwrapped manifold
    """
    while isinstance(manifold, Scaled):
        manifold = manifold.base
    return manifold


def insert_docs(doc, pattern=None, repl=None):
    def wrapper(fn):
        # assume wrapping
        if pattern is not None:
            if repl is None:
                raise RuntimeError("need repl parameter")
            fn.__doc__ = re.sub(pattern, repl, doc)
        else:
            fn.__doc__ = doc
        return fn

    return wrapper

class ScalingInfo(object):
    """
    Scaling info for each argument that requires rescaling.

    .. code:: python

        scaled_value = value * scaling ** power if power != 0 else value

    For results it is not always required to set powers of scaling, then it is no-op.

    The convention for this info is the following. The output of a function is either a tuple or a single object.
    In any case, outputs are treated as positionals. Function inputs, in contrast, are treated by keywords.
    It is a common practice to maintain function signature when overriding, so this way may be considered
    as a sufficient in this particular scenario. The only required info for formula above is ``power``.
    """

    # marks method to be not working with Scaled wrapper
    NotCompatible = object()
    __slots__ = ["kwargs", "results"]

    def __init__(self, *results: float, **kwargs: float):
        self.results = results
        self.kwargs = kwargs


class ScalingStorage(dict):
    """
    Helper class to make implementation transparent.

    This is just a dictionary with additional overriden ``__call__``
    for more explicit and elegant API to declare members. A usage example may be found in :class:`Manifold`.

    Methods that require rescaling when wrapped into :class:`Scaled` should be defined as follows

    1. Regular methods like ``dist``, ``dist2``, ``expmap``, ``retr`` etc. that are already present in the base class
    do not require registration, it has already happened in the base :class:`Manifold` class.

    2. New methods (like in :class:`PoincareBall`) should be treated with care.

    .. code-block:: python

        class PoincareBall(Manifold):
            # make a class copy of __scaling__ info. Default methods are already present there
            __scaling__ = Manifold.__scaling__.copy()
            ... # here come regular implementation of the required methods

            @__scaling__(ScalingInfo(1))  # rescale output according to rule `out * scaling ** 1`
            def dist0(self, x: torch.Tensor, *, dim=-1, keepdim=False):
                return math.dist0(x, c=self.c, dim=dim, keepdim=keepdim)

            @__scaling__(ScalingInfo(u=-1))  # rescale argument `u` according to the rule `out * scaling ** -1`
            def expmap0(self, u: torch.Tensor, *, dim=-1, project=True):
                res = math.expmap0(u, c=self.c, dim=dim)
                if project:
                    return math.project(res, c=self.c, dim=dim)
                else:
                    return res
            ... # other special methods implementation

    3. Some methods are not compliant with the above rescaling rules. We should mark them as `NotCompatible`

    .. code-block:: python

            # continuation of the PoincareBall definition
            @__scaling__(ScalingInfo.NotCompatible)
            def mobius_fn_apply(
                self, fn: callable, x: torch.Tensor, *args, dim=-1, project=True, **kwargs
            ):
                res = math.mobius_fn_apply(fn, x, *args, c=self.c, dim=dim, **kwargs)
                if project:
                    return math.project(res, c=self.c, dim=dim)
                else:
                    return res
    """

    def __call__(self, scaling_info: ScalingInfo, *aliases):
        def register(fn):
            self[fn.__name__] = scaling_info
            for alias in aliases:
                self[alias] = scaling_info
            return fn

        return register

    def copy(self):
        return self.__class__(self)


class Manifold(torch.nn.Module, metaclass=abc.ABCMeta):
    __scaling__ = ScalingStorage()  # will be filled along with implementation below
    name = None
    ndim = None
    reversible = None

    forward = NotImplemented

    def __init__(self, **kwargs):
        super().__init__()

    @property
    def device(self) -> Optional[torch.device]:
        """
        Manifold device.

        Returns
        -------
        Optional[torch.device]
        """
        p = next(itertools.chain(self.buffers(), self.parameters()), None)
        if p is not None:
            return p.device
        else:
            return None

    @property
    def dtype(self) -> Optional[torch.dtype]:
        """
        Manifold dtype.

        Returns
        -------
        Optional[torch.dtype]
        """

        p = next(itertools.chain(self.buffers(), self.parameters()), None)
        if p is not None:
            return p.dtype
        else:
            return None

    def check_point(
        self, x: torch.Tensor, *, explain=False
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        """
        Check if point is valid to be used with the manifold.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        explain: bool
            return an additional information on check

        Returns
        -------
        bool
            boolean indicating if tensor is valid and reason of failure if False

        Notes
        -----
        This check is compatible to what optimizer expects, last dimensions are treated as manifold dimensions
        """
        ok, reason = self._check_shape(x.shape, "x")
        if explain:
            return ok, reason
        else:
            return ok

    def assert_check_point(self, x: torch.Tensor):
        """
        Check if point is valid to be used with the manifold and raise an error with informative message on failure.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold

        Notes
        -----
        This check is compatible to what optimizer expects, last dimensions are treated as manifold dimensions
        """

        ok, reason = self._check_shape(x.shape, "x")
        if not ok:
            raise ValueError(
                "`x` seems to be not valid "
                "tensor for {} manifold.\nerror: {}".format(self.name, reason)
            )

    def check_vector(self, u: torch.Tensor, *, explain=False):
        """
        Check if vector is valid to be used with the manifold.

        Parameters
        ----------
        u : torch.Tensor
            vector on the tangent plane
        explain: bool
            return an additional information on check

        Returns
        -------
        bool
            boolean indicating if tensor is valid and reason of failure if False

        Notes
        -----
        This check is compatible to what optimizer expects, last dimensions are treated as manifold dimensions
        """
        ok, reason = self._check_shape(u.shape, "u")
        if explain:
            return ok, reason
        else:
            return ok

    def assert_check_vector(self, u: torch.Tensor):
        """
        Check if vector is valid to be used with the manifold and raise an error with informative message on failure.

        Parameters
        ----------
        u : torch.Tensor
            vector on the tangent plane

        Notes
        -----
        This check is compatible to what optimizer expects, last dimensions are treated as manifold dimensions
        """

        ok, reason = self._check_shape(u.shape, "u")
        if not ok:
            raise ValueError(
                "`u` seems to be not valid "
                "tensor for {} manifold.\nerror: {}".format(self.name, reason)
            )

    def check_point_on_manifold(
        self, x: torch.Tensor, *, explain=False, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        """
        Check if point :math:`x` is lying on the manifold.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        atol: float
            absolute tolerance as in :func:`numpy.allclose`
        rtol: float
            relative tolerance as in :func:`numpy.allclose`
        explain: bool
            return an additional information on check

        Returns
        -------
        bool
            boolean indicating if tensor is valid and reason of failure if False

        Notes
        -----
        This check is compatible to what optimizer expects, last dimensions are treated as manifold dimensions
        """
        ok, reason = self._check_shape(x.shape, "x")
        if ok:
            ok, reason = self._check_point_on_manifold(x, atol=atol, rtol=rtol)
        if explain:
            return ok, reason
        else:
            return ok

    def assert_check_point_on_manifold(self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5):
        """
        Check if point :math`x` is lying on the manifold and raise an error with informative message on failure.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        atol: float
            absolute tolerance as in :func:`numpy.allclose`
        rtol: float
            relative tolerance as in :func:`numpy.allclose`
        """
        self.assert_check_point(x)
        ok, reason = self._check_point_on_manifold(x, atol=atol, rtol=rtol)
        if not ok:
            raise ValueError(
                "`x` seems to be a tensor "
                "not lying on {} manifold.\nerror: {}".format(self.name, reason)
            )

    def check_vector_on_tangent(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        *,
        ok_point=False,
        explain=False,
        atol=1e-5,
        rtol=1e-5,
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        """
        Check if :math:`u` is lying on the tangent space to x.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        u : torch.Tensor
            vector on the tangent space to :math:`x`
        atol: float
            absolute tolerance as in :func:`numpy.allclose`
        rtol: float
            relative tolerance as in :func:`numpy.allclose`
        explain: bool
            return an additional information on check
        ok_point: bool
            is a check for point required?

        Returns
        -------
        bool
            boolean indicating if tensor is valid and reason of failure if False
        """
        if not ok_point:
            ok, reason = self._check_shape(x.shape, "x")
            if ok:
                ok, reason = self._check_shape(u.shape, "u")
            if ok:
                ok, reason = self._check_point_on_manifold(x, atol=atol, rtol=rtol)
        else:
            ok = True
            reason = None
        if ok:
            ok, reason = self._check_vector_on_tangent(x, u, atol=atol, rtol=rtol)
        if explain:
            return ok, reason
        else:
            return ok

    def assert_check_vector_on_tangent(
        self, x: torch.Tensor, u: torch.Tensor, *, ok_point=False, atol=1e-5, rtol=1e-5
    ):
        """
        Check if u :math:`u` is lying on the tangent space to x and raise an error on fail.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        u : torch.Tensor
            vector on the tangent space to :math:`x`
        atol: float
            absolute tolerance as in :func:`numpy.allclose`
        rtol: float
            relative tolerance as in :func:`numpy.allclose`
        ok_point: bool
            is a check for point required?
        """
        if not ok_point:
            ok, reason = self._check_shape(x.shape, "x")
            if ok:
                ok, reason = self._check_shape(u.shape, "u")
            if ok:
                ok, reason = self._check_point_on_manifold(x, atol=atol, rtol=rtol)
        else:
            ok = True
            reason = None
        if ok:
            ok, reason = self._check_vector_on_tangent(x, u, atol=atol, rtol=rtol)
        if not ok:
            raise ValueError(
                "`u` seems to be a tensor "
                "not lying on tangent space to `x` for {} manifold.\nerror: {}".format(
                    self.name, reason
                )
            )

    @__scaling__(ScalingInfo(1))
    def dist(self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False) -> torch.Tensor:
        """
        Compute distance between 2 points on the manifold that is the shortest path along geodesics.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        y : torch.Tensor
            point on the manifold
        keepdim : bool
            keep the last dim?

        Returns
        -------
        torch.Tensor
            distance between two points
        """
        raise NotImplementedError

    @__scaling__(ScalingInfo(2))
    def dist2(self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False) -> torch.Tensor:
        """
        Compute squared distance between 2 points on the manifold that is the shortest path along geodesics.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        y : torch.Tensor
            point on the manifold
        keepdim : bool
            keep the last dim?

        Returns
        -------
        torch.Tensor
            squared distance between two points
        """
        return self.dist(x, y, keepdim=keepdim) ** 2

    @abc.abstractmethod
    @__scaling__(ScalingInfo(u=-1))
    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Perform a retraction from point :math:`x` with given direction :math:`u`.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        u : torch.Tensor
            tangent vector at point :math:`x`

        Returns
        -------
        torch.Tensor
            transported point
        """
        raise NotImplementedError

    @abc.abstractmethod
    @__scaling__(ScalingInfo(u=-1))
    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        r"""
        Perform an exponential map :math:`\operatorname{Exp}_x(u)`.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        u : torch.Tensor
            tangent vector at point :math:`x`

        Returns
        -------
        torch.Tensor
            transported point
        """
        raise NotImplementedError

    @__scaling__(ScalingInfo(1))
    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""
        Perform an logarithmic map :math:`\operatorname{Log}_{x}(y)`.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        y : torch.Tensor
            point on the manifold

        Returns
        -------
        torch.Tensor
            tangent vector
        """
        raise NotImplementedError

    @__scaling__(ScalingInfo(u=-1))
    def expmap_transp(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform an exponential map and vector transport from point :math:`x` with given direction :math:`u`.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        u : torch.Tensor
            tangent vector at point :math:`x`
        v : torch.Tensor
            tangent vector at point :math:`x` to be transported

        Returns
        -------
        torch.Tensor
            transported point
        """
        y = self.expmap(x, u)
        v_transp = self.transp(x, y, v)
        return y, v_transp

    @__scaling__(ScalingInfo(u=-1))
    def retr_transp(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a retraction + vector transport at once.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        u : torch.Tensor
            tangent vector at point :math:`x`
        v : torch.Tensor
            tangent vector at point :math:`x` to be transported

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            transported point and vectors

        Notes
        -----
        Sometimes this is a far more optimal way to preform retraction + vector transport
        """
        y = self.retr(x, u)
        v_transp = self.transp(x, y, v)
        return y, v_transp

    @__scaling__(ScalingInfo(u=-1))
    def transp_follow_retr(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Perform vector transport following :math:`u`: :math:`\mathfrak{T}_{x\to\operatorname{retr}(x, u)}(v)`.

        This operation is sometimes is much more simpler and can be optimized.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        u : torch.Tensor
            tangent vector at point :math:`x`
        v : torch.Tensor
            tangent vector at point :math:`x` to be transported

        Returns
        -------
        torch.Tensor
            transported tensor
        """
        y = self.retr(x, u)
        return self.transp(x, y, v)

    @__scaling__(ScalingInfo(u=-1))
    def transp_follow_expmap(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Perform vector transport following :math:`u`: :math:`\mathfrak{T}_{x\to\operatorname{Exp}(x, u)}(v)`.

        Here, :math:`\operatorname{Exp}` is the best possible approximation of the true exponential map.
        There are cases when the exact variant is hard or impossible implement, therefore a
        fallback, non-exact, implementation is used.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        u : torch.Tensor
            tangent vector at point :math:`x`
        v : torch.Tensor
            tangent vector at point :math:`x` to be transported

        Returns
        -------
        torch.Tensor
            transported tensor
        """
        y = self.expmap(x, u)
        return self.transp(x, y, v)

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        r"""
        Perform vector transport :math:`\mathfrak{T}_{x\to y}(v)`.

        Parameters
        ----------
        x : torch.Tensor
            start point on the manifold
        y : torch.Tensor
            target point on the manifold
        v : torch.Tensor
            tangent vector at point :math:`x`

        Returns
        -------
        torch.Tensor
           transported tensor
        """
        raise NotImplementedError

    @abc.abstractmethod
    def inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None, *, keepdim=False
    ) -> torch.Tensor:
        """
        Inner product for tangent vectors at point :math:`x`.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        u : torch.Tensor
            tangent vector at point :math:`x`
        v : Optional[torch.Tensor]
            tangent vector at point :math:`x`
        keepdim : bool
            keep the last dim?

        Returns
        -------
        torch.Tensor
            inner product (broadcasted)
        """
        raise NotImplementedError

    def component_inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Inner product for tangent vectors at point :math:`x` according to components of the manifold.

        The result of the function is same as ``inner`` with ``keepdim=True`` for
        all the manifolds except ProductManifold. For this manifold it acts different way
        computing inner product for each component and then building an output correctly
        tiling and reshaping the result.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        u : torch.Tensor
            tangent vector at point :math:`x`
        v : Optional[torch.Tensor]
            tangent vector at point :math:`x`

        Returns
        -------
        torch.Tensor
            inner product component wise (broadcasted)

        Notes
        -----
        The purpose of this method is better adaptive properties in optimization since ProductManifold
        will "hide" the structure in public API.
        """
        return self.inner(x, u, v, keepdim=True)

    def norm(self, x: torch.Tensor, u: torch.Tensor, *, keepdim=False) -> torch.Tensor:
        """
        Norm of a tangent vector at point :math:`x`.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        u : torch.Tensor
            tangent vector at point :math:`x`
        keepdim : bool
            keep the last dim?

        Returns
        -------
        torch.Tensor
            inner product (broadcasted)
        """
        return self.inner(x, u, keepdim=keepdim) ** 0.5

    @abc.abstractmethod
    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Project vector :math:`u` on a tangent space for :math:`x`, usually is the same as :meth:`egrad2rgrad`.

        Parameters
        ----------
        x torch.Tensor
            point on the manifold
        u torch.Tensor
            vector to be projected

        Returns
        -------
        torch.Tensor
            projected vector
        """
        raise NotImplementedError

    @abc.abstractmethod
    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Transform gradient computed using autodiff to the correct Riemannian gradient for the point :math:`x`.

        Parameters
        ----------
        x torch.Tensor
            point on the manifold
        u torch.Tensor
            gradient to be projected

        Returns
        -------
        torch.Tensor
            grad vector in the Riemannian manifold
        """
        raise NotImplementedError

    @abc.abstractmethod
    def projx(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project point :math:`x` on the manifold.

        Parameters
        ----------
        x torch.Tensor
            point to be projected

        Returns
        -------
        torch.Tensor
            projected point
        """
        raise NotImplementedError

    def _check_shape(
        self, shape: Tuple[int], name: str
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        """
        Util to check shape.

        Exhaustive implementation for checking if
        a given point has valid dimension size,
        shape, etc. It should return boolean and
        a reason of failure if check is not passed

        Parameters
        ----------
        shape : Tuple[int]
            shape of point on the manifold
        name : str
            name to be present in errors

        Returns
        -------
        bool, str or None
            check result and the reason of fail if any
        """
        ok = len(shape) >= self.ndim
        if not ok:
            reason = "'{}' on the {} requires more than {} dim".format(
                name, self, self.ndim
            )
        else:
            reason = None
        return ok, reason

    def _assert_check_shape(self, shape: Tuple[int], name: str):
        """
        Util to check shape and raise an error if needed.

        Exhaustive implementation for checking if
        a given point has valid dimension size,
        shape, etc. It will raise a ValueError if check is not passed

        Parameters
        ----------
        shape : tuple
            shape of point on the manifold
        name : str
            name to be present in errors

        Raises
        ------
        ValueError
        """
        ok, reason = self._check_shape(shape, name)
        if not ok:
            raise ValueError(reason)

    @abc.abstractmethod
    def _check_point_on_manifold(
        self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        """
        Util to check point lies on the manifold.

        Exhaustive implementation for checking if
        a given point lies on the manifold. It
        should return boolean and a reason of
        failure if check is not passed. You can
        assume assert_check_point is already
        passed beforehand

        Parameters
        ----------
        x torch.Tensor
            point on the manifold
        atol: float
            absolute tolerance as in :func:`numpy.allclose`
        rtol: float
            relative tolerance as in :func:`numpy.allclose`

        Returns
        -------
        bool, str or None
            check result and the reason of fail if any
        """
        # return True, None
        raise NotImplementedError

    @abc.abstractmethod
    def _check_vector_on_tangent(
        self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        """
        Util to check a vector belongs to the tangent space of a point.

        Exhaustive implementation for checking if
        a given point lies in the tangent space at x
        of the manifold. It should return a boolean
        indicating whether the test was passed
        and a reason of failure if check is not passed.
        You can assume assert_check_point is already
        passed beforehand

        Parameters
        ----------
        x torch.Tensor
        u torch.Tensor
        atol : float
            absolute tolerance
        rtol :
            relative tolerance

        Returns
        -------
        bool, str or None
            check result and the reason of fail if any
        """
        # return True, None
        raise NotImplementedError

    def extra_repr(self):
        return ""

    def __repr__(self):
        extra = self.extra_repr()
        if extra:
            return self.name + "({}) manifold".format(extra)
        else:
            return self.name + " manifold"

    def unpack_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Construct a point on the manifold.

        This method should help to work with product and compound manifolds.
        Internally all points on the manifold are stored in an intuitive format.
        However, there might be cases, when this representation is simpler or more efficient to store in
        a different way that is hard to use in practice.

        Parameters
        ----------
        tensor : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        return tensor

    def pack_point(self, *tensors: torch.Tensor) -> torch.Tensor:
        """
        Construct a tensor representation of a manifold point.

        In case of regular manifolds this will return the same tensor. However, for e.g. Product manifold
        this function will pack all non-batch dimensions.

        Parameters
        ----------
        tensors : Tuple[torch.Tensor]

        Returns
        -------
        torch.Tensor
        """
        if len(tensors) != 1:
            raise ValueError("1 tensor expected, got {}".format(len(tensors)))
        return tensors[0]

    def random(self, *size, dtype=None, device=None, **kwargs) -> torch.Tensor:
        """
        Random sampling on the manifold.

        The exact implementation depends on manifold and usually does not follow all
        assumptions about uniform measure, etc.
        """
        raise NotImplementedError

    def origin(
        self,
        *size: Union[int, Tuple[int]],
        dtype=None,
        device=None,
        seed: Optional[int] = 42,
    ) -> torch.Tensor:
        """
        Create some reasonable point on the manifold in a deterministic way.

        For some manifolds there may exist e.g. zero vector or some analogy.
        In case it is possible to define this special point, this point is returned with the desired size.
        In other case, the returned point is sampled on the manifold in a deterministic way.

        Parameters
        ----------
        size : Union[int, Tuple[int]]
            the desired shape
        device : torch.device
            the desired device
        dtype : torch.dtype
            the desired dtype
        seed : Optional[int]
            A parameter controlling deterministic randomness for manifolds that do not provide ``.origin``,
            but provide ``.random``. (default: 42)

        Returns
        -------
        torch.Tensor
        """
        if seed is not None:
            # we promise pseudorandom behaviour but do not want to modify global seed
            state = torch.random.get_rng_state()
            torch.random.manual_seed(seed)
            try:
                return self.random(*size, dtype=dtype, device=device)
            finally:
                torch.random.set_rng_state(state)
        else:
            return self.random(*size, dtype=dtype, device=device)

class Euclidean(Manifold):
    """
    Simple Euclidean manifold, every coordinate is treated as an independent element.

    Parameters
    ----------
    ndim : int
        number of trailing dimensions treated as manifold dimensions. All the operations acting on cuch
        as inner products, etc will respect the :attr:`ndim`.
    """

    __scaling__ = Manifold.__scaling__.copy()
    name = "Euclidean"
    ndim = 0
    reversible = True

    def __init__(self, ndim=0):
        super().__init__()
        self.ndim = ndim

    def _check_point_on_manifold(
        self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        return True, None

    def _check_vector_on_tangent(
        self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        return True, None

    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return x + u

    def inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None, *, keepdim=False
    ) -> torch.Tensor:
        if v is None:
            inner = u.pow(2)
        else:
            inner = u * v
        if self.ndim > 0:
            inner = inner.sum(dim=tuple(range(-self.ndim, 0)), keepdim=keepdim)
            x_shape = x.shape[: -self.ndim] + (1,) * self.ndim * keepdim
        else:
            x_shape = x.shape
        i_shape = inner.shape
        target_shape = broadcast_shapes(x_shape, i_shape)
        return inner.expand(target_shape)

    def component_inner(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None
    ) -> torch.Tensor:
        # it is possible to factorize the manifold
        if v is None:
            inner = u.pow(2)
        else:
            inner = u * v
        target_shape = broadcast_shapes(x.shape, inner.shape)
        return inner.expand(target_shape)

    def norm(self, x: torch.Tensor, u: torch.Tensor, *, keepdim=False):
        if self.ndim > 0:
            return u.norm(dim=tuple(range(-self.ndim, 0)), keepdim=keepdim)
        else:
            return u.abs()

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        target_shape = broadcast_shapes(x.shape, u.shape)
        return u.expand(target_shape)

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return y - x

    def dist(self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False) -> torch.Tensor:
        if self.ndim > 0:
            return (x - y).norm(dim=tuple(range(-self.ndim, 0)), keepdim=keepdim)
        else:
            return (x - y).abs()

    def dist2(self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False) -> torch.Tensor:
        if self.ndim > 0:
            return (x - y).pow(2).sum(dim=tuple(range(-self.ndim, 0)), keepdim=keepdim)
        else:
            return (x - y).pow(2)

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        target_shape = broadcast_shapes(x.shape, u.shape)
        return u.expand(target_shape)

    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return x + u

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        target_shape = broadcast_shapes(x.shape, y.shape, v.shape)
        return v.expand(target_shape)

    @__scaling__(ScalingInfo(std=-1), "random")
    def random_normal(
        self, *size, mean=0.0, std=1.0, device=None, dtype=None
    ) -> "ManifoldTensor":
        """
        Create a point on the manifold, measure is induced by Normal distribution.

        Parameters
        ----------
        size : shape
            the desired shape
        mean : float|tensor
            mean value for the Normal distribution
        std : float|tensor
            std value for the Normal distribution
        device : torch.device
            the desired device
        dtype : torch.dtype
            the desired dtype

        Returns
        -------
        ManifoldTensor
            random point on the manifold
        """
        self._assert_check_shape(size2shape(*size), "x")
        mean = torch.as_tensor(mean, device=device, dtype=dtype)
        std = torch.as_tensor(std, device=device, dtype=dtype)
        tens = std.new_empty(*size).normal_() * std + mean
        return ManifoldTensor(tens, manifold=self)

    random = random_normal

    def origin(
        self, *size, dtype=None, device=None, seed=42
    ) -> "ManifoldTensor":
        """
        Zero point origin.

        Parameters
        ----------
        size : shape
            the desired shape
        device : torch.device
            the desired device
        dtype : torch.dtype
            the desired dtype
        seed : int
            ignored

        Returns
        -------
        ManifoldTensor
        """
        self._assert_check_shape(size2shape(*size), "x")
        return ManifoldTensor(
            torch.zeros(*size, dtype=dtype, device=device), manifold=self
        )

    def extra_repr(self):
        return "ndim={}".format(self.ndim)

class ManifoldTensor(torch.Tensor):
    """Same as :class:`torch.Tensor` that has information about its manifold.

    Other Parameters
    ----------------
    manifold : :class:`geoopt.Manifold`
        A manifold for the tensor, (default: :class:`geoopt.Euclidean`)
    """

    try:
        # https://github.com/pytorch/pytorch/issues/46159#issuecomment-707817037
        from torch._C import _disabled_torch_function_impl  # noqa

        __torch_function__ = _disabled_torch_function_impl

    except ImportError:
        pass

    def __new__(
        cls, *args, manifold: Manifold = Euclidean(), requires_grad=False, **kwargs
    ):
        if len(args) == 1 and isinstance(args[0], torch.Tensor):
            data = args[0].data
        else:
            data = torch.Tensor(*args, **kwargs)
        if kwargs.get("device") is not None:
            data.data = data.data.to(kwargs.get("device"))
        with torch.no_grad():
            manifold.assert_check_point(data)
        instance = torch.Tensor._make_subclass(cls, data, requires_grad)
        instance.manifold = manifold
        return instance

    manifold: Manifold

    def proj_(self) -> torch.Tensor:
        """
        Inplace projection to the manifold.

        Returns
        -------
        tensor
            same instance
        """
        return self.copy_(self.manifold.projx(self))

    @insert_docs(Manifold.retr.__doc__, r"\s+x : .+\n.+", "")
    def retr(self, u: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.manifold.retr(self, u=u, **kwargs)

    @insert_docs(Manifold.expmap.__doc__, r"\s+x : .+\n.+", "")
    def expmap(self, u: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.manifold.expmap(self, u=u, **kwargs)

    @insert_docs(Manifold.inner.__doc__, r"\s+x : .+\n.+", "")
    def inner(self, u: torch.Tensor, v: torch.Tensor = None, **kwargs) -> torch.Tensor:
        return self.manifold.inner(self, u=u, v=v, **kwargs)

    @insert_docs(Manifold.proju.__doc__, r"\s+x : .+\n.+", "")
    def proju(self, u: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.manifold.proju(self, u, **kwargs)

    @insert_docs(Manifold.transp.__doc__, r"\s+x : .+\n.+", "")
    def transp(self, y: torch.Tensor, v: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.manifold.transp(self, y, v, **kwargs)

    @insert_docs(Manifold.retr_transp.__doc__, r"\s+x : .+\n.+", "")
    def retr_transp(
        self, u: torch.Tensor, v: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.manifold.retr_transp(self, u, v, **kwargs)

    @insert_docs(Manifold.expmap_transp.__doc__, r"\s+x : .+\n.+", "")
    def expmap_transp(self, u: torch.Tensor, v: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.manifold.expmap_transp(self, u, v, **kwargs)

    @insert_docs(Manifold.transp_follow_expmap.__doc__, r"\s+x : .+\n.+", "")
    def transp_follow_expmap(
        self, u: torch.Tensor, v: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        return self.manifold.transp_follow_expmap(self, u, v, **kwargs)

    @insert_docs(Manifold.transp_follow_retr.__doc__, r"\s+x : .+\n.+", "")
    def transp_follow_retr(
        self, u: torch.Tensor, v: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        return self.manifold.transp_follow_retr(self, u, v, **kwargs)

    def dist(
        self, other: torch.Tensor, p: Union[int, float, bool, str] = 2, **kwargs
    ) -> torch.Tensor:
        """
        Return euclidean  or geodesic distance between points on the manifold. Allows broadcasting.

        Parameters
        ----------
        other : tensor
        p : str|int
            The norm to use. The default behaviour is not changed and is just euclidean distance.
            To compute geodesic distance, :attr:`p` should be set to ``"g"``

        Returns
        -------
        scalar
        """
        if p == "g":
            return self.manifold.dist(self, other, **kwargs)
        else:
            return super().dist(other)

    @insert_docs(Manifold.logmap.__doc__, r"\s+x : .+\n.+", "")
    def logmap(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.manifold.logmap(self, y, **kwargs)

    def __repr__(self):
        return "Tensor on {} containing:\n".format(
            self.manifold
        ) + torch.Tensor.__repr__(self)

    # noinspection PyUnresolvedReferences
    def __reduce_ex__(self, proto):
        build, proto = super(ManifoldTensor, self).__reduce_ex__(proto)
        new_build = functools.partial(_rebuild_manifold_tensor, build_fn=build)
        new_proto = proto + (dict(), self.__class__, self.manifold, self.requires_grad)
        return new_build, new_proto

    @insert_docs(Manifold.unpack_tensor.__doc__, r"\s+tensor : .+\n.+", "")
    def unpack_tensor(self) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        return self.manifold.unpack_tensor(self)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(
                self.data.clone(memory_format=torch.preserve_format),
                manifold=copy.deepcopy(self.manifold, memo=memo),
                requires_grad=self.requires_grad,
            )
            memo[id(self)] = result
            return result


class ManifoldParameter(ManifoldTensor, torch.nn.Parameter):
    """Same as :class:`torch.nn.Parameter` that has information about its manifold.

    It should be used within :class:`torch.nn.Module` to be recognized
    in parameter collection.

    Other Parameters
    ----------------
    manifold : :class:`geoopt.Manifold` (optional)
        A manifold for the tensor if ``data`` is not a :class:`geoopt.ManifoldTensor`
    """

    def __new__(cls, data=None, manifold=None, requires_grad=True):
        if data is None:
            data = ManifoldTensor(manifold=manifold or Euclidean())
        elif not isinstance(data, ManifoldTensor):
            data = ManifoldTensor(data, manifold=manifold or Euclidean())
        else:
            if manifold is not None and data.manifold != manifold:
                raise ValueError(
                    "Manifolds do not match: {}, {}".format(data.manifold, manifold)
                )
        instance = ManifoldTensor._make_subclass(cls, data, requires_grad)
        instance.manifold = data.manifold
        return instance

    def __repr__(self):
        return "Parameter on {} containing:\n".format(
            self.manifold
        ) + torch.Tensor.__repr__(self)


def _rebuild_manifold_tensor(*args, build_fn):
    tensor = build_fn(*args[:-4])
    return args[-3](tensor, manifold=args[-2], requires_grad=args[-1])



class SymmetricPositiveDefinite(Manifold):
    """
    Subclass of the SymmetricPositiveDefinite manifold using the 
    affine invariant Riemannian metric (AIRM) as default metric
    """

    __scaling__ = Manifold.__scaling__.copy()
    name = "SymmetricPositiveDefinite"
    ndim = 2
    reversible = False

    def __init__(self):
        super().__init__()

    def dist(self, x: torch.Tensor, y: torch.Tensor, keepdim) -> torch.Tensor:
        """
        Computes the affine invariant Riemannian metric (AIM)
        """
        inv_sqrt_x = functional.sym_invsqrtm.apply(x)
        return torch.norm(
            functional.sym_logm.apply(inv_sqrt_x @ y @ inv_sqrt_x),
            dim=[-1, -2],
            keepdim=keepdim,
        )

    def _check_point_on_manifold(
        self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        ok = torch.allclose(x, x.transpose(-1, -2), atol=atol, rtol=rtol)
        if not ok:
            return False, "`x != x.transpose` with atol={}, rtol={}".format(atol, rtol)
        e = torch.linalg.eigvalsh(x)
        ok = (e > -atol).min()
        if not ok:
            return False, "eigenvalues of x are not all greater than 0."
        return True, None

    def _check_vector_on_tangent(
        self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        ok = torch.allclose(u, u.transpose(-1, -2), atol=atol, rtol=rtol)
        if not ok:
            return False, "`u != u.transpose` with atol={}, rtol={}".format(atol, rtol)
        return True, None

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        symx = functional.ensure_sym(x)
        return functional.sym_abseig.apply(symx)

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return functional.ensure_sym(u)

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return x @ self.proju(x, u) @ x

    def inner(self, x: torch.Tensor, u: torch.Tensor, v: Optional[torch.Tensor], keepdim) -> torch.Tensor:
        if v is None:
            v = u
        inv_x = functional.sym_invm.apply(x)
        ret = torch.diagonal(inv_x @ u @ inv_x @ v, dim1=-2, dim2=-1).sum(-1)
        if keepdim:
            return torch.unsqueeze(torch.unsqueeze(ret, -1), -1)
        return ret

    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        inv_x = functional.sym_invm.apply(x)
        return functional.ensure_sym(x + u + 0.5 * u @ inv_x @ u)
        # return self.expmap(x, u)

    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        sqrt_x, inv_sqrt_x = functional.sym_invsqrtm2.apply(x)
        return sqrt_x @ functional.sym_expm.apply(inv_sqrt_x @ u @ inv_sqrt_x) @ sqrt_x

    def logmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        sqrt_x, inv_sqrt_x = functional.sym_invsqrtm2.apply(x)
        return sqrt_x @ functional.sym_logm.apply(inv_sqrt_x @ u @ inv_sqrt_x) @ sqrt_x

    def extra_repr(self) -> str:
        return "default_metric=AIM"

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:

        xinvy = torch.linalg.solve(x.double(),y.double())
        s, U = torch.linalg.eig(xinvy.transpose(-2,-1))
        s = s.real
        U = U.real

        Ut = U.transpose(-2,-1)
        Esqm = torch.linalg.solve(Ut, torch.diag_embed(s.sqrt()) @ Ut).transpose(-2,-1).to(y.dtype)

        return Esqm @ v @ Esqm.transpose(-1,-2)

    def random(self, *size, dtype=None, device=None, **kwargs) -> torch.Tensor:
        tens = torch.randn(*size, dtype=dtype, device=device, **kwargs)
        tens = functional.ensure_sym(tens)
        tens = functional.sym_expm.apply(tens)
        return tens

    def barycenter(self, X : torch.Tensor, steps : int = 1, dim = 0) -> torch.Tensor:
        """
        Compute several steps of the Kracher flow algorithm to estimate the 
        Barycenter on the manifold.
        """
        return functional.spd_mean_kracher_flow(X, None, maxiter=steps, dim=dim, return_dist=False)

    def geodesic(self, A : torch.Tensor, B : torch.Tensor, t : torch.Tensor) -> torch.Tensor:
        """
        Compute geodesic between two SPD tensors A and B and return 
        point on the geodesic at length t \in [0,1]
        if t = 0, then A is returned
        if t = 1, then B is returned
        """
        Asq, Ainvsq = functional.sym_invsqrtm2.apply(A)
        return Asq @ functional.sym_powm.apply(Ainvsq @ B @ Ainvsq, t) @ Asq

    def transp_via_identity(self, X : torch.Tensor, A : torch.Tensor, B : torch.Tensor) -> torch.Tensor:
        """
        Parallel transport of the tensors in X around A to the identity matrix I
        Parallel transport from around the identity matrix to the new center (tensor B)
        """
        Ainvsq = functional.sym_invsqrtm.apply(A)
        Bsq = functional.sym_sqrtm.apply(B)
        return Bsq @ (Ainvsq @ X @ Ainvsq) @ Bsq

    def transp_identity_rescale_transp(self, X : torch.Tensor, A : torch.Tensor, s : torch.Tensor, B : torch.Tensor) -> torch.Tensor:
        """
        Parallel transport of the tensors in X around A to the identity matrix I
        Rescales the dispersion by the factor s
        Parallel transport from the identity to the new center (tensor B)
        """
        Ainvsq = functional.sym_invsqrtm.apply(A)
        Bsq = functional.sym_sqrtm.apply(B)
        return Bsq @ functional.sym_powm.apply(Ainvsq @ X @ Ainvsq, s) @ Bsq

    def transp_identity_rescale_rotate_transp(self, X : torch.Tensor, A : torch.Tensor, s : torch.Tensor, B : torch.Tensor, W : torch.Tensor) -> torch.Tensor:
        """
        Parallel transport of the tensors in X around A to the identity matrix I
        Rescales the dispersion by the factor s
        Parallel transport from the identity to the new center (tensor B)
        """
        Ainvsq = functional.sym_invsqrtm.apply(A)
        Bsq = functional.sym_sqrtm.apply(B)
        WBsq = W @ Bsq
        return WBsq.transpose(-2,-1) @ functional.sym_powm.apply(Ainvsq @ X @ Ainvsq, s) @ WBsq