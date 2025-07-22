import numpy as np
import xarray as xr

def sort_da(da, dim, kind=None, stable=None, **kwargs):
    """Sort along dimension using DataArray values.
    Copied from https://einstats.python.arviz.org/en/latest/_modules/xarray_einstats.html#sort to avoid the dependency on xarray_einstats.

    Wrapper around :func:`numpy.sort`

    The returned DataArray has the same shape and dimensions as the original one
    but the coordinate values along `dim` no longer make sense given each subset
    along the other dimensions can have a different order along `dim` so they are removed.

    Parameters
    ----------
    da : DataArray
        Input data
    dim : hashable
        Dimension along which to sort using dataarray values, not coordinates.
    kind : str, optional
    stable : bool, optional
    **kwargs
        Passed to :func:`xarray.apply_ufunc`

    Returns
    -------
    DataArray
    """
    sort_kwargs = {"axis": -1, "kind": kind, "stable": stable}
    return xr.apply_ufunc(
        np.sort,
        da,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        kwargs=sort_kwargs,
        **kwargs,
    ).drop_vars(dim, errors="ignore")