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

def interp_multi(ds, da_coord, interp_dim, interp_coord_value):
    #copied from MyPython.xarray
    """In xarray.interp or numpy.interp, you specify positions along a 1D coordinate and the values belonging to the neighboring points are interpolated. This function extends this functionality to multidimensional coordinates, where the coordinate values are not the same for all other dimensions.
    In principal, this could be achieved by looping 1D interpolation with apply_ufunc, but this implementation is more efficient.

    !!!Caution: It is assumed that da_coord is sorted along interp_dim!!!

    Implementation: Based on da_coord, search the left and right neighbor index for the given value. Then, look up the corresponding function values in ds and explicitly write out the linear interpolation.
    """
    if isinstance(da_coord, str):
        da_coord=ds[da_coord] # assume that da_coord is a multidimensional coordinate in ds

    ind_left=(da_coord<interp_coord_value).sum(interp_dim)-1 #find the index of the left neighbor
    ind_right=(ind_left+1).clip(max=ds.sizes[interp_dim]-1)
    c_left=da_coord.isel({interp_dim:ind_left})
    c_right=da_coord.isel({interp_dim:ind_right})
    val_left=ds.isel(it=ind_left)
    val_right=ds.isel(it=ind_right)
    interpolation=(val_right - val_left) / (c_right - c_left) * (interp_coord_value - c_left) + val_left

    if da_coord.name is not None: # set the selected value as coordinate in the output
        interpolation.coords[da_coord.name]=interp_coord_value
    return interpolation