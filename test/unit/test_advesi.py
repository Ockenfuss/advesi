#%%
import unittest as ut
import numpy as np
import numpy.testing as npt
import xarray as xr
import xarray.testing as xrt
import advesi as adv
#%%


class Test_Flowfield_Collection(ut.TestCase):
    def test_flatten_da(self):
        da=xr.DataArray(1.0)
        expected=xr.DataArray([1.0], coords=[('n', [0])])
        xrt.assert_equal(adv.flatten_da(da),expected)

        da=xr.DataArray([1.0], coords=[('t', [4])])
        expected=xr.DataArray([1.0], coords=[('n', [0])])
        xrt.assert_equal(adv.flatten_da(da),expected)
    
        da=xr.DataArray(np.arange(4).reshape(2,2), coords=[('t', [1,2]), ('x', [1,2])])
        expected=xr.DataArray([0,1,2,3], coords=[('n', [0,1,2,3])])
        xrt.assert_equal(adv.flatten_da(da),expected)

    def test_flowfield_collection(self):
        u=xr.DataArray(np.arange(10), coords=[(('x', np.arange(10)))])
        ff=adv.FlowField_Collection(u,0.0, 0.0)
        npt.assert_array_equal(ff.u.y, [-np.inf, np.inf]) #missing dimensions are added from -inf to inf
        npt.assert_array_equal(ff.v.y, [-np.inf, np.inf]) #missing dimensions are added from -inf to inf
        npt.assert_array_equal(ff.v.values, np.zeros((2,2,2))) #floats are accepted and converted to infinitely valid arrays

        w=xr.DataArray(np.arange(10), coords=[(('w', np.arange(10)))]) #we can add additional dimensions to the flowfield components
        ff=adv.FlowField_Collection(u,0.0,w)
        assert({'x', 'y', 'z', 'w'}==set(ff.w.dims))

    def test_path_collection(self):
        #%%
        x=xr.DataArray([[0,np.nan], [2,3]], coords=[('n', [0,1]), ('t', [0.1, 0.2])])
        y=xr.DataArray([[0,1], [np.nan,3]], coords=[('n', [0,1]), ('t', [0.1, 0.2])])
        z=x.copy()
        path_coll=adv.Path_Collection(x,y,z)
        expected=xr.DataArray([[0,np.nan], [np.nan,3]], coords=[('n', [0,1]), ('t', [0.1, 0.2])]) #nan in one coordinate leads to nans in other coordinates as well
        xrt.assert_equal(path_coll.x, expected)
        xrt.assert_equal(path_coll.y, expected)

