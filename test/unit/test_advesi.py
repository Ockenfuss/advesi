import pytest
import numpy as np
import numpy.testing as npt
import xarray as xr
import xarray.testing as xrt
import advesi as adv


class Test_Flowfield_Collection:
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
        # existing dimensions are extended
        npt.assert_array_equal(ff.ds.u.x, [-adv.FIELD_BOUNDARY] + list(np.arange(10)) + [adv.FIELD_BOUNDARY])
        #missing dimensions are added from -inf to inf
        npt.assert_array_equal(ff.ds.u.y, [-adv.FIELD_BOUNDARY, adv.FIELD_BOUNDARY])
        npt.assert_array_equal(ff.ds.v.y, [-adv.FIELD_BOUNDARY, adv.FIELD_BOUNDARY])
        npt.assert_array_equal(ff.ds.v.s, [1.0]) #for the selector, no extension is performed and a default value of 1.0 is used
        npt.assert_array_equal(ff.ds.v.values, np.zeros((1,2,2,2,12))) #floats are accepted and converted to infinitely valid arrays
        assert({'x', 'y', 'z', 't', 's'}==set(ff.ds.w.dims))

        w=xr.DataArray(np.arange(10), coords=[(('w0', np.arange(10)))]) #other dimensions than x,y,z,t,s are not allowed
        with pytest.raises(adv.DimensionError):
            ff=adv.FlowField_Collection(u,0.0,w)

        w=xr.DataArray(np.arange(10), coords=[(('w', np.arange(10)))]) #we can add additional dimensions to the flowfield components
        with pytest.raises(adv.DimensionError):
            ff=adv.FlowField_Collection(u,0.0,w)

    def test_path_collection(self):
        x=xr.DataArray([[0,np.nan], [2,3]], coords=[('n', [0,1]), ('it', [0.1, 0.2])])
        y=xr.DataArray([[0,1], [np.nan,3]], coords=[('n', [0,1]), ('it', [0.1, 0.2])])
        z=x.copy()
        t=xr.DataArray([[0.1, 0.2], [0.1, 0.2]], coords=[('n', [0,1]), ('t', [0.1, 0.2])])
        with pytest.raises(adv.DimensionError):
            ds=xr.Dataset({'x': x, 'y': y, 'z': z, 't': t})
            path_coll=adv.Path_Collection(ds) #Dimension 't' not allowed
        t=xr.DataArray([[0.1, 0.2], [0.1, 0.2]], coords=[('n', [0,1]), ('it', [0.1, 0.2])])
        ds=xr.Dataset({'x': x, 'y': y, 'z': z, 't': t})
        path_coll=adv.Path_Collection(ds) #Dimension 't' not allowed
        expected=xr.DataArray([[0,np.nan], [np.nan,3]], coords=[('n', [0,1]), ('it', [0.1, 0.2])]) #nan in one coordinate leads to nans in other coordinates as well
        xrt.assert_equal(path_coll.ds.x, expected)
        xrt.assert_equal(path_coll.ds.y, expected)

class Test_EulerAdvector:
    def test_advection(self):
        #steady 3 m/s flow
        u=xr.DataArray([3.0, 3.0], coords=[('z', [0,1.0])]).astype(float)
        ff=adv.FlowField_Collection(u, 0,0)
        advector=adv.EulerAdvector(dt=0.1, steps=10, steps_backward=10, savesteps=None)
        x0=xr.DataArray([0.0], coords=[('n', [0])])
        y0=xr.DataArray([0.0], coords=[('n', [0])])
        z0=xr.DataArray([0.0], coords=[('n', [0])])
        t0=xr.DataArray([0.0], coords=[('n', [0])])
        selector=adv.DataArraySelector(xr.ones_like(x0))
        x,y,z,t=advector.advect(ff, x0=x0, y0=y0, z0=z0, t0=t0, n=x0.n, selector=selector)
        #after 0.2s, the particle should be at 0.6m
        assert x.sel(it=2).item() == pytest.approx(0.6)
        assert x.sel(it=-2).item() == pytest.approx(-0.6)
class Test_RK4Advector:
    def test_advection(self):
        #steady 3 m/s flow
        u=xr.DataArray([3.0, 3.0], coords=[('z', [0,1.0])]).astype(float)
        ff=adv.FlowField_Collection(u, 0, 0)
        advector=adv.RK4Advector(dt=0.1, steps=10, steps_backward=10, savesteps=None)
        x0=xr.DataArray([0.0], coords=[('n', [0])])
        y0=xr.DataArray([0.0], coords=[('n', [0])])
        z0=xr.DataArray([0.0], coords=[('n', [0])])
        t0=xr.DataArray([0.0], coords=[('n', [0])])
        selector=adv.DataArraySelector(xr.ones_like(x0))
        x,y,z,t=advector.advect(ff, x0=x0, y0=y0, z0=z0, t0=t0, n=x0.n, selector=selector)
        #after 0.2s, the particle should be at 0.6m
        assert x.sel(it=2).item() == pytest.approx(0.6)
        assert x.sel(it=-2).item() == pytest.approx(-0.6)
    
    def test_accuracy_comparison(self):
        """Test that RK4 is more accurate than Euler for the same timestep"""
        # Use a simple velocity field: u = -y, v = x (rotation)
        # Analytical solution is circular motion
        y_coords = xr.DataArray([-1.0, 0.0, 1.0], coords=[('y', [-1.0, 0.0, 1.0])])
        x_coords = xr.DataArray([-1.0, 0.0, 1.0], coords=[('x', [-1.0, 0.0, 1.0])])
        u = -y_coords  # u = -y
        v = x_coords   # v = x
        ff = adv.FlowField_Collection(u, v, 0)
        
        # Starting at (1, 0), after time t, particle should be at (cos(t), sin(t))
        x0 = xr.DataArray([1.0], coords=[('n', [0])])
        y0 = xr.DataArray([0.0], coords=[('n', [0])])
        z0 = xr.DataArray([0.0], coords=[('n', [0])])
        t0 = xr.DataArray([0.0], coords=[('n', [0])])
        selector = adv.DataArraySelector(xr.ones_like(x0))
        
        dt = 0.1
        steps = 10  # total time = 1.0
        
        # Euler advection
        euler_advector = adv.EulerAdvector(dt=dt, steps=steps, savesteps=None)
        x_euler, y_euler, z_euler, t_euler = euler_advector.advect(ff, x0=x0, y0=y0, z0=z0, t0=t0, n=x0.n, selector=selector)
        
        # RK4 advection
        rk4_advector = adv.RK4Advector(dt=dt, steps=steps, savesteps=None)
        x_rk4, y_rk4, z_rk4, t_rk4 = rk4_advector.advect(ff, x0=x0, y0=y0, z0=z0, t0=t0, n=x0.n, selector=selector)
        
        # the time should be the same for both methods
        assert t_euler.sel(it=9).item() == t_rk4.sel(it=9).item()

        # Analytical solution
        t_final= t_euler.sel(it=9).item()
        x_analytical = np.cos(t_final)
        y_analytical = np.sin(t_final)
        
        # RK4 should be more accurate than Euler
        euler_error = np.sqrt((x_euler.sel(it=9).item() - x_analytical)**2 + 
                             (y_euler.sel(it=9).item() - y_analytical)**2)
        rk4_error = np.sqrt((x_rk4.sel(it=9).item() - x_analytical)**2 + 
                           (y_rk4.sel(it=9).item() - y_analytical)**2)
        # print(f"Euler error: {euler_error}, RK4 error: {rk4_error}")
        
        assert rk4_error < euler_error

class Test_TrajectoryCollection:
    def test_backward(self):
        #steady 3 m/s flow
        u=xr.DataArray([3.0, 3.0], coords=[('z', [0,1.0])]).astype(float)
        ff=adv.FlowField_Collection(u, 0,0)
        advector=adv.EulerAdvector(dt=0.1, steps=10, steps_backward=10, savesteps=None)
        traj_coll=adv.Trajectory_Collection.from_flowfield(ff, np.array([0.0]), np.array([0.0]), np.array([0.5]), np.array([0.0]), advector=advector)
        #after 0.2s, the particle should be at 0.6m
        assert traj_coll.ds.dx.sel(it=2).squeeze().item() == pytest.approx(0.6)
        assert traj_coll.ds.dx.sel(it=-2).squeeze().item() == pytest.approx(-0.6)

class Test_ParticleCollection:
    def test_create_particle_collection(self):
        #it is possible to create a particle collection from data arrays
        x0=xr.DataArray([1,2,3], coords={'foo': [1,2,3]})
        y0=xr.DataArray([4,5,6], coords={'bar': [1,2,3]})
        z0=xr.DataArray([7,8,9], coords={'baz': [1,2,3]})
        property=xr.DataArray([13,14,15], coords={'foo': [1,2,3]})
        particles=adv.Particle_Collection(x0=x0, y0=y0, z0=z0, t0=1.0, property=property)
        assert {'n'} == set(particles.ds.dims)
        # if there is a nan, remove the particle by default
        x0=xr.DataArray([np.nan,2,3], coords={'foo': [1,2,3]})
        particles=adv.Particle_Collection(x0=x0, y0=y0, z0=z0, t0=1.0, property=property)
        assert len(particles.ds.n) == 18 #3x3x2
        # this also holds for the field selectors
        selector=xr.DataArray([1,np.nan,3], coords={'foo': [1,2,3]})
        particles=adv.Particle_Collection(x0=x0, y0=y0, z0=z0, t0=1.0, property=property, selector=selector)
        assert len(particles.ds.n) == 9 #3x3x1
        assert len(particles.selector.selector.n)==9
        # after removal, the n coordinate is reset to consecutive integers
        # npt.assert_array_equal(particles.ds.n.values, np.arange(18))
        # what happens if the array has a coordinate with the same name?
        x0=xr.DataArray(np.arange(20), coords=[('x0', np.arange(20))]).astype(float)
        part_coll=adv.Particle_Collection(x0,0.0,9.0,0.0, 1.0)
        assert {'x0', 'y0', 'z0', 't0', 'property'} == set(part_coll.ds.data_vars.keys())
    
    def test_advect_multiple_particles(self):
        #it is possible to advect multiple particles at once
        x0=xr.DataArray([1,2,3], coords={'foo': [1,2,3]})
        y0=xr.DataArray([4,5,6], coords={'bar': [1,2,3]})
        z0=xr.DataArray([7,8,9], coords={'baz': [1,2,3]})
        property=xr.DataArray([13,14,15], coords={'foo': [1,2,3]})
        particles=adv.Particle_Collection(x0=x0, y0=y0, z0=z0, t0=1.0, property=property)
        u=xr.DataArray([1.0], coords=[('z', [0])])
        ff=adv.FlowField_Collection(u=u, v=0.0, w=0.0)
        advector=adv.EulerAdvector(dt=0.1, steps=11)
        paths=adv.Path_Collection.from_flowfield_collection(particles, ff, advector)
        #after 11 steps, each particle should have moved 1m
        x_expected=particles.ds.x0 + 1.0
        npt.assert_allclose(paths.ds.x.sel(it=10).values, x_expected.values)

    def test_to_path_collection(self):
        #it is possible to create a one-step path collection from a particle collection
        x0=xr.DataArray(np.arange(10), dims=['foo'])
        particles=adv.Particle_Collection(x0=x0, y0=0.0, z0=0.0, t0=0.0, property=1.0)
        paths=particles.to_path_collection()
        assert {'n', 'it'}== set(paths.ds.dims)
        xrt.assert_equal(paths.ds.x.isel(it=0).drop_vars('it'), particles.ds.x0)
    def test_broadcast_dim_only(self):
        #trying to create a particle collection from data arrays with non-matching coordinates should raise an error
        x0=xr.DataArray([1,2,3], coords={'foo': [1,2,3]})
        y0=xr.DataArray([1,2,3], coords={'foo': [4,5,6]})
        with pytest.raises(ValueError):
            adv.Particle_Collection(x0=x0, y0=y0, z0=0.0, t0=0.0, property=1.0)

class Test_FieldCollection:
    def test_fill_with_remove_duplicates(self):
        part_coll=adv.Particle_Collection(x0=0, y0=0, z0=0, t0=0, property=1.0)
        it=np.linspace(0,1,10)
        path_x=xr.DataArray(np.linspace(0,1,10), coords=[('it', it)])
        path_y=xr.DataArray(0.0)
        path_z=xr.DataArray(np.linspace(0,1,10), coords=[('it', it)])
        path_t=xr.DataArray(it, coords=[('it', it)])
        path_x, path_y, path_z, path_t=xr.broadcast(path_x, path_y, path_z, path_t)
        ds=xr.Dataset({'x': path_x, 'y': path_y, 'z': path_z, 't': path_t})
        ds=ds.expand_dims('n')
        path_coll=adv.Path_Collection(ds)
        # create a 3x3 field
        field_coll=adv.Field_Collection.create_regular(times=[0,1], xlim=(0,1), ylim=(0,1), zlim=(0,1), nx=4, ny=1, nz=4)
        # if duplicats are not removed, we will sum the particle multiple times
        field_coll.fill_with(part_coll, path_coll, aggregation='sum', remove_duplicates=False)
        assert field_coll.f.max() == 3.0
        # if duplicats are removed, we should at most get the property value of 1
        field_coll=adv.Field_Collection.create_regular(times=[0,1], xlim=(0,1), ylim=(0,1), zlim=(0,1), nx=4, ny=1, nz=4)
        field_coll.fill_with(part_coll, path_coll, aggregation='sum', remove_duplicates=True)
        assert field_coll.f.max() == 1.0


