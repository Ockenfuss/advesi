from __future__ import annotations
from typing import Dict
import numpy as np
import xarray as xr
from advesi.helpers import sort_da, interp_multi

FIELD_BOUNDARY=1e10

def broadcast_dim_only(*arrays: xr.DataArray):
    """Broadcast arrays only by dimensions, raising an error if their coordinates do not match."""
    aligned_arrays = xr.align(*arrays, join='exact')
    return xr.broadcast(*aligned_arrays)

def _broadcast_like_list(da: xr.DataArray, *args: xr.DataArray, exclude=None):
    """Broadcast a given array to multiple others.
    Currently, this is a workaround since xr.DataArray.broadcast_like(other) does not support other to be a list of Arrays."""
    for o in args:
        da=da.broadcast_like(o, exclude=exclude)
    return da

def flatten_da(da):
    """Flatten a DataArray or Dataset to a 1D array with dimension 'n'."""
    if len(da.dims)==0:
        return da.expand_dims(n=[0])
    if len(da.dims)==1:
        dim=list(da.sizes.keys())[0] #get the only dimension
        da=da.rename({dim:'n'})
        da.coords['n']=('n', np.arange(len(da.n)))
        return da
    if len(da.dims)>=2:
        if 'n' in da.dims:
            raise DimensionError("'n' cannot be in dimensions when flattening array!")
        da=da.stack(n=da.dims)
        da=da.drop_vars(list(da.coords.keys())) #drop complete multiindex
        da.coords['n']=('n', np.arange(len(da.n))) #assign new, linear coordinates
        return da


class DimensionError(Exception):
    def __init__(self, value): 
        self.value = value 
    def __str__(self): 
        return(repr(self.value)) 


class FlowField_Collection(object):
    def __init__(self, u,v,w, extend='infinite'):
        u=xr.DataArray(u).astype(float).squeeze(drop=True) #remove 1D dimensions and add them as -Boundary to +Boundary again
        v=xr.DataArray(v).astype(float).squeeze(drop=True)
        w=xr.DataArray(w).astype(float).squeeze(drop=True)
        flowdims={'x','y','z','t','s'}
        for da in [u,v,w]:
            if not set(da.dims).issubset(flowdims):
                raise DimensionError(f"FlowField_Collection is not allowed to have dimensions {da.dims}, only 'x', 'y', 'z', 't', 's' are allowed.")
        u,v,w=broadcast_dim_only(u,v,w)
        u=self._add_missing_dimensions(u)
        v=self._add_missing_dimensions(v)
        w=self._add_missing_dimensions(w)
        # sort all dimensions
        for dim in flowdims:
            u=u.sortby(dim)
            v=v.sortby(dim)
            w=w.sortby(dim)
        if extend=='infinite':
            u=self._extend_dimension(u)
            v=self._extend_dimension(v)
            w=self._extend_dimension(w)
        self.ds= xr.Dataset({'u': u, 'v': v, 'w': w})

    @classmethod
    def from_doppler_birdbath_collection(cls, bb_coll: Birdbath_Collection, u, v, w_points=100):
        """Create a set of flow fields from birdbath doppler velocity information and horizontal wind profiles.

        Parameters
        ----------
        bb_coll : Birdbath_collection
            Birdbath scan with doppler velocity as moment variable.
        u : numeric or xr.DataArray
            u wind field
        v : numeric or xr.DataArray
            v wind field
        w_points : int, optional
            Number of vertical velocities in the resulting flow field collection. The range is taken from the birdbath measurements. by default 100

        Returns
        -------
        FlowField_Collection
            flow fields with different vertical velocities out of the range of doppler velocities in the birdbath scan

        Raises
        ------
        DimensionError
            The moments of the birdbath collection are not allowed to have a 'w' dimension, since this one is introduced for the collection of flow fields.
        """
        if 'w' in bb_coll.moment.dims:
            raise DimensionError("Birdbath_Collection is not allowed to have a 'w' dimension, since this name will be used for the new doppler velocity axis")
        w=np.linspace(bb_coll.moment.min(), bb_coll.moment.max(),w_points)
        w=xr.DataArray(w, coords=[('w', w)])
        return cls(u,v,w)



    
    def _add_missing_dimensions(self,da: xr.DataArray):
        """If one of x,y,z,t,s is missing in the input velocity components, add it as a coordinate from -advesi.FIELD_BOUNDARY to advesi.FIELD_BOUNDARY.
        For the selector, add it as 1"""
        for d in ["x", "y","z", "t", "s"]:
            if d not in da.dims:
                if d == "s":
                    coord= xr.DataArray([1], coords=[(d, [1])])
                else:
                    coord=xr.DataArray([-FIELD_BOUNDARY, FIELD_BOUNDARY], coords=[(d, [-FIELD_BOUNDARY, FIELD_BOUNDARY])])
                da=da.broadcast_like(coord)
        return da
    
    def _extend_dimension(self, da: xr.DataArray):
        """Extend the given DataArray in all dimensions to -advesi.FIELD_BOUNDARY and advesi.FIELD_BOUNDARY if not already present."""
        for d in ["x", "y","z", "t"]:
            if da[d].isel({d:0}).values>-FIELD_BOUNDARY:
                lower=da.isel({d:[0]})
                lower.coords[d]=(d, [-FIELD_BOUNDARY])
                da=xr.concat([lower, da], dim=d)
            if da[d].isel({d:-1}).values<FIELD_BOUNDARY:
                upper=da.isel({d:[-1]})
                upper.coords[d]=(d, [FIELD_BOUNDARY])
                da=xr.concat([da, upper], dim=d)
        return da
    
    def _get_nearest(self, x, y, z, t, s):
        return self.ds.sel(x=x, y=y, z=z, t=t, s=s, method='nearest').drop_vars(["x","y","z", "t", "s"])
    
    def _get_interp(self, x, y, z, t, s):
        return self.ds.interp(x=x, y=y, z=z, t=t, s=s, kwargs={'fill_value':None}).drop_vars(["x","y","z", "t", "s"])

    def get_values(self, x, y, z, t,s, method='nearest'):
        if method=='nearest':
            return self._get_nearest(x, y, z, t, s)
        elif method=='interpolate':
            return self._get_interp(x, y, z, t, s)
        else:
            raise KeyError(f"Method '{method}' not available. Use 'nearest' or 'interpolate'.")
    
    def __repr__(self):
        return self.ds.__repr__()

class FieldSelector(object):
    def get_field_selector(self, x,y,z,t,n):
        raise NotImplementedError("This method should be implemented in subclasses.")


class FunctionSelector(FieldSelector):
    def __init__(self, s_func):
        self.s_func=s_func
    def get_field_selector(self, x, y, z, t, n):
            return self.s_func(x,y,z,t,n)

class DataArraySelector(FieldSelector):
    def __init__(self, selector: xr.DataArray):
        if not isinstance(selector, xr.DataArray):
            raise TypeError("Selector must be an xarray DataArray.")
        self.selector=selector
    def get_field_selector(self, x, y, z, t, n):
        return self.selector.sel(n=n)



class Advector(object):
    def __init__(self):
        """Advector class to advect particles in a flow field. This solves the general advection equation dP/dt = F(P,t) where P is the particle position and F the flow field."""
        pass

    def _get_F(self, flowfield : FlowField_Collection, x, y, z, t, n, selector: FieldSelector):
        ds= flowfield.get_values(x, y, z, t, s=selector.get_field_selector(x,y,z,t,n), method=self.interp_method)
        return ds.u, ds.v, ds.w
    
    def _one_step(self, flowfield : FlowField_Collection, x, y, z, t,n, dt, selector: FieldSelector):
        """Advect the particles one time step. Must return xnew, ynew, znew, tnew."""
        pass

    def _forward(self, flowfield : FlowField_Collection, x0, y0, z0, t0,n, dt, save_steps, intermediate_interval, selector: FieldSelector):
            it=np.sign(save_steps)*np.arange(0, abs(save_steps))
            it=xr.DataArray(it, coords=[('it', it)])
            X,Y,Z,T,n=broadcast_dim_only(x0,y0,z0,t0,n)
            # X,Y,Z,T,_=broadcast_dim_only(X,Y,Z,T,it)
            # X=X.copy() # broadcast returns only a view
            # Y=Y.copy()
            # Z=Z.copy()
            # T=T.copy()
            # Xc=X.isel(it=0).copy().drop_vars("it") #current positions in iteration scheme
            # Yc=Y.isel(it=0).copy().drop_vars("it")
            # Zc=Z.isel(it=0).copy().drop_vars("it")
            # Tc=T.isel(it=0).copy().drop_vars("it")
            Xc=X
            Yc=Y
            Zc=Z
            Tc=T

            Xc_list=[]
            Yc_list=[]
            Zc_list=[]
            Tc_list=[]
            for save_it in range(len(it)):
                Xc_list.append(Xc)
                Yc_list.append(Yc)
                Zc_list.append(Zc)
                Tc_list.append(Tc)
                # X[{"it":save_it}]=Xc
                # Y[{"it":save_it}]=Yc
                # Z[{"it":save_it}]=Zc
                # T[{"it":save_it}]=Tc
                for intermediate_it in range(intermediate_interval):
                    Xc,Yc,Zc,Tc=self._one_step(flowfield, Xc, Yc, Zc, Tc,n, dt, selector)
            X=xr.concat(Xc_list, dim=it)
            Y=xr.concat(Yc_list, dim=it)
            Z=xr.concat(Zc_list, dim=it)
            T=xr.concat(Tc_list, dim=it)
            return X,Y,Z,T
    
    def advect(self, flowfield : FlowField_Collection, x0, y0, z0, t0,n, selector):
            Xf,Yf,Zf,Tf=self._forward(flowfield, x0, y0, z0, t0,n, self.dt, self.steps_forward, self.intermediate_interval, selector)
            if self.steps_backward>0:
                Xb,Yb,Zb,Tb=self._forward(flowfield, x0, y0, z0, t0,n, -self.dt, -self.steps_backward, self.intermediate_interval, selector)
                Xf=xr.concat([Xb.drop_sel(it=0), Xf], dim='it').sortby('it')
                Yf=xr.concat([Yb.drop_sel(it=0), Yf], dim='it').sortby('it')
                Zf=xr.concat([Zb.drop_sel(it=0), Zf], dim='it').sortby('it')
                Tf=xr.concat([Tb.drop_sel(it=0), Tf], dim='it').sortby('it')
            return Xf, Yf, Zf, Tf

class EulerAdvector(Advector):
    def __init__(self, dt, steps, steps_backward=0, savesteps=None, interp_method='interpolate'):
        self.dt= dt
        total_steps=steps+steps_backward
        if savesteps is None:
            savesteps=total_steps
        if savesteps<2:
            raise ValueError("savesteps must be at least 2")
        if total_steps<savesteps:
            raise ValueError("steps must be at least savesteps")
        intermediate_interval=int(total_steps/savesteps)
        self.steps_forward=int(steps/total_steps*savesteps)
        self.steps_backward=int(steps_backward/total_steps*savesteps)
        self.intermediate_interval=intermediate_interval
        self.interp_method=interp_method
    
    def _one_step(self, flowfield, x, y, z, t, n, dt, selector):
        u,v,w=self._get_F(flowfield, x, y, z, t, n, selector)
        xnew=x+u*dt
        ynew=y+v*dt
        znew=z+w*dt
        tnew=t+dt
        return xnew, ynew, znew, tnew
        
class RK4Advector(Advector):
    def __init__(self, dt, steps, steps_backward=0, savesteps=None, interp_method='interpolate'):
        self.dt = dt
        total_steps = steps + steps_backward
        if savesteps is None:
            savesteps = total_steps
        if savesteps < 2:
            raise ValueError("savesteps must be at least 2")
        if total_steps < savesteps:
            raise ValueError("steps must be at least savesteps")
        intermediate_interval = int(total_steps / savesteps)
        self.steps_forward = int(steps / total_steps * savesteps)
        self.steps_backward = int(steps_backward / total_steps * savesteps)
        self.intermediate_interval = intermediate_interval
        self.interp_method = interp_method
    
    def _one_step(self, flowfield, x, y, z, t, n, dt, selector):
        # k1 = F(P, t)
        u1, v1, w1 = self._get_F(flowfield, x, y, z, t, n, selector)
        
        # k2 = F(P + 0.5*dt*k1, t + 0.5*dt)
        x2 = x + 0.5 * dt * u1
        y2 = y + 0.5 * dt * v1
        z2 = z + 0.5 * dt * w1
        t2 = t + 0.5 * dt
        u2, v2, w2 = self._get_F(flowfield, x2, y2, z2, t2, n, selector)
        
        # k3 = F(P + 0.5*dt*k2, t + 0.5*dt)
        x3 = x + 0.5 * dt * u2
        y3 = y + 0.5 * dt * v2
        z3 = z + 0.5 * dt * w2
        t3 = t + 0.5 * dt
        u3, v3, w3 = self._get_F(flowfield, x3, y3, z3, t3, n, selector)
        
        # k4 = F(P + dt*k3, t + dt)
        x4 = x + dt * u3
        y4 = y + dt * v3
        z4 = z + dt * w3
        t4 = t + dt
        u4, v4, w4 = self._get_F(flowfield, x4, y4, z4, t4, n, selector)
        
        # P_new = P + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        xnew = x + dt / 6.0 * (u1 + 2*u2 + 2*u3 + u4)
        ynew = y + dt / 6.0 * (v1 + 2*v2 + 2*v3 + v4)
        znew = z + dt / 6.0 * (w1 + 2*w2 + 2*w3 + w4)
        tnew = t + dt
        
        return xnew, ynew, znew, tnew


class Particle_Collection(object):
    def __init__(self, x0: float | xr.DataArray, y0: float | xr.DataArray, z0: float | xr.DataArray, t0: float | xr.DataArray, property: float | xr.DataArray, selector=None, remove_nans=True):
        x0=xr.DataArray(x0).astype(float)
        y0=xr.DataArray(y0).astype(float)
        z0=xr.DataArray(z0).astype(float)
        t0=xr.DataArray(t0).astype(float)
        property=xr.DataArray(property).astype(float)
        x0,y0,z0,t0,property=broadcast_dim_only(x0,y0,z0,t0,property)
        all_arrays={'x0': x0, 'y0': y0, 'z0': z0, 't0': t0, 'property': property}
        if callable(selector):
            self.selector=FunctionSelector(selector)
        else:
            if selector is None:
                selector=xr.ones_like(x0)
            selector=xr.DataArray(selector).astype(float)
            all_arrays['selector']=selector
        
        # broadcast all arrays to the same shape
        broadcasted_arrays=broadcast_dim_only(*all_arrays.values())
        for i, key in enumerate(all_arrays.keys()):
            all_arrays[key]=broadcasted_arrays[i]
        # flatten all arrays to 1D
        all_arrays={k:flatten_da(v) for k,v in all_arrays.items()}
        # remove nans if desired
        if remove_nans:
            valid=np.logical_and.reduce([da.notnull() for da in all_arrays.values()])
            all_arrays={k:v.isel(n=valid) for k,v in all_arrays.items()}
        # create ds and optionally DataArraySelector
        ds=xr.Dataset({k:all_arrays[k] for k in ['x0','y0','z0','t0','property']})
        self.ds=ds
        if 'selector' in all_arrays:
            self.selector=DataArraySelector(all_arrays['selector'])

    
    @classmethod
    def from_field_collection(cls, field_coll: Field_Collection, selector=None):
        return cls(field_coll.f.x, field_coll.f.y, field_coll.f.z, field_coll.f.t, property=field_coll.f, selector=selector)
    
    def to_path_collection(self):
        """Convert the particle collection to a path collection with only the initial positions"""
        ds=xr.Dataset({'x': self.ds.x0, 'y': self.ds.y0, 'z': self.ds.z0, 't': self.ds.t0})
        ds=ds.expand_dims(it=[0])
        return Path_Collection(ds)
    
    def __repr__(self):
        return self.ds.__repr__()
class Trajectory_Collection(object):
    def __init__(self, ds_trajectories: xr.Dataset) -> None:
        """Create a collection of trajectories. Currently, trajectory collections are not allowed to be completely unstructured, but rather the starting points must form a regular grid.
        This choice facilitates lookups of trajectories and allows for interpolation to arbitrary points collections.

        Parameters
        ----------
        ds_trajectories : xr.Dataset
            Dataset with coordinates of the trajectories. Must have variables dx,dy,dz, dt and coordinates 'x0', 'y0', 'z0', 't0' and it
        """
        if not {'dx', 'dy', 'dz', 'dt'} == set(ds_trajectories.data_vars.keys()):
            raise DimensionError("Trajectory collections must have variables 'dx', 'dy', 'dz', 'dt'.")
        if not {'x0', 'y0', 'z0', 't0', 'it'} <= set(ds_trajectories.coords.keys()):
                raise DimensionError("Trajectories must have 'x0', 'y0', 'z0', 't0' and 'it' coordinates.")
        self.ds=ds_trajectories
    
    @classmethod
    def from_flowfield(cls, flowfield: FlowField_Collection,x0:np.ndarray | float , y0: np.ndarray | float, z0: np.ndarray | float, t0: np.ndarray | float, advector: Advector):
        da_x0=np.array(x0, dtype=float).flatten()
        da_y0=np.array(y0, dtype=float).flatten()
        da_z0=np.array(z0, dtype=float).flatten()
        da_t0=np.array(t0, dtype=float).flatten()
        da_x0=xr.DataArray(da_x0, coords=[('x0',da_x0)])
        da_y0=xr.DataArray(da_y0, coords=[('y0',da_y0)])
        da_z0=xr.DataArray(da_z0, coords=[('z0',da_z0)])
        da_t0=xr.DataArray(da_t0, coords=[('t0', da_t0)])
        n=xr.zeros_like(da_x0)
        s0=flowfield.ds.s
        n,_=broadcast_dim_only(n,s0)
        selector=FunctionSelector(lambda x,y,z,t,n: s0.broadcast_like(x))
        X,Y,Z,t=advector.advect(flowfield, da_x0, da_y0, da_z0, da_t0,n, selector=selector) #result: [x0, y0, z0, t0, s, it]
        # calculate relative displacements
        dx=X - da_x0
        dy=Y - da_y0
        dz=Z - da_z0
        dt=t - da_t0
        ds_trajectories=xr.Dataset({'dx': dx, 'dy': dy, 'dz': dz, 'dt': dt})
        return cls(ds_trajectories)
    
    def __repr__(self):
        return self.ds.__repr__()


class Path_Collection(object):
    def __init__(self, ds):
        if not {'n', 'it'}==set(ds.dims):
            raise DimensionError(f"Path Collection dataset has only dimensions 'n' and 'it' allowed. Dimensions are {ds.dims}.")
        if not {'x', 'y', 'z', 't'} <= set(ds.data_vars.keys()):
            raise DimensionError("Path collections must have variables 'x', 'y', 'z', 't'.")
        valid=ds.notnull()
        reduced=np.logical_and.reduce([v for v in valid.data_vars.values()])
        self.ds=ds.where(reduced)
        

    
    @classmethod
    def from_trajectory_collection(cls, particle_coll: Particle_Collection, rel_traj_coll: Trajectory_Collection, matching: str='nearest', substeps=1):
        if substeps !=1:
            if matching not in ['interpolate', 'nearest_interpolate']:
                raise ValueError("Substeps only make sense for 'interpolate' or 'nearest_interpolate' matching.")
            stepping=1/substeps
            it=np.arange(0, rel_traj_coll.ds.it.max().item(), stepping)
            it=xr.DataArray(it, coords=[('it', it)])
            it_selector={'it': it}
        else:
            it_selector={}
        field_selector=particle_coll.selector.get_field_selector(particle_coll.ds.x0, particle_coll.ds.y0, particle_coll.ds.z0, particle_coll.ds.t0, particle_coll.ds.n) #We choose the field purely based on the initial time. "Evovling" particles are not supported in combination with trajectories.
        field_selector={'s': field_selector}
        if matching == 'exact':
            selector={'x0': particle_coll.ds.x0, 'y0': particle_coll.ds.y0, 'z0': particle_coll.ds.z0, 't0': particle_coll.ds.t0} | field_selector
            rel_paths=rel_traj_coll.ds.sel(selector)
        elif matching == 'nearest':
            selector={'x0': particle_coll.ds.x0, 'y0': particle_coll.ds.y0, 'z0': particle_coll.ds.z0, 't0': particle_coll.ds.t0} | field_selector
            rel_paths=rel_traj_coll.ds.sel(selector, method='nearest')
        elif matching == 'interpolate':
            selector={'x0': particle_coll.ds.x0, 'y0': particle_coll.ds.y0, 'z0': particle_coll.ds.z0, 't0': particle_coll.ds.t0} | field_selector | it_selector
            rel_paths=rel_traj_coll.ds.interp(selector, kwargs={'fill_value': None})
        # Problem: These hybrid selections can produce very large intermediate arrays, if len 1 dims are first blown up before the second selection reduces the size again.
        # Therefore: First select along len>1 dimensions, then select nearest along len=1 dimensions.
        elif matching == 'nearest_exact':
            #len=1 dimensions are selected nearest, since the flowfield is likely symmetric in these dimensions
            len1_dims=[d for d in ['x0','y0','z0','t0'] if rel_traj_coll.ds[d].size==1]
            lenn_dims=[d for d in ['x0','y0','z0','t0'] if rel_traj_coll.ds[d].size>1]
            nearest_selector={d:particle_coll.ds[d] for d in len1_dims}
            exact_selectors={d:particle_coll.ds[d] for d in lenn_dims} | field_selector
            rel_paths=rel_traj_coll.ds.sel(exact_selectors).sel(nearest_selector, method='nearest')
        elif matching == 'nearest_interpolate':
            #len=1 dimensions are selected nearest, others are interpolated
            len1_dims=[d for d in ['x0','y0','z0','t0'] if rel_traj_coll.ds[d].size==1]
            lenn_dims=[d for d in ['x0','y0','z0','t0'] if rel_traj_coll.ds[d].size>1]
            nearest_selector={d:particle_coll.ds[d] for d in len1_dims}
            interp_selectors={d:particle_coll.ds[d] for d in lenn_dims} | it_selector | field_selector
            # exact_selectors= field_selector
            rel_paths=rel_traj_coll.ds.interp(interp_selectors).sel(nearest_selector, method='nearest')
        else:
            raise KeyError(f"Matching '{matching}' not available.")

        rel_paths=rel_paths.drop_vars(['x0', 'y0', 'z0', 't0'], errors='ignore').squeeze(drop=True) 
        abs_paths=xr.Dataset({
            'x': rel_paths.dx + particle_coll.ds.x0,
            'y': rel_paths.dy + particle_coll.ds.y0,
            'z': rel_paths.dz + particle_coll.ds.z0,
            't': rel_paths.dt + particle_coll.ds.t0
        })
        return cls(abs_paths)
    
    @classmethod
    def from_flowfield_collection(cls, particle_coll: Particle_Collection, field_coll: Field_Collection, advector: Advector):
        """Create a path collection from initial particle positions and a field collection.
            In this case, the fields can be time dependent and the paths are calculated explicitly for each particle. For stationary fields, it is usually more efficient to calculate trajectories first and then create the path collection from the trajectory collection.
        """
        X,Y,Z,T=advector.advect(field_coll, particle_coll.ds.x0, particle_coll.ds.y0, particle_coll.ds.z0, particle_coll.ds.t0, particle_coll.ds.n, particle_coll.selector)
        ds=xr.Dataset({'x': X, 'y': Y, 'z': Z, 't': T})
        return cls(ds)

    def __repr__(self):
        return self.ds.__repr__()



    # @classmethod
    # def from_doppler_birdbath_collection(cls,bb_coll: Birdbath_Collection, traj_coll: Trajectory_Collection, t:xr.DataArray=None):
    #     part_coll=Particle_Collection.from_doppler_birdbath_collection(bb_coll)
    #     if t is None:
    #         t=bb_coll.moment.t
    #     return cls.from_trajectory_collection(part_coll, traj_coll,t)
    
    def get_valid_mask(self):
        valid=self.ds.x.notnull() #it is enough to check x, since by construction we equalize nans between x,y,z
        return valid
    
    def _sel_nearest(self, var:str, dim:str, value, tolerance=None):
        """Select the nearest value along a given dimension."""
        if var not in {'x', 'y', 'z', 't'}:
            raise KeyError(f"Variable '{var}' must be one of 'x', 'y', 'z', 't'.")
        if dim not in {'n', 'it'}:
            raise KeyError(f"Dimension '{dim}' must be one of 'n', 'it'.")
        nearest_idx=abs(self.ds[var]-value).argmin(dim)
        result=self.ds.isel({dim: nearest_idx})
        if tolerance is not None:
            valid=abs(self.ds[var]-value).min(dim)<=tolerance
            result=result.where(valid)
        return result

    def sel_nearest_time(self, time, tolerance=None):
        """For every particle, choose the nearest time step in the dataset.
        This is useful, since paths are stored in format (n, it), so selecting an absolute time is not straigtforward.
        """
        #maybe, in the future this can be acomplisehd with NDPointIndex, but currently, the tolerance is not supported for sel() calls with this index.
        result=self._sel_nearest('t', 'it', time, tolerance)
        return result
    
    def _interp_to_position(self, direction, position, check_sorted=False):
        """Interpolate the paths to the given position in the given direction.

        Parameters
        ----------
        direction : str
            Direction to interpolate to. Must be one of 'x', 'y', 'z', 't'.
        position : float
            Position in the given direction to interpolate to.

        Returns
        -------
        xr.Dataset
            Dataset with variables x,y,z,t at the interpolated positions.
        """
        if direction not in {'x', 'y', 'z', 't'}:
            raise KeyError(f"Direction '{direction}' must be one of 'x', 'y', 'z', 't'.")
        if check_sorted:
            diff=self.ds[direction].diff('it')
            if (diff<0).any():
                raise ValueError(f"Path collection is not sorted along direction '{direction}' in increasing order. Cannot interpolate.")
        interpolation=interp_multi(self.ds, direction, 'it', position)
        return interpolation

    def interp_to_xyz(self, direction, position, check_sorted=True):
        return self._interp_to_position(direction, position, check_sorted)
    
    def interp_to_time(self, time):
        """Like sel_nearest_time, but interpolate to the given time instead of selecting the nearest time step."""
        return self._interp_to_position('t', time, check_sorted=False)



class Field_Collection(object):
    def __init__(self, f) -> None:
        if not {'x', 'y', 'z', 't'}<=set(f.dims):
            raise DimensionError("A field collection must have dimensions 'x', 'y', 'z' and 't'.")
        self.f=f.astype(float).sortby(['x', 'y', 'z', 't'])
        ix=xr.DataArray(np.arange(len(f.x)), coords=[('x', f.x.values)])
        iy=xr.DataArray(np.arange(len(f.y)), coords=[('y', f.y.values)])
        iz=xr.DataArray(np.arange(len(f.z)), coords=[('z', f.z.values)])
        it=xr.DataArray(np.arange(len(f.t)), coords=[('t', f.t.values)])
        ny=len(f.y)
        nz=len(f.z)
        nt=len(f.t)
        # unique id for every cell in the field
        self.id=ix*ny*nz*nt + iy*nz*nt + iz*nt + it
    
    def unravel_index(self, ind:xr.DataArray):
        """xarray wrapper around np.unravel_index with C style order."""
        nx=len(self.f.x)
        ny=len(self.f.y)
        nz=len(self.f.z)
        nt=len(self.f.t)
        unravel=lambda da: np.unravel_index(da, shape=(nx, ny, nz, nt), order='C')
        indx, indy, indz, indt=xr.apply_ufunc(unravel, ind, output_core_dims=[[],[],[], []])
        return indx, indy, indz, indt
    
    def _out_of_field(self, positions: xr.Dataset):
        """Check if the given positions are out of the field."""
        x=positions.x
        y=positions.y
        z=positions.z
        t=positions.t
        xmin, xmax = self.f.x.min(), self.f.x.max()
        ymin, ymax = self.f.y.min(), self.f.y.max()
        zmin, zmax = self.f.z.min(), self.f.z.max()
        tmin, tmax = self.f.t.min(), self.f.t.max()
        out_of_range= (x<xmin) | (x>xmax) | (y<ymin) | (y>ymax) | (z<zmin) | (z>zmax) | (t<tmin) | (t>tmax)
        return out_of_range

    
    def _get_id(self, positions: xr.Dataset):
        """Given a Dataset with positions as variables x,y,z,t, find the id of the corresponding field cells."""
        x=positions.x
        y=positions.y
        z=positions.z
        t=positions.t
        xmin, xmax = self.f.x.min(), self.f.x.max()
        ymin, ymax = self.f.y.min(), self.f.y.max()
        zmin, zmax = self.f.z.min(), self.f.z.max()
        tmin, tmax = self.f.t.min(), self.f.t.max()
        out_of_range= (x<xmin) | (x>xmax) | (y<ymin) | (y>ymax) | (z<zmin) | (z>zmax) | (t<tmin) | (t>tmax)
        x=positions.x.clip(xmin, xmax)
        y=positions.y.clip(ymin, ymax)
        z=positions.z.clip(zmin, zmax) 
        t=positions.t.clip(tmin, tmax)
        id=self.id.sel(x=x, y=y, z=z, t=t, method='ffill')
        id=id.where(~out_of_range, other=-1)
        return id
        

    
    @classmethod
    def create_regular(cls, times: np.ndarray | float, xlim: tuple, ylim: tuple, zlim: tuple, nx=100, ny=100, nz=100, fill_value=np.nan):
        #create field axes, big enough to hold all paths completely
        x=np.linspace(xlim[0], xlim[1], nx)
        x=xr.DataArray(x, coords=[('x', x)])
        y=np.linspace(ylim[0], ylim[1], ny)
        y=xr.DataArray(y, coords=[('y', y)])
        z=np.linspace(zlim[0], zlim[1], nz)
        z=xr.DataArray(z, coords=[('z', z)])
        #Time axis is given as argument
        ft=np.array(times).flatten()
        ft=xr.DataArray(ft, coords=[('t', ft)])
        field_da=xr.full_like(broadcast_dim_only(x,y,z,ft)[0],fill_value=fill_value)
        field=cls(field_da)
        return field

    @classmethod
    def from_particle_collection(cls, part_coll: Particle_Collection, traj_coll: Trajectory_Collection, times: np.ndarray, nx=100, ny=100, nz=100, aggregation='max'):
        path_coll=Path_Collection.from_trajectory_collection(part_coll, traj_coll, times)
        ds_min=path_coll.ds.min()
        ds_max=path_coll.ds.max()
        xlim=(ds_min.x.item(), ds_max.x.item())
        ylim=(ds_min.y.item(), ds_max.y.item())
        zlim=(ds_min.z.item(), ds_max.z.item())
        f=cls.create_regular(times, xlim, ylim, zlim, nx, ny, nz)
        f.fill_with(part_coll, path_coll, aggregation)
        return f

    def fill_with(self, part_coll: Particle_Collection, path_coll: Path_Collection, aggregation='max', remove_duplicates=False):
        """_summary_

        Parameters
        ----------
        part_coll : Particle_Collection
            _description_
        path_coll : Path_Collection
            _description_
        aggregation : str, optional
            _description_, by default 'max'
        remove_duplicates : bool, optional
            Make sure every particle is counted only once per cell. This is important for aggregations like 'sum' and e.g. very high trajectory resolutions. By default False

        Raises
        ------
        ValueError
            _description_
        """
        #set all positions outside of the field to invalid values
        id=self._get_id(path_coll.ds)
        if remove_duplicates:
            #idea: sort ids in id[n,it] along it. Then, make sure that every cell id occurs at most once. This can be achieved by a diff on the sorted ids, then setting all ids with a diff of 0 to -1.
            id=sort_da(id, dim='it')
            id.coords['it']=np.arange(len(id.it)) 
            diffs=id.diff('it', label='upper') #with upper, we will keep the first id
            diffs=diffs.reindex_like(id, fill_value=999)
            id=id.where(diffs!=0, other=-1)
        #Add the combined index as a multi index to the particle property field and use groupby
        particle_property=part_coll.ds.property.broadcast_like(path_coll.ds) #add the time dimension
        particle_property.coords["id"]=(id.dims, id.values)
        particle_property=particle_property.groupby('id')
        if aggregation=='max':
            particle_property=particle_property.max()
        elif aggregation=='min':
            particle_property=particle_property.min()
        elif aggregation=='mean':
            particle_property=particle_property.mean()
        elif aggregation=='median':
            particle_property=particle_property.median()
        elif aggregation=='sum':
            particle_property=particle_property.sum()
        else:
            raise ValueError(f"Aggregation '{aggregation}' is currently not implemented.")
        #remove the group, which contains all invalid spatio-temporal particle positions
        particle_property=particle_property.where(particle_property.id>=0, drop=True)
        property_index_x, property_index_y, property_index_z, property_index_ft=self.unravel_index(particle_property.id)
        self.f[{"x":property_index_x,"y":property_index_y, "z":property_index_z, "t":property_index_ft}]=particle_property.values
    
    def __repr__(self):
        return self.f.__repr__()
