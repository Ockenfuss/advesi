from __future__ import annotations
from typing import Dict
import numpy as np
import xarray as xr

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
    if len(da.dims)==0:
        return da.expand_dims(n=[0])
    if len(da.dims)==1:
        da=da.rename({da.dims[0]:'n'})
        da.coords['n']=('n', np.arange(len(da)))
        return da
    if len(da.dims)>=2:
        if 'n' in da.dims:
            raise DimensionError("'n' cannot be in dimensions when flattening array!")
        da=da.stack(n=da.dims)
        da=da.drop_vars(list(da.coords.keys())) #drop complete multiindex
        da.coords['n']=('n', np.arange(len(da))) #assign new, linear coordinates
        return da


class DimensionError(Exception):
    def __init__(self, value): 
        self.value = value 
    def __str__(self): 
        return(repr(self.value)) 


class FlowField_Collection(object):
    def __init__(self, u,v,w):
        u=xr.DataArray(u).astype(float).squeeze(drop=True) #remove 1D dimensions and add them as -Boundary to +Boundary again
        v=xr.DataArray(v).astype(float).squeeze(drop=True)
        w=xr.DataArray(w).astype(float).squeeze(drop=True)
        for da in [u,v,w]:
            for d in ['u', 'v', 'w']:
                if d in da.dims:
                    raise DimensionError(f"FlowField_Collection is not allowed to have a '{d}' dimension, since this name will be used for the variable '{d}' in the internal dataset.")
        u,v,w=broadcast_dim_only(u,v,w)
        u=self._add_missing_dimensions(u)
        v=self._add_missing_dimensions(v)
        w=self._add_missing_dimensions(w)
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
        """If one of x,y,z,t is missing in the input velocity components, add it as a coordinate from -advesi.FIELD_BOUNDARY to advesi.FIELD_BOUNDARY."""
        for d in ["x", "y","z", "t"]:
            if d not in da.dims:
                coord=xr.DataArray([-FIELD_BOUNDARY, FIELD_BOUNDARY], coords=[(d, [-FIELD_BOUNDARY, FIELD_BOUNDARY])])
                da=da.broadcast_like(coord)
        return da
    
    def _broadcast_array_to_field(self,da):
        return _broadcast_like_list(da, self.ds.u, self.ds.v, self.ds.w, exclude=["x", "y","z", "t"])
    
    def _get_nearest(self, x, y, z, t):
        # u=self.u.sel(x=x, y=y, z=z, t=t, method='nearest').drop_vars(["x","y","z"])
        # v=self.v.sel(x=x, y=y, z=z, t=t, method='nearest').drop_vars(["x","y","z"])
        # w=self.w.sel(x=x, y=y, z=z, t=t, method='nearest').drop_vars(["x","y","z"])
        return self.ds.sel(x=x, y=y, z=z, t=t, method='nearest').drop_vars(["x","y","z"])
    
    def _get_interp(self, x, y, z, t):
        # u=self.u.interp(x=x, y=y, z=z, t=t, kwargs={'fill_value':None}).drop_vars(["x","y","z"])
        # v=self.v.interp(x=x, y=y, z=z, t=t, kwargs={'fill_value':None}).drop_vars(["x","y","z"])
        # w=self.w.interp(x=x, y=y, z=z, t=t, kwargs={'fill_value':None}).drop_vars(["x","y","z"])
        return self.ds.interp(x=x, y=y, z=z, t=t, kwargs={'fill_value':None}).drop_vars(["x","y","z"])

    def get_values(self, x, y, z, t, method='nearest'):
        if method=='nearest':
            return self._get_nearest(x, y, z, t)
        elif method=='interpolate':
            return self._get_interp(x, y, z, t)
        else:
            raise KeyError(f"Method '{method}' not available. Use 'nearest' or 'interpolate'.")
    
    def __repr__(self):
        return self.ds.__repr__()

class Advector(object):
    def __init__(self):
        """Advector class to advect particles in a flow field."""
        pass
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

    def _forward(self, flowfield : FlowField_Collection, x0, y0, z0, t0, dt, save_steps, intermediate_interval):
            it=np.sign(save_steps)*np.arange(0, abs(save_steps))
            it=xr.DataArray(it, coords=[('it', it)])
            X,Y,Z,T,_=broadcast_dim_only(x0,y0,z0,t0, it)
            X=flowfield._broadcast_array_to_field(X).copy()
            Y=flowfield._broadcast_array_to_field(Y).copy()
            Z=flowfield._broadcast_array_to_field(Z).copy()
            T=flowfield._broadcast_array_to_field(T).copy()
            Xi=X.isel(it=0).copy().drop_vars("it") #current positions in iteration scheme
            Yi=Y.isel(it=0).copy().drop_vars("it")
            Zi=Z.isel(it=0).copy().drop_vars("it")
            Ti=T.isel(it=0).copy().drop_vars("it")

            for save_it in range(len(it)):
                X[{"it":save_it}]=Xi
                Y[{"it":save_it}]=Yi
                Z[{"it":save_it}]=Zi
                T[{"it":save_it}]=Ti
                for intermediate_it in range(intermediate_interval):
                    ds= flowfield.get_values(Xi, Yi, Zi, Ti, method=self.interp_method)
                    Xi=Xi+ds.u*dt
                    Yi=Yi+ds.v*dt
                    Zi=Zi+ds.w*dt
                    Ti=Ti+dt
            return X,Y,Z,T
    
    def advect(self, flowfield : FlowField_Collection, x0, y0, z0, t0):
            Xf,Yf,Zf,Tf=self._forward(flowfield, x0, y0, z0, t0, self.dt, self.steps_forward, self.intermediate_interval)
            if self.steps_backward>0:
                Xb,Yb,Zb,Tb=self._forward(flowfield, x0, y0, z0, t0, -self.dt, -self.steps_backward, self.intermediate_interval)
                Xf=xr.concat([Xb.drop_sel(it=0), Xf], dim='it').sortby('it')
                Yf=xr.concat([Yb.drop_sel(it=0), Yf], dim='it').sortby('it')
                Zf=xr.concat([Zb.drop_sel(it=0), Zf], dim='it').sortby('it')
                Tf=xr.concat([Tb.drop_sel(it=0), Tf], dim='it').sortby('it')
            return Xf, Yf, Zf, Tf

        

class Particle_Collection(object):
    def __init__(self, x0: float | xr.DataArray, y0: float | xr.DataArray, z0: float | xr.DataArray, t0: float | xr.DataArray, property: float | xr.DataArray, field_selectors: Dict[str, xr.DataArray]={}):
        x0=xr.DataArray(x0).astype(float)
        y0=xr.DataArray(y0).astype(float)
        z0=xr.DataArray(z0).astype(float)
        t0=xr.DataArray(t0).astype(float)
        property=xr.DataArray(property).astype(float)
        field_selectors={k:xr.DataArray(v) for k,v in field_selectors.items()}
        #first, broadcast all given arrays
        select_keys=list(field_selectors.keys())
        select_arrays=list(field_selectors.values())
        x0,y0,z0,t0,property,*select_arrays=broadcast_dim_only(x0,y0,z0,t0,property,*select_arrays)
        field_selectors=dict(zip(select_keys, select_arrays))
        #flatten all arrays
        #TODO: Maybe, we can use numpy from here on, since flattened, the multidimensional capabilities of xarray are not necessary
        self.x0=flatten_da(x0)
        self.y0=flatten_da(y0)
        self.z0=flatten_da(z0)
        self.t0=flatten_da(t0)
        self.property=flatten_da(property)
        field_selectors={k:flatten_da(da) for k,da in field_selectors.items()}
        self.field_selectors=field_selectors

    
    @classmethod
    def from_field_collection(cls, field_coll: Field_Collection, field_selectors: Dict[str, xr.DataArray]={}):
        return cls(field_coll.f.x, field_coll.f.y, field_coll.f.z, field_coll.f.t, property=field_coll.f, field_selectors=field_selectors)
    
    def to_path_collection(self):
        """Convert the particle collection to a path collection with only the initial positions"""
        ds=xr.Dataset({'x': self.x0, 'y': self.y0, 'z': self.z0, 't': self.t0})
        ds=ds.expand_dims(it=[0])
        return Path_Collection(ds)
        
class Trajectory_Collection(object):
    def __init__(self, ds_trajectories: xr.Dataset) -> None:
        """Create a collection of trajectories. Currently, trajectory collections are not allowed to be completely unstructured, but rather the starting points must form a regular grid.
        This choice facilitates lookups of trajectories and allows for interpolation to arbitrary points collections.

        Parameters
        ----------
        ds_trajectories : xr.Dataset
            Dataset with coordinates of the trajectories. Must have variables x,y,z and coordinates 'x0', 'y0', 'z0' and T, where T is the time relative to the starting position.
        """
        if not {'x', 'y', 'z'} == set(ds_trajectories.data_vars.keys()):
            raise DimensionError("Trajectory collections must have variables 'x', 'y', 'z'.")
        if not {'x0', 'y0', 'z0', 'T'} <= set(ds_trajectories.coords.keys()):
                raise DimensionError("Trajectories must have 'x0', 'y0', 'z0', 'T' coordinates.")
        self.ds=ds_trajectories
    
    def concat(self, other: Trajectory_Collection):
        ds=xr.concat([self.ds, other.ds], dim='T').sortby('T')
        return Trajectory_Collection(ds)


    @classmethod
    def from_flowfield(cls, flowfield: FlowField_Collection,x0:np.ndarray | float , y0: np.ndarray | float, z0: np.ndarray | float, advector: Advector):
        da_x0=np.array(x0, dtype=float).flatten()
        da_y0=np.array(y0, dtype=float).flatten()
        da_z0=np.array(z0, dtype=float).flatten()
        da_x0=xr.DataArray(da_x0, coords=[('x0',da_x0)])
        da_y0=xr.DataArray(da_y0, coords=[('y0',da_y0)])
        da_z0=xr.DataArray(da_z0, coords=[('z0',da_z0)])
        da_t0=xr.DataArray([0.0], coords=[('t0', [0.0])])
        X,Y,Z,T=advector.advect(flowfield, da_x0, da_y0, da_z0, da_t0) #result: [x0, y0, z0, it]
        # check if all particles have the same time
        #this should be the case for simple euler advection, but may not be fulfilled for more complex advection schemes
        min_t=T.min(dim=[d for d in T.dims if d!='it'])
        max_t=T.max(dim=[d for d in T.dims if d!='it'])
        if abs(min_t-max_t).any()>1e-6:
            raise ValueError("All particles must follow the same time evolution in a trajectory collection.")
        #now, we can add absolute time coordinates to the time index dimension
        times=T.isel({d:0 for d in T.dims if d!='it'}) #take the first time, since all times are equal
        ds_trajectories=xr.Dataset({'x': X, 'y': Y, 'z': Z})
        ds_trajectories.coords['it']=('it', times.values)
        ds_trajectories=ds_trajectories.rename(it='T').drop_vars('t0') 
        return cls(ds_trajectories)
        # forward=cls(ds_trajectories)
        # if steps_backward>0:
        #     backward=cls.from_flowfield(flowfield, x0, y0, z0, -1*dt, steps=steps_backward, savesteps=savesteps, interp_method=interp_method)
        #     backward=cls(backward.ds.drop_sel(T=0.0))
        #     forward=forward.concat(backward)
        # return forward
    
    def __repr__(self):
        return self.ds.__repr__()


class Path_Collection(object):
    def __init__(self, ds):
        if not {'n', 'it'}==set(ds.dims):
            raise DimensionError(f"Path Collection dataset has only dimensions 'n' and 'it' allowed. Dimensions are {ds.dims}.")
        if not {'x', 'y', 'z', 't'} <= set(ds.data_vars.keys()):
            raise DimensionError("Path collections must have variables 'x', 'y', 'z', 't'.")
        valid=ds.notnull()
        reduced=np.logical_and(valid.x, valid.y)
        reduced=np.logical_and(reduced, valid.z)
        reduced=np.logical_and(reduced, valid.t)
        self.ds=ds.where(reduced)
        

    @classmethod
    def from_trajectory_collection(cls, particle_coll: Particle_Collection, trajectory_coll: Trajectory_Collection, t: np.ndarray, matching: str='hybrid'):
        t=np.array(t).flatten()
        t=xr.DataArray(t, coords=[('t', t)])
        T=t-particle_coll.t0 #relative time
        if matching=='exact':
            selector=particle_coll.field_selectors | {'T':T, 'x0': particle_coll.x0, 'y0':particle_coll.y0, 'z0':particle_coll.z0}
            ds=trajectory_coll.ds.sel(selector).drop_vars(['x0', 'y0', 'z0', 'T'])
        elif matching=='interpolate':
            #Known bugs: interpolations do not work if we have only one value along a dimension.
            #Solution: implement interpolation only on >1 dimensions
            if 1 in trajectory_coll.x.shape:
                raise DimensionError("Interpolation matching is currently not working if any of the dimensions has length one.")
            selector=particle_coll.field_selectors | {'T':T, 'x0': particle_coll.x0, 'y0':particle_coll.y0, 'z0':particle_coll.z0}
            ds=trajectory_coll.ds.interp(selector).drop_vars(['x0', 'y0', 'z0', 'T'])
        elif matching=='hybrid': #select initial location exactly, interpolate time and additional selectors like fallspeed
            #TODO: After selection, the array has dimensions [n,T,...]. Therefore, the 'n' is interpolated again in the interpolation.
            #Currently, there seems to be no way in xarray to to a hybrid (partly exact, partly interpolate) selection in one go
            interpolation_selector=particle_coll.field_selectors | {'T':T} 
            ds=trajectory_coll.ds.sel(x0=particle_coll.x0,y0=particle_coll.y0, z0=particle_coll.z0).squeeze(drop=True).interp(interpolation_selector).drop_vars(['x0', 'y0', 'z0', 'T'], errors='ignore') #squeeze out 'n' if it has length one, since otherwise, the interpolation in 'n' will fail
        else:
            raise KeyError(f"Matching '{matching}' not available.")
        ds=ds.rename(t='it') #here, coordinates are actually absolute times, but path collections are more general and just require a general time index 'it'
        ds['t']=ds.it.broadcast_like(ds.x)
        return cls(ds)
    
    @classmethod
    def from_flowfield_collection(cls, particle_coll: Particle_Collection, field_coll: Field_Collection, advector: Advector):
        """Create a path collection from initial particle positions and a field collection.
            In this case, the fields can be time dependent and the paths are calculated explicitly for each particle. For stationary fields, it is usually more efficient to calculate trajectories first and then create the path collection from the trajectory collection.
        """
        X,Y,Z,T=advector.advect(field_coll, particle_coll.x0, particle_coll.y0, particle_coll.z0, particle_coll.t0)
        ds=xr.Dataset({'x': X, 'y': Y, 'z': Z, 't': T})
        return cls(ds)




    @classmethod
    def from_doppler_birdbath_collection(cls,bb_coll: Birdbath_Collection, traj_coll: Trajectory_Collection, t:xr.DataArray=None):
        part_coll=Particle_Collection.from_doppler_birdbath_collection(bb_coll)
        if t is None:
            t=bb_coll.moment.t
        return cls.from_trajectory_collection(part_coll, traj_coll,t)
    
    def get_valid_mask(self):
        valid=self.ds.x.notnull() #it is enough to check x, since by construction we equalize nans between x,y,z
        return valid

    def sel_nearest_time(self, time, tolerance=None):
        """For every particle, choose the nearest time step in the dataset.
        This is useful, since paths are stored in format (n, it), so selecting an absolute time is not straigtforward.
        """
        #maybe, in the future this can be acomplisehd with NDPointIndex, but currently, the tolerance is not supported for sel() calls with this index.
        nearest_it=abs(self.ds.t-time).argmin('it')
        result=self.ds.isel(it=nearest_it)
        if tolerance is not None:
            valid=abs(self.ds.t-time).min('it')<=tolerance
            result=result.where(valid)
        return result


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

    def fill_with(self, part_coll: Particle_Collection, path_coll: Path_Collection, aggregation='max'):
        #set all positions outside of the field to invalid values
        id=self._get_id(path_coll.ds)

        #Add the combined index as a multi index to the particle property field and use groupby
        particle_property=part_coll.property.broadcast_like(path_coll.ds) #add the time dimension
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
