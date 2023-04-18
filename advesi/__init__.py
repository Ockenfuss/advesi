from __future__ import annotations
from typing import Dict
import numpy as np
import xarray as xr

FIELD_BOUNDARY=1e10

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
        u=xr.DataArray(u)
        v=xr.DataArray(v)
        w=xr.DataArray(w)
        u=self._add_missing_dimensions(u)
        v=self._add_missing_dimensions(v)
        w=self._add_missing_dimensions(w)
        self.u=u
        self.v=v
        self.w=w

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
        """If one of x,y,z is missing in the input velocity components, add it as a coordinate from -advesi.FIELD_BOUNDARY to advesi.FIELD_BOUNDARY."""
        for d in ["x", "y","z"]:
            if d not in da.dims:
                coord=xr.DataArray([-FIELD_BOUNDARY, FIELD_BOUNDARY], coords=[(d, [-FIELD_BOUNDARY, FIELD_BOUNDARY])])
                da=da.broadcast_like(coord)
        return da
    
    def _broadcast_array_to_field(self,da):
        return _broadcast_like_list(da, self.u, self.v, self.w, exclude=["x", "y","z"])
        

class Particle_Collection(object):
    def __init__(self, x0: float | xr.DataArray, y0: float | xr.DataArray, z0: float | xr.DataArray, t0: float | xr.DataArray, property: float | xr.DataArray, field_selectors: Dict[str, xr.DataArray]={}):
        x0=xr.DataArray(x0)
        y0=xr.DataArray(y0)
        z0=xr.DataArray(z0)
        t0=xr.DataArray(t0)
        property=xr.DataArray(property)
        field_selectors={k:xr.DataArray(v) for k,v in field_selectors.items()}
        #first, broadcast all given arrays
        select_keys=list(field_selectors.keys())
        select_arrays=list(field_selectors.values())
        x0,y0,z0,t0,property,*select_arrays=xr.broadcast(x0,y0,z0,t0,property,*select_arrays)
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
    def from_doppler_birdbath_collection(cls, bb_coll: Birdbath_Collection):
        return cls(bb_coll.xr, bb_coll.yr, bb_coll.moment.z, bb_coll.moment.t, bb_coll.moment, field_selectors={'w':bb_coll.moment})
    
    @classmethod
    def from_field_collection(cls, field_coll: Field_Collection, field_selectors: Dict[str, xr.DataArray]={}):
        return cls(field_coll.f.x, field_coll.f.y, field_coll.f.z, field_coll.f.t, property=field_coll.f, field_selectors=field_selectors)
        

class Trajectory_Collection(object):
    def __init__(self, x,y,z) -> None:
        """Create a collection of trajectories. Currently, trajectory collections are not allowed to be completely unstructured, but rather the starting points must form a regular grid.
        This choice facilitates lookups of trajectories and allows for interpolation to arbitrary points collections.

        Parameters
        ----------
        x : xr.DataArray
            x coordinate of the trajectories. Must have coordinates 'x0', 'y0', 'z0' and T, where T is the time relative to the starting position.
        y : xr.DataArray
            y coordinate of the trajectories. Must have coordinates 'x0', 'y0', 'z0' and T, where T is the time relative to the starting position.
        z : xr.DataArray
            z coordinate of the trajectories. Must have coordinates 'x0', 'y0', 'z0' and T, where T is the time relative to the starting position.
        """
        for d in [x,y,z]:
            if not {'x0', 'y0', 'z0', 'T'} <= set(d.coords.keys()):
                raise DimensionError("Trajectories must have 'x0', 'y0', 'z0', 'T' coordinates.")
        for d in ['x0', 'y0', 'z0', 'T']: #TODO: This check is incomplete. In theory, x,y,z must be able to form one commont DataArray
            if not (x[d].equals(y[d]) and x[d].equals(z[d])):
                raise DimensionError(f"Coordinate {d} is not equal on all of the given arrays!")
        self.T=x.T
        self.x=x
        self.y=y
        self.z=z

    @classmethod
    def from_flowfield(cls, flowfield: FlowField_Collection,x0:np.ndarray | float , y0: np.ndarray | float, z0: np.ndarray | float, dt: float, steps, savesteps=1, interp=False):
        x0=np.array(x0).flatten()
        y0=np.array(y0).flatten()
        z0=np.array(z0).flatten()
        x0=xr.DataArray(x0, coords=[('x0',x0)])
        y0=xr.DataArray(y0, coords=[('y0',y0)])
        z0=xr.DataArray(z0, coords=[('z0',z0)])
        T=np.arange(0,steps*dt, dt*savesteps)
        T=xr.DataArray(T,coords=[('T', T)])
        X,Y,Z,_=xr.broadcast(x0,y0,z0,T)
        X=flowfield._broadcast_array_to_field(X).copy()
        Y=flowfield._broadcast_array_to_field(Y).copy()
        Z=flowfield._broadcast_array_to_field(Z).copy()
        Xi=X.isel(T=0).copy().drop_vars("T") #current positions in iteration scheme
        Yi=Y.isel(T=0).copy().drop_vars("T")
        Zi=Z.isel(T=0).copy().drop_vars("T")

        for iT in range(len(T)):
            X[{"T":iT}]=Xi
            Y[{"T":iT}]=Yi
            Z[{"T":iT}]=Zi
            for it in range(savesteps):
                if interp:
                    Xi=Xi+flowfield.u.interp(x=Xi, y=Yi, z=Zi, kwargs={'fill_value':None}).drop(["x","y","z"])*dt
                    Yi=Yi+flowfield.v.interp(x=Xi, y=Yi, z=Zi, kwargs={'fill_value':None}).drop(["x","y","z"])*dt
                    Zi=Zi+flowfield.w.interp(x=Xi, y=Yi, z=Zi, kwargs={'fill_value':None}).drop(["x","y","z"])*dt
                else:
                    Xi=Xi+flowfield.u.sel(x=Xi, y=Yi, z=Zi, method='nearest').drop(["x","y","z"])*dt
                    Yi=Yi+flowfield.v.sel(x=Xi, y=Yi, z=Zi, method='nearest').drop(["x","y","z"])*dt
                    Zi=Zi+flowfield.w.sel(x=Xi, y=Yi, z=Zi, method='nearest').drop(["x","y","z"])*dt
        return cls(X,Y,Z)

class Path_Collection(object):
    def __init__(self, x,y,z):
        for d in [x,y,z]:
            if not {'n', 't'}==set(d.dims):
                raise DimensionError("Path Collection fields have only dimensions 'n' and 't' allowed.")
        valid=np.logical_and(x.notnull(), y.notnull())
        valid=np.logical_and(valid, z.notnull())
        self.x=x.where(valid)
        self.y=y.where(valid)
        self.z=z.where(valid)
        

    @classmethod
    def from_particle_collection(cls, particle_coll: Particle_Collection, trajectory_coll: Trajectory_Collection, t: np.ndarray, matching: str='hybrid'):
        t=np.array(t).flatten()
        t=xr.DataArray(t, coords=[('t', t)])
        T=t-particle_coll.t0 #relative time
        if matching=='exact':
            selector=particle_coll.field_selectors | {'T':T, 'x0': particle_coll.x0, 'y0':particle_coll.y0, 'z0':particle_coll.z0}
            x=trajectory_coll.x.sel(selector).drop_vars(['x0', 'y0', 'z0', 'T'])
            y=trajectory_coll.y.sel(selector).drop_vars(['x0', 'y0', 'z0', 'T'])
            z=trajectory_coll.z.sel(selector).drop_vars(['x0', 'y0', 'z0', 'T'])
            return cls(x,y,z)
        elif matching=='interpolate':
            #Known bugs: interpolations do not work if we have only one value along a dimension.
            #Solution: implement interpolation only on >1 dimensions
            if 1 in trajectory_coll.x.shape:
                raise DimensionError("Interpolation matching is currently not working if any of the dimensions has length one.")
            selector=particle_coll.field_selectors | {'T':T, 'x0': particle_coll.x0, 'y0':particle_coll.y0, 'z0':particle_coll.z0}
            x=trajectory_coll.x.interp(selector).drop_vars(['x0', 'y0', 'z0', 'T'])
            y=trajectory_coll.y.interp(selector).drop_vars(['x0', 'y0', 'z0', 'T'])
            z=trajectory_coll.z.interp(selector).drop_vars(['x0', 'y0', 'z0', 'T'])
            return cls(x,y,z)
        elif matching=='hybrid': #select initial location exactly, interpolate time and additional selectors like fallspeed
            #TODO: After selection, the array has dimensions [n,T,...]. Therefore, the 'n' is interpolated again in the interpolation.
            #Currently, there seems to be no way in xarray to to a hybrid (partly exact, partly interpolate) selection in one go
            interpolation_selector=particle_coll.field_selectors | {'T':T} 
            x=trajectory_coll.x.sel(x0=particle_coll.x0,y0=particle_coll.y0, z0=particle_coll.z0).squeeze(drop=True).interp(interpolation_selector).drop_vars(['x0', 'y0', 'z0', 'T'], errors='ignore') #squeeze out 'n' if it has length one, since otherwise, the interpolation in 'n' will fail
            y=trajectory_coll.y.sel(x0=particle_coll.x0,y0=particle_coll.y0, z0=particle_coll.z0).squeeze(drop=True).interp(interpolation_selector).drop_vars(['x0', 'y0', 'z0', 'T'], errors='ignore')
            z=trajectory_coll.z.sel(x0=particle_coll.x0,y0=particle_coll.y0, z0=particle_coll.z0).squeeze(drop=True).interp(interpolation_selector).drop_vars(['x0', 'y0', 'z0', 'T'], errors='ignore')
            return cls(x,y,z)
        else:
            raise KeyError(f"Matching '{matching}' not available.")


    @classmethod
    def from_doppler_birdbath_collection(cls,bb_coll: Birdbath_Collection, traj_coll: Trajectory_Collection, t:xr.DataArray=None):
        part_coll=Particle_Collection.from_doppler_birdbath_collection(bb_coll)
        if t is None:
            t=bb_coll.moment.t
        return cls.from_particle_collection(part_coll, traj_coll,t)
    
    def get_valid_mask(self):
        valid=self.x.notnull() #it is enough to check x, since by construction we equalize nans between x,y,z
        return valid


class Field_Collection(object):
    def __init__(self, f) -> None:
        if not {'x', 'y', 'z', 't'}==set(f.dims):
            raise DimensionError("A field collection must have dimensions 'x', 'y', 'z' and 't'.")
        self.f=f.sortby(['x', 'y', 'z', 't'])
        self.index_x=xr.DataArray(np.arange(len(f.x)), coords=[('x', f.x.values)])
        self.index_y=xr.DataArray(np.arange(len(f.y)), coords=[('x', f.y.values)])
        self.index_z=xr.DataArray(np.arange(len(f.z)), coords=[('x', f.z.values)])
        self.index_ft=xr.DataArray(np.arange(len(f.t)), coords=[('t', f.t.values)])

    
    def _get_index_for_values(self, values: xr.DataArray, indexes: xr.DataArray):
        coordname=indexes.dims[0]
        #check if values are within bounds of given index mapping
        assert(values.min()>=indexes[coordname].min())
        assert(values.max()<=indexes[coordname].max())
        #create an additional index '-1' in the index array outside the range of given values
        invalid_position=indexes[coordname].min().item()-1.0
        indexes=xr.concat([indexes, xr.DataArray([-1], coords=[(coordname, [invalid_position])])], dim=coordname).sortby(coordname)
        #fill nan values with the invalid value, pointing to index '-1'
        values=values.fillna(invalid_position)
        #for every position, look for the corresponding index
        indexes_positions=indexes.sel({coordname:values}, method='ffill')
        return indexes_positions
    
    @classmethod
    def create_regular(cls, times: np.ndarray, xlim: tuple, ylim: tuple, zlim: tuple, nx=100, ny=100, nz=100):
        #create field axes, big enough to hold all paths completely
        x=np.linspace(xlim[0], xlim[1], nx)
        x=xr.DataArray(x, coords=[('x', x)])
        y=np.linspace(ylim[0], ylim[1], ny)
        y=xr.DataArray(y, coords=[('y', y)])
        z=np.linspace(zlim[0], zlim[1], nz)
        z=xr.DataArray(z, coords=[('z', z)])
        #Time axis is given as argument
        ft=times.flatten()
        ft=xr.DataArray(ft, coords=[('t', ft)])
        field_da=xr.full_like(xr.broadcast(x,y,z,ft)[0],fill_value=np.nan)
        field=cls(field_da)
        return field

    @classmethod
    def from_particle_collection(cls, part_coll: Particle_Collection, traj_coll: Trajectory_Collection, times: np.ndarray, nx=100, ny=100, nz=100, aggregation='max'):
        path_coll=Path_Collection.from_particle_collection(part_coll, traj_coll, times)
        xlim=(path_coll.x.min().item(), path_coll.x.max().item())
        ylim=(path_coll.y.min().item(), path_coll.y.max().item())
        zlim=(path_coll.z.min().item(), path_coll.z.max().item())
        f=cls.create_regular(times, xlim, ylim, zlim, nx, ny, nz)
        f.fill_with(part_coll, path_coll, aggregation)
        return f

    def fill_with(self, part_coll: Particle_Collection, path_coll: Path_Collection, aggregation='max'):
        #set all positions outside of the field to invalid values
        isinfield=  (path_coll.x>=self.f.x.min()) & (path_coll.x<=self.f.x.max()) & \
                    (path_coll.y>=self.f.y.min()) & (path_coll.y<=self.f.y.max()) & \
                    (path_coll.z>=self.f.z.min()) & (path_coll.z<=self.f.z.max()) & \
                    (path_coll.x.t>=self.f.t.min()) & (path_coll.x.t<=self.f.t.max())
        valid=path_coll.get_valid_mask()
        #For every position in the path collection, find the corresponding index in the field
        particle_index_x=self._get_index_for_values(path_coll.x.where(isinfield), self.index_x)
        particle_index_y=self._get_index_for_values(path_coll.y.where(isinfield), self.index_y)
        particle_index_z=self._get_index_for_values(path_coll.z.where(isinfield), self.index_z)
        path_times=path_coll.x.t.where(valid) #time is not a variable, but an axis. This will broadcast it and set to nan where the positions are invalid
        particle_index_ft=self._get_index_for_values(path_times.where(isinfield), self.index_ft)

        nix=len(self.index_x)
        niy=len(self.index_y)
        niz=len(self.index_z)
        nit=len(self.index_ft)

        def ravel_multi_index(ix, iy, iz, it, nx, ny, nz, nt):
            """Ravel index in C style order. In contrast to np.ravel_multi_index, this function does not raise an error for negative indices."""
            return ix*ny*nz*nt+iy*nz*nt+iz*nt+it
        def unravel_index(ind:xr.DataArray, nx, ny, nz, nt):
            """xarray wrapper around np.unravel_index with C style order."""
            unravel=lambda da: np.unravel_index(da, shape=(nx, ny, nz, nt), order='C')
            indx, indy, indz, indt=xr.apply_ufunc(unravel, ind, output_core_dims=[[],[],[], []])
            return indx, indy, indz, indt

        #create a combined index for every spation-temporal index
        index_n=ravel_multi_index(particle_index_x, particle_index_y, particle_index_z, particle_index_ft, nix, niy, niz, nit)

        #Add the combined index as a multi index to the particle property field and use groupby
        particle_property=part_coll.property.broadcast_like(path_coll.x.t) #add the time dimension
        particle_property.coords["index_n"]=(index_n.dims, index_n.values)
        particle_property=particle_property.groupby('index_n')
        if aggregation=='max':
            particle_property=particle_property.max()
        elif aggregation=='mean':
            particle_property=particle_property.mean()
        elif aggregation=='median':
            particle_property=particle_property.median()
        else:
            raise ValueError(f"Aggregation '{aggregation}' is currently not implemented.")
        #remove the group, which contains all invalid spatio-temporal particle positions
        particle_property=particle_property.where(particle_property.index_n>=0, drop=True)
        property_index_x, property_index_y, property_index_z, property_index_ft=unravel_index(particle_property.index_n, nix, niy, niz, nit)
        self.f[{"x":property_index_x,"y":property_index_y, "z":property_index_z, "t":property_index_ft}]=particle_property.values



class Birdbath_Collection(object):
    def __init__(self) -> None:
        # self.moment=
        pass

