# Advesi- Advection Simulator

When working with radar data, especially time-height displays ("Birdbath Scans"), two common questions arise: 
* Given an atmospheric structure, how does the radar image look like?  
* Given a radar image, how does the atmospheric structure look like? 

Advesi is a module to answer such questions. It is based on a lagrangian perspective, solving the advection equation for a collection of particles in a static velocity field, using a simple discrete integration. Thereby, advesi provides functions to create a collection of particles from radar images as well as to return a simulated radar image form particle collections. Internally, advesi is using numpy or xarray operations almost everywhere, making it very fast even for large arrays/particle collections.

## Terminology
The following represents the core concepts of advesi:
* A `flowfield` is a 3D vector field in 3D space, representing the wind which transports the particles. Currently, flowfields must be static fields, i.e. without time dependence. A `Flowfield_Collection` object contains a collection of flowfields.
* A `Particle_Collection` is an unstructured collection of particles. Each particle is defined by its coordinates (3 in space + time). Additionally, it can have a `property`, which is an arbitrary value associated with it.
* A `trajectory` is a 3D path in space, starting at a specific initial position and following the streamlines of the flowfield. A `Trajectory_Collection` can store multiple trajectories, e.g. for different initial positions.
* A `path` is like a trajectory, but with an absolute time reference. A `Path_Collection` is a collection of paths, usually one path for every particle in a Particle_Collection.
* A `field` is a scalar field in 3D space. Multiple field, for example at different times, can be stored in a `Field_Collection`.
* A `birdbath scan` is a special field with high resolution in time and just one cell in x and y direction. This is what is measured by a vertically looking radar or lidar.

Two comments to this structure:

* **Trajectory vs Path**: The distinction between path and trajectory is useful for performance reasons: In a static field, the trajectory of a particle at a given starting position is always the same, regardless of time. Therefore, the expensive integration of the trajectory has to be done only once. Afterwards, this trajectory can be applied to every particle at the starting position, regardless of the absolute time.
* **Fall Speed**: A particle has no separate fall speed associated with it. Instead, a lighter particle 'sees' a different velocity field than a heavier one. Therefore, every particle has a general `field_selector` property, which connect each particle to a certain flowfield in a Field_Collection.

## Usage
At first, you need to define the flowfield. You can do this by providing xarray DataArrays for the u, v, and w components of the field.
If `x`, `y` or `z` are missing as dimensions, they are added, extending to infinity (+-adv.FIELD_BOUNDARY). If additional dimensions are present, they are interpreted as multiple, independent fields.
```python
import numpy as np
import xarray as xr
import advesi as adv
u=xr.DataArray(np.arange(10)[::-1], coords=[('z', np.arange(10))]).astype(float)
w=xr.DataArray([-1,-2,-3], coords=[('w', [-1,-2,-3])]).astype(float) #Here, we make a collection of fields with different vertical velocities
ff=adv.FlowField_Collection(u,0,w) #scalars are allowed as well
```

Next, you need to define a set of particles. For every particle, you need to define the initial position and time. In this case, we also provide a `field_selector`, associating every particle with a specific field in our field collection.
```python
part_coll=adv.Particle_Collection([0.0,0.0,0.0],0.0,9.0,0.0,[-1,-2,-3], field_selectors={'w':[-1,-2,-3]})
```

Now, we calculate a set of trajectories, ideally at every position where we placed a particle. This serves as a lookup table to create the paths of the particles:
```python
traj_coll=adv.Trajectory_Collection.from_flowfield(ff, 0.0,0.0,9.0,0.1,100,1)
```

Using the trajectories, we create a collection of paths. Therefore, we need to specify a vector of discrete, absolute timesteps. For every particle, the trajectory collection (with relative times) will be interpolated to these timesteps and the particle starting position if `matching='interpolate'` is specified. If `matching='exact'`, a direct lookup is perfomed. If `matching='hybrid'`, particle starting positions are looked up directly, timesteps are interpolated.
```python
path_coll=adv.Path_Collection.from_particle_collection(part_coll, traj_coll, t=path_times, matching='hybrid')
```
Now, with the paths of very particle available, we can define arbitrary fields as output and group the paricles into the cells of the output fields. Every field collection needs to have `x`, `y`, `z` and `t` dimensions.
```python
field_times=np.linspace(0.0,10.0,5)
field=adv.Field_Collection.create_regular(field_times, [0,20], [0,0], [0,9], 100, 1,50) #create a regularly spaces field
```
Once you have a field defined, you can fill it with the particles. If multiple particles are in the same field cell at the same time, you need to tell the function how to combine the values of the particles in this cell to a single value in the field. This is done by the `aggregation` argument.
```python
field.fill_with(part_coll, path_coll, aggregation='max') #fill the empty field with values from the particles
f=field.f.squeeze() #extract the underlying DataArray
f.max('t').plot(x='x') #plot it
```

For more examples, see the provided jupyter notebook.

## Data Structures
Internally, the data is structured as follows:
- Flowfield_Collection: `ds:u,v,w` with dimension `x,y,z,t`
- Particle_Collection: Properties `x0,y0,z0,t0,property,selectors` with dimension `n`
- Trajectory_Collection: `ds:x,y,z` with dimensions `x0,y0,z0,T`
- Path_Collection: `ds:x,y,z,t` with dimensions `n,it`
- Field_Collection: `ds:f` with dimensions `x,y,z,t`