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
* A `birdbath scan` is a scalar field in time-height space, like measured by a vertically looking radar or lidar. It has an associated x and y position. Multiple birdbath scans can be stored in a `Birdbath_Collection`.

Two comments to this structure:

* **Trajectory vs Path**: The distinction between path and trajectory is useful for performance reasons: In a static field, the trajectory of a particle at a given starting position is always the same, regardless of time. Therefore, the expensive integration of the trajectory has to be done only once. Afterwards, this trajectory can be applied to every particle at the starting position, regardless of the absolute time.
* **Fall Speed**: A particle has no separate fall speed associated with it. Instead, a lighter particle 'sees' a different velocity field than a heavier one. Therefore, every particle has a general `field_selector` property, which connect each particle to a certain flowfield in a Field_Collection.


