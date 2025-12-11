#%%
import numpy as np
import xarray as xr
import advesi as adv
import matplotlib.pyplot as plt
from pathlib import Path
# %%
zmin=0
zmax=4
umin=10
umax=50 # m/s
u=xr.DataArray(np.linspace(umin, umax, 10), coords=[('z', np.linspace(zmin, zmax, 10))]).astype(float)
u=u/1000 # convert to km/s
u.plot(y='z')
w=-1.0
w=w/1000 # convert to km/s
#%%
ff=adv.FlowField_Collection(u,0,w)
advector=adv.EulerAdvector(dt=30, steps=200)
traj_coll=adv.Trajectory_Collection.from_flowfield(ff, x0=0.0, y0=0.0, z0=zmax, t0=0.0, advector=advector)

n_particles=100
generating_cell_positions=np.linspace(-200,300,n_particles)
generating_cell_times=generating_cell_positions / u.isel(z=-1).values
part_coll=adv.Particle_Collection(generating_cell_positions,0.0,zmax,generating_cell_times,1.0)
path_coll=adv.Path_Collection.from_trajectory_collection(part_coll, traj_coll, matching='interpolate', substeps=10)
#%%
fig, ax=plt.subplots()
for n in np.arange(0, n_particles, 10):
    line_traj=ax.plot(path_coll.ds.x.isel(n=n), path_coll.ds.z.isel(n=n), ls='--', color='gray', alpha=0.5)

# select some timesteps to plot
n_select=40
plot_times=np.linspace(0, 6000,10)
plot_times=xr.DataArray(plot_times, coords=[('plot_time', plot_times)])
path_coll_t=path_coll.interp_to_time(plot_times)
# select one particle to highlight
path_coll_single=path_coll_t.isel(n=n_select)
line_single=ax.plot(path_coll_single.x, path_coll_single.z, 'o', color='blue')
# Optionally: draw other particles as well
# ax.scatter(path_coll_t.x.values.flatten(), path_coll_t.z.values.flatten(), color='grey', alpha=0.1)

# position of all particles at selected instants in time
for t in plot_times.values:
    paths_t=path_coll_t.sel(plot_time=t)
    line_all=ax.plot(paths_t.x.values, paths_t.z.values, '-', color='red', alpha=0.3)

# format plot
ax.set_xlim(0, 270)
ax.set_ylim(zmin, zmax)
ax.set_xlabel('distance (km)')
ax.set_ylabel('depth (km)')
ax.set_title('Particle Advection example, similar to Marshall (1953)')
ax.legend([line_single[0], line_all[0], line_traj[0]], ['selected particle', 'all particles at fixed time', 'particle trajectories'], loc='upper right')
# %%
fig.savefig('Marshal1953_example.png', dpi=300)
