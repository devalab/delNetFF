# Machine Learning Force Field for Argon
Machine Learning for Accurate Force Calculations in Molecular Dynamics Simulation

### Setting it up
##### Building the cython files
The cython package needs to be built once before they can be used. To do this, go to the folder located inside the `cython_files` folder and execute the command given in `command_to_build.txt`.
```
cd cython_files/pbc_adjustment;
python setup.py build_ext --inplace
cd -;
```

##### Installing the required python packages
Using a fresh virtual environment using an environment manager like [Conda](https://docs.conda.io/en/latest/) is highly recommended before installing the python packages. Run `conda env create -f environment.yml` if using conda to get an environment with the required packages installed. Additionally you may need to configure CUDA on your setup in order for tensorflow to use the GPU.

### Running a simulation
##### Classical

```
python simulation.py --mode=classical --save_traj_at_end=False --write_pos_only=True --also_save_npz=True
```
##### Classical along with forces from the model
```
python simulation.py --mode=classical-and-predictor --model_folder=saved_weights/weights --save_traj_at_end=False --write_pos_only=True --also_save_npz=True
```
 - `--model_folder`: Folder containing the saved weights
 - `--mode`: Either `classical` or `classical-and-predictor`. Specifies whether a classical only simulation or a classical + model forces simulation is to be run.
 - `--save_traj_at_end`: Either `true` or `false`. Setting this to true will save the whole trajectory at the end, while storing it in the RAM while the simulation is being run. Setting this to false will flush the trajectory at regular intervals from the RAM to disk. This should be set to false for large systems or longer trajectories as the whole trajctory might not fit in RAM.
 - `--write_pos_only`: Either `true` or `false`. If true, only the positions trajectory is saved. If false, positions, velocities and forces will be saved.
 - `--also_save_npz`: Either `true` or `false`. If true, also saves all traj files as npz at the end(converts the .xyz files to npz). If false, just xyz files are saved. npz files are written at the end, so they wont be written partially if the simulation crashes.


