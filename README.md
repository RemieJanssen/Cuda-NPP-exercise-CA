# Cuda NPP excercise

A piece of code to practice with Nvidia NPP libraries.
The code reads an image, reduces the number of colours, and then executes a number of cellular automaton steps on it.

## Build
Create and activate the cuda conda environment in `envs/cuda.source.yaml`.
Then build the code with nvcc.
```
  conda env update -f ./envs/cuda.source.yaml
  conda activate cuda
  nvcc ca.cpp -I$CONDA_PREFIX/targets/x86_64-linux/include -I./utils -I./utils/UtilNPP -L$CONDA_PREFIX/targets/x86_64-linux/lib -o ca
  OR
  g++ -c ca.cpp -I$CONDA_PREFIX/targets/x86_64-linux/include -I./utils -I./utils/UtilNPP -L$CONDA_PREFIX/targets/x86_64-linux/lib -o ca
  chmod 755 ca
```