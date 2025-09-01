# Cuda NPP excercise

A piece of code to practice with Nvidia NPP libraries.
The code reads an image, reduces the number of colours, and then executes a number of cellular automaton steps on it.

## Build
Create and activate the cuda conda environment in `envs/cuda.source.yaml`.
Then build the code with nvcc. (I run and build the code on an LSF compute cluster, hence the bsub command)
```
  conda env update -f ./envs/cuda.source.yaml
  conda activate cuda
  bsub -q bio-gpu-m10 "nvcc ca.cpp -I/usr/local/cuda-12.9/targets/x86_64-linux/include -I./utils -I./utils/UtilNPP -L/usr/local/cuda-12.9/targets/x86_64-linux/lib -L$CONDA_PREFIX/lib -lfreeimage -lnppc -lnppial -lnppicc -lnppif -lnppig -lnppim -lnppist -lnppisu -lnppitc -lnpps -o ca"
```
Run on the compute cluster with
```
  bsub -q bio-gpu-m10 -o cl.out -e cl.err "./ca --input ./examples/5.2.10.tiff --steps 10 > ca.out 2> ca.err"
```

## Example
The example in the examples folder is from https://sipi.usc.edu/database/database.php?volume=misc&image=22#top