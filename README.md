# Ising Model

Implement in CUDA the evolution of an Ising model in two dimensions for a given number of steps k.
For more information about the Ising model visit the above [link](https://en.wikipedia.org/wiki/Ising_model)
To execute the produced executable files Nvidia GPU is necessary. Parameters were chosen for optimal performance on Nvidia Tesla P100

## Installation

Clone the repository

```bash
git clone https://github.com/vagzikopis/IsingCuda.git
```

## Usage

### Local Validation Test (Nvidia GPU prerequisite)
```bash
cd src
make
./validate_v1 or ./validate_v2 or ./validate_v3
```
### Local Execution Time Test (Nvidia GPU prerequisite)
```bash
cd src
make
./v1 $n $k or ./v2 $n $k or ./v3 $n $k
```
### Aristotle University HPC Validation Test
```bash
cd src
make
cd ..
cd batch
sbatch validate_v1.sh or sbatch validate_v2.sh or sbatch validate_v3.sh
```
### Aristotle University Time Test (Nvidia GPU prerequisite)
To edit parameters n(square lattice axis) and k(iterations), edit v1.sh or v2.sh or v3.sh.
Results are printed on slurm file. Read results with command ```tail -f slurm``` .
```bash
cd src
make
cd ..
cd batch
sbatch v1.sh or sbatch v2.sh or sbatch v3.sh
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
