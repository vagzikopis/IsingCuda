# define the shell to bash
SHELL := /bin/bash

# define the C/C++ compiler to use
CC = gcc
NVCC = nvcc
EXECS = validate_v1 validate_v2 validate_v3 validate_sequential v1 v2 v3 sequential
.PHONY: $(EXECS)

all: $(EXECS)

validate_v1:
   	$(NVCC) /src/v1.cu validator.cu -o $@

validate_v2:
   	$(NVCC) /src/v2.cu validator.cu -o $@

validate_v3:
	$(NVCC) /src/v3.cu validator.cu -o $@

validate_sequential:
 	$(CC) /src/sequential.c validator.c -o $@ -std=c99

v1:
   	$(NVCC) /src/v1.cu main.cu -o $@

v2:
   	$(NVCC) /src/v2.cu main.cu -o $@

v3:
	$(NVCC) /src/v3.cu main.cu -o $@

sequential:
 	$(CC) /src/sequential.c main.c -o $@ -std=c99
