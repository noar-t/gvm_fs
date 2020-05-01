test:
	nvcc *cu -rdc=true -gencode arch=compute_61,code=sm_61 -m64 -x cu 
