#-------------------- 
#CUDACODE = code1
#CUDACODE = code2
#CUDACODE = code3
#CUDACODE = code4
CUDACODE = code5
#CUDACODE = code6
#--------------------

CXX = g++
INCLUDES  = -I. -I$(CUDA_INSTALL_PATH)/include -I$(SDK_INSTALL_PATH)/common/inc
MAIN = main
TARGET = run.x

all: $(TARGET)


c: clean

clean : 
	rm -f *~ .*~ *.o
	rm -f $(CUDACODE).ptx


$(TARGET) : $(MAIN).o $(CUDACODE).o
	$(CXX) $(MAIN).o $(CUDACODE).o -o run.x \
        -lm -L$(CUDA_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/lib \
        -L$(CUDA_INSTALL_PATH)/../NVIDIA_GPU_Computing_SDK/CUDALibraries/common/lib \
	      -lcuda -lcudart -lGL -lGLU -lcutil_x86_64



$(MAIN).o : $(MAIN).cc
	$(CXX) -O3 -c $(MAIN).cc


$(CUDACODE).o : $(CUDACODE).cu
	nvcc  --host-compilation 'C++'   -Xcompiler "-march=core2" \
          -c $(CUDACODE).cu \
          -Xcompiler "-m64" -Xcompiler "-O3" -Xcompiler "-ffast-math" \
          --machine 64 -DUNIX -O3 $(INCLUDES)
	rm  -f kernel.o
	ln -s $(CUDACODE).o kernel.o

ptx : clean
	nvcc  --host-compilation 'C++' cunbody_lib.cu --ptx $(INCLUDES)


cubin:
	nvcc  --host-compilation 'C++' -cubin dummy_main.cu  $(INCLUDES)
	grep smem dummy_main.cubin
	grep reg dummy_main.cubin

export:
	tar cjvfps ../lib6.`date +"%Y%m%d-%H%M%S"`.tbz ../lib6


b: export

t: test

test:
	./run.x gen-plum/data.inp.016384

