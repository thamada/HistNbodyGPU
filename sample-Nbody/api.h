//-------------------------
#define NMAX (1<<17)
#include <cstdio>
#include <cstdlib>
#define USE_CUDA_MALLOC_HOST (yes)
#define IDIM (4)
#define JDIM (4)
#define FDIM (3)
#define NJ_SHMEM (NTHRE)

float4* h_xj;
float* h_xi;
float* h_fo;
float4* d_xj;
float* d_xi;
float* d_fo;

extern "C" void 
force(double xj[][3], double mj[], double xi[][3], double eps2, double a[][3], int ni, int nj)
{
  static bool is_open = false;
  int devid = 0;
  int ndev;
  int nj1 = ((nj+NJ_SHMEM-1)/NJ_SHMEM)*NJ_SHMEM;

  if( ni > NMAX ){
    printf("ERROR %s|%d\n",__FILE__, __LINE__);
    printf(" ni > %d : ni=%d\n",ni, NMAX);
    exit(-1);
  }

  if( nj > NMAX ){
    printf("ERROR %s|%d\n",__FILE__, __LINE__);
    printf(" nj > %d : nj=%d\n",nj, NMAX);
    exit(-1);
  }

  unsigned int ip_size = sizeof(float) * ni * IDIM;
  unsigned int jp_size = sizeof(float4) * nj1;
  unsigned int fo_size = sizeof(float) * ni * FDIM;

  if(is_open == false){
    char gpu_name[256];
    CUDA_SAFE_CALL(cudaSetDevice(devid));
    CUDA_SAFE_CALL(cudaGetDeviceCount(&ndev));
    if(ndev == 0){
      fprintf(stdout, "ndev = %d @ %s|%d\n", ndev, __FILE__, __LINE__);
      exit(-1);
    }else{
      int dev = devid;
      cudaDeviceProp deviceProp;
      CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev));
      if (deviceProp.major == 9999 && deviceProp.minor == 9999){
	printf("There is no device supporting CUDA.\n");
      }
      fprintf(stderr, "  GPU : %s\n", deviceProp.name);
      sprintf(gpu_name,"%s", deviceProp.name);
    }

    unsigned int _nmax = NMAX;
    unsigned int _ip_size = sizeof( float) * _nmax * IDIM;
    unsigned int _jp_size = sizeof(float4) * _nmax;
    unsigned int _fo_size = sizeof(float) * _nmax * FDIM;

#if defined(USE_CUDA_MALLOC_HOST)
    CUDA_SAFE_CALL(  cudaMallocHost( (void**)&h_xj, _jp_size)  );
    CUDA_SAFE_CALL(  cudaMallocHost( (void**)&h_xi, _ip_size)  );
    CUDA_SAFE_CALL(  cudaMallocHost( (void**)&h_fo, _fo_size)  );
#else
    h_xj = (float4*) malloc(_jp_size);
    h_xi = (float*)  malloc(_ip_size);
    h_fo = (float*)  malloc(_fo_size);
#endif
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_xj, _jp_size));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_xi, _ip_size));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_fo, _fo_size));
    for(int i = 0; i < _nmax ; i++) h_xj[i] = make_float4(0.0, 0.0, 0.0, 0.0);
    CUDA_SAFE_CALL( cudaMemcpy( d_xj, h_xj, _jp_size, cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy( d_xi, h_xj, _ip_size, cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy( d_fo, h_xj, _fo_size, cudaMemcpyHostToDevice) );
    fprintf(stderr, "open %s by CUNBODY-1 library: rev.hamada20080905  (^<_^)/ %d\n", gpu_name, devid);
    is_open = true;
  }
  for(int i = 0; i < nj; i++){
    h_xj[i].x = (float) xj[i][0];
    h_xj[i].y = (float) xj[i][1];
    h_xj[i].z = (float) xj[i][2];
    h_xj[i].w = (float) mj[i];
    //    printf("h_xj:%d\t%e\t%e\t%e\t%e\n",i, h_xj[i].x, h_xj[i].y, h_xj[i].z, h_xj[i].w );
  }
  if(nj < nj1){
    for(int i = nj; i < nj1; i++){
      h_xj[i] = make_float4(0.0, 0.0, 0.0, 0.0);
      //      printf("h_xj:%d\t%e\t%e\t%e\t%e\n",i, h_xj[i].x, h_xj[i].y, h_xj[i].z, h_xj[i].w );
    }
  }
  CUDA_SAFE_CALL( cudaMemcpy( d_xj, h_xj, jp_size, cudaMemcpyHostToDevice) );
  for(int i = 0; i < ni; i++){
    h_xi[i     ] = (float) xi[i][0];
    h_xi[i+ni  ] = (float) xi[i][1];
    h_xi[i+ni*2] = (float) xi[i][2];
    h_xi[i+ni*3] = (float) eps2;
    //    printf("h_xi:%d\t%e\t%e\t%e\t%e\n",i, h_xi[i], h_xi[i+ni], h_xi[i+ni*2], h_xi[i+ni*3] );
  }
  CUDA_SAFE_CALL( cudaMemcpy( d_xi, h_xi, ip_size, cudaMemcpyHostToDevice) );

  //  printf("Nj=%d\n",nj1);
  /*
  dim3 grid((ni+NTHRE)/NTHRE);
  dim3 threads  (NTHRE);
  */
  //  printf("------------------------ %d, %d\n",ni, nj1);
  dim3 grid(ni/128);
  dim3 threads  (128);
  kernel<<< grid, threads >>>(d_xj, d_xi, d_fo, ni, nj1);
  CUT_CHECK_ERROR("Kernel execution failed");

  CUDA_SAFE_CALL( cudaMemcpy( h_fo, d_fo, fo_size, cudaMemcpyDeviceToHost) );

  for(int i=0;i<ni; i++){
    a[i][0] = (double)h_fo[i];
    a[i][1] = (double)h_fo[i+ni];
    a[i][2] = (double)h_fo[i+ni*2];
    //    printf("a:%d\t%e\t%e\t%e\n",i, h_fo[i], h_fo[i+ni], h_fo[i+ni*2] );
  }
}
