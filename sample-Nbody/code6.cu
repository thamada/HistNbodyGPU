//Time-stamp: <2010-06-18 11:31:00 hamada>
// Copyright(C) 2008 by 
// Tsuyoshi Hamada <hamada@progrape.jp>

#include <iostream>
#include "cuda.h"
#include "cutil.h"

namespace libcunbody{

  using namespace std;

  // I want to move these member functions into class cunbody.
  __device__ float4 dev_inter(float4 xi, float4 xj);
  __device__ float4 dev_inter_001(float4 xi, float4 xj);
  __device__ float4 dev_inter_011(float4 xi, float4 xj, float* pot);
  __device__ float4 dev_apot(float4 xi, float4 xj, float4 apot);
  __global__ void cunbody_kernel_tree_001(float4 *xilist, float4 *xjlist, float4 *apotlist, unsigned int *ioffset, unsigned int *joffset);
  __global__ void cunbody_kernel_tree_002(float4 *xilist, float4 *xjlist, float4 *apotlist, unsigned int *ioffset, unsigned int *joffset);
  __global__ void cunbody_kernel_tree_003(float4 *xilist, float4 *xjlist, float4 *apotlist, unsigned int *ioffset, unsigned int *joffset);
  __global__ void cunbody_kernel_tree_011(float4 *xilist, float4 *xjlist, float4 *apotlist, unsigned int *ioffset, unsigned int *joffset);
  __global__ void cunbody_kernel_tree_012(float4 *xilist, float4 *xjlist, float4 *apotlist, unsigned int *ioffset, unsigned int *joffset);
  __global__ void cunbody_kernel_tree_013(float4 *xilist, float4 *xjlist, float4 *apotlist, unsigned int *ioffset, unsigned int *joffset);
  __global__ void cunbody_kernel_tree_999(float4 *xilist, float4 *xjlist, float4 *apotlist, unsigned int *ioffset, unsigned int *joffset);

  // -------------------------------------- TUNING PROPERTY
#define KERNEL_TYPE (0x002)                 // (XXX) = (Jark tyep, Pot type, Acc type), 0: not implement 
  //#define KERNEL_TYPE (0x013)
#define ACC_TYPE (1)
#define MAX_WALK (256)
#define CUDA_MALLOC_TYPE (0)
#define CUDA_MALLOC_HOST_TYPE (0)
#define DEV_OPEN_STRATEGY (1)
#define NTHRE (128)
  //----------------------------------------.

  class cunbody
  {
  private:
    bool is_open;
    int host_pid;      // host process id
    int host_tid;      // host thread Id
    float4 *dev_xi;
    float4 *dev_xj;
    float4 *dev_apot;
    unsigned int *dev_ioff;
    unsigned int *dev_joff;
    unsigned int isize;
    unsigned int jsize;
    unsigned int max_isize;
    unsigned int max_jsize;

//#define ENABLE_DITAILED_OPENMSG

    void dev_check(){
      int ndev;
      CUDA_SAFE_CALL(cudaGetDeviceCount(&ndev));
      if(ndev == 0){
	fprintf(stdout, "ndev = %d @ %s|%d\n", ndev, __FILE__, __LINE__);
	fprintf(stdout, "There is no GPUs.\n");
	exit(-1);
      }else{
	int dev = host_pid % ndev;
	CUDA_SAFE_CALL(cudaSetDevice(dev));
	printf("[%d]  : cudaSetDevice to %d-th GPU\n",        host_pid, dev+1);
	cudaDeviceProp deviceProp;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev));
	if (deviceProp.major == 9999 && deviceProp.minor == 9999){
	  printf("There is no device supporting CUDA.\n");
	}
#if defined(ENABLE_DITAILED_OPENMSG)
	printf("[%d]  : Major revision number:  %d\n",        host_pid, deviceProp.major);
	printf("[%d]  : Minor revision number:  %d\n",        host_pid, deviceProp.minor);
        printf("[%d]  : core clock rate:  %.2f GHz\n",        host_pid, deviceProp.clockRate * 1e-6f);
#  if  (CUDART_VERSION >= 2000)
	printf("[%d]  : Number of cores:  %d\n",              host_pid, 8 * deviceProp.multiProcessorCount);
        printf("[%d]  : Concurrent copy and execution: %s\n", host_pid, deviceProp.deviceOverlap ? "Yes" : "No");
#  endif
#endif

      }
    }

    void dev_open(int hpid, int htid) {
      //      CUT_DEVICE_INIT(, ); // thread safe ?
      host_pid = hpid;
      host_tid = htid;
      fprintf(stdout, "[%d] CUNBODY-1 library: rev.hamada20080920 (^-^)v\n", host_pid);
#if defined(ENABLE_DITAILED_OPENMSG)
      fprintf(stdout, "[%d]  open GPU by host thread %d\n", host_pid, host_tid);
      fprintf(stdout, "[%d]  : KERNEL_TYPE %03x\n",     host_pid, KERNEL_TYPE);
      fprintf(stdout, "[%d]  : MAX_WALK %d\n",          host_pid, MAX_WALK);
      fprintf(stdout, "[%d]  : NTHRE    %d\n",          host_pid, NTHRE);
      fprintf(stdout, "[%d]  : ACC_TYPE %d\n",          host_pid, ACC_TYPE);
      fprintf(stdout, "[%d]  : CUDA_MALLOC_TYPE %d\n",  host_pid, CUDA_MALLOC_TYPE);
      fprintf(stdout, "[%d]  : CUDA_MALLOC_HOST_TYPE %d\n", host_pid, CUDA_MALLOC_HOST_TYPE);
      fprintf(stdout, "[%d]  : DEV_OPEN_STRATEGY %d\n", host_pid, DEV_OPEN_STRATEGY);
#endif
      dev_check();
      //      max_isize = 1500000;
      //      max_jsize = (this->jsize)*2;

      //      max_isize = (this->isize)*3;
      //      max_jsize = (this->jsize) + (this->jsize)>>1;

      //theta=0.5
      //      max_isize = 380000;
      //      max_jsize = 2100000;

      //theta=0.4
#if 0
      max_isize = 380000;
      max_jsize = 4000000;
#else
      max_isize = 520000;
      max_jsize = 4200000;
#endif

      CUDA_SAFE_CALL(cudaMalloc((void **)&dev_xi,   (NTHRE + max_isize) * sizeof(float4)));
      CUDA_SAFE_CALL(cudaMalloc((void **)&dev_apot, max_isize * sizeof(float4)));
      CUDA_SAFE_CALL(cudaMalloc((void **)&dev_xj,   max_jsize * sizeof(float4)));
      CUDA_SAFE_CALL(cudaMalloc((void **)&dev_ioff, (MAX_WALK + 1) * sizeof(unsigned int)));
      CUDA_SAFE_CALL(cudaMalloc((void **)&dev_joff, (MAX_WALK + 1) * sizeof(unsigned int)));
      printf("[%d-proc, %d-thread] ******** cudaMalloc in total at dev_open() : %d MB (j=%d, i=%d)(j=%d MB, i=%d MB)\n", host_pid, host_tid, 
	     (
	      (NTHRE + max_isize) * sizeof(float4) + 
	      max_isize * sizeof(float4) + 
	      max_jsize * sizeof(float4) + 
	      (MAX_WALK + 1) * sizeof(unsigned int) + 
	      (MAX_WALK + 1) * sizeof(unsigned int)
	      )/(1024*1024), //                           <----------- total(i, j, offset)

	     max_jsize, max_isize,

	     (max_jsize * sizeof(float4))/(1024*1024), // <----------- only j

	     ((NTHRE + max_isize) * sizeof(float4) +
	      max_isize * sizeof(float4))/(1024*1024)  // <----------- only i
	     );

      is_open = true;
    }

    void dev_close(void) {
      CUDA_SAFE_CALL(cudaFree(dev_xi));
      CUDA_SAFE_CALL(cudaFree(dev_xj));
      CUDA_SAFE_CALL(cudaFree(dev_apot));
      CUDA_SAFE_CALL(cudaFree(dev_ioff));
      CUDA_SAFE_CALL(cudaFree(dev_joff));
    }

  public:

    cunbody() {
      is_open = false;
      host_pid = -1;
      host_tid = -1;
      max_isize = 0;
      max_jsize = 0;
      dev_xi = NULL;
      dev_xj = NULL;
      dev_apot = NULL;
      dev_ioff = NULL;
      dev_joff = NULL;
    }

    ~cunbody() {
      this->dev_close();
      dev_xi = NULL;
      dev_xj = NULL;
      dev_apot = NULL;
      dev_ioff = NULL;
      dev_joff = NULL;
      max_isize = 0;
      max_jsize = 0;
      host_tid = -1;
      host_pid = -1;
      is_open = false;
    }

    void vforce_open(int host_pid, int host_tid)
    {
      if(is_open == false) this->dev_open(host_pid, host_tid);
    }

    void vforce_mp(int host_pid,
		   int host_tid, 
		float4 xilist[], 
		float4 xjlist[], 
		float4 apotlist[], 
		unsigned int ioff[], 
		unsigned int joff[], 
		unsigned int nwalk) 
    {
      isize = ioff[nwalk];
      jsize = joff[nwalk];

      if(is_open == false) this->dev_open(host_pid, host_tid);

#define WARNING_CUDA_MALLOC

      if(isize > max_isize){
	int isize_bak = isize;
	isize = (int)(isize*1.1);
	max_isize = isize_bak;
#if defined(WARNING_CUDA_MALLOC)
	int megabyte = ((NTHRE + isize) * sizeof(float4)+isize * sizeof(float4))>>20;
	printf("[%d @%d]================== cudaMalloc for i : %d MB\n", host_pid, host_tid, megabyte);
#endif
	CUDA_SAFE_CALL(cudaFree(dev_xi));
	CUDA_SAFE_CALL(cudaFree(dev_apot));
	CUDA_SAFE_CALL(cudaMalloc((void **)&dev_xi,   (NTHRE + isize) * sizeof(float4)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&dev_apot, isize * sizeof(float4)));
	isize = isize_bak;
      }

      if(jsize > max_jsize){
	max_jsize = jsize;
#if defined(WARNING_CUDA_MALLOC)
	int megabyte = (jsize * sizeof(float4))>>20;
	printf("[%d @%d]------------------ cudaMalloc for j : %d MB\n",  host_pid, host_tid, megabyte);
#endif
	CUDA_SAFE_CALL(cudaFree(dev_xj));
	CUDA_SAFE_CALL(cudaMalloc((void **)&dev_xj,   jsize * sizeof(float4)));
      }

      CUDA_SAFE_CALL(cudaMemcpy(dev_xi, xilist, isize * sizeof(float4), cudaMemcpyHostToDevice));
      CUDA_SAFE_CALL(cudaMemcpy(dev_xj, xjlist, jsize * sizeof(float4), cudaMemcpyHostToDevice));
      CUDA_SAFE_CALL(cudaMemcpy(dev_ioff, ioff, (nwalk + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice));
      CUDA_SAFE_CALL(cudaMemcpy(dev_joff, joff, (nwalk + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice));

      dim3 grid(nwalk);
      dim3 threads(NTHRE);

#if   (KERNEL_TYPE == 0x001)
      cunbody_kernel_tree_001 <<< grid, threads >>> (dev_xi, dev_xj, dev_apot, dev_ioff, dev_joff);
#elif (KERNEL_TYPE == 0x002)
      cunbody_kernel_tree_002 <<< grid, threads >>> (dev_xi, dev_xj, dev_apot, dev_ioff, dev_joff);
#elif (KERNEL_TYPE == 0x011)
      cunbody_kernel_tree_011 <<< grid, threads >>> (dev_xi, dev_xj, dev_apot, dev_ioff, dev_joff);
#elif (KERNEL_TYPE == 0x012)
      cunbody_kernel_tree_012 <<< grid, threads >>> (dev_xi, dev_xj, dev_apot, dev_ioff, dev_joff);
#elif (KERNEL_TYPE == 0x013)
      cunbody_kernel_tree_013 <<< grid, threads >>> (dev_xi, dev_xj, dev_apot, dev_ioff, dev_joff);
#elif (KERNEL_TYPE == 0x003)
      cunbody_kernel_tree_003 <<< grid, threads >>> (dev_xi, dev_xj, dev_apot, dev_ioff, dev_joff); // has some bug (# of registers overflow ?)
#elif (KERNEL_TYPE == 0x999)
      cunbody_kernel_tree_999 <<< grid, threads >>> (dev_xi, dev_xj, dev_apot, dev_ioff, dev_joff); // has some bug
#else
      cunbody_kernel_tree_000 <<< grid, threads >>> (dev_xi, dev_xj, dev_apot, dev_ioff, dev_joff);
#endif

      CUT_CHECK_ERROR("KERNEL EXECUTION FAILED");
      CUDA_SAFE_CALL(cudaMemcpy(apotlist, dev_apot, isize * sizeof(float4), cudaMemcpyDeviceToHost));
    }

  }; // class cunbody __END__

}; // namespace libcunbody __END__  ----------------------------------------------------------------------------------



#define MAX_OMP_THRE (4)
static libcunbody::cunbody cunObj[MAX_OMP_THRE];

extern "C"
void vforce(int host_tid, float4 xilist[], float4 xjlist[], float4 apotlist[], unsigned int ioff[], unsigned int joff[], unsigned int nwalk)
{
  using namespace std;
  using namespace libcunbody;
  cunObj[host_tid].vforce_mp(0, host_tid, xilist, xjlist, apotlist, ioff, joff, nwalk);
}

extern "C"
void vforce_mp(int proc_id, int host_tid, float4 xilist[], float4 xjlist[], float4 apotlist[], unsigned int ioff[], unsigned int joff[], unsigned int nwalk)
{
  using namespace std;
  using namespace libcunbody;
  cunObj[host_tid].vforce_mp(proc_id, host_tid, xilist, xjlist, apotlist, ioff, joff, nwalk);
}

extern "C"
void vforce_open(int proc_id, int host_tid)
{
  using namespace std;
  using namespace libcunbody;
  cunObj[host_tid].vforce_open(proc_id, host_tid);
}

#if (NTHRE != 128)
compile-touccha-dame
#endif

namespace libcunbody{

  using namespace std;

  __device__ float4 dev_inter(float4 xi, float4 xj)
  {
    float dx = xj.x - xi.x;
    float dy = xj.y - xi.y;
    float dz = xj.z - xi.z;
    float eps2 = xi.w;
    float mj   = xj.w;

#if (0)
    float r2   = (dz*dz +(dy*dy +(dx*dx+eps2)));
#else
    float r2   = (dx*dx + eps2) + dy*dy + dz*dz;
#endif

    float r1i  = 1/sqrt(r2);
    float r2i  = r1i*r1i;
    float mr3i = mj * r1i * r2i;

#if (0)  
    float4 retval = make_float4(dx, dy, dz, mr3i);
#else
    float4 retval;
    retval.x = dx;
    retval.y = dy;
    retval.z = dz;
    retval.w = mr3i;
#endif

    return retval;
  }

  __global__ 
  void cunbody_kernel_tree_999(float4 *xilist,
				float4 *xjlist,
				float4 *apotlist,
				unsigned int *ioffset,
				unsigned int *joffset)
  {
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int ibegin = ioffset[bid];
    unsigned int iend   = ioffset[bid+1];
    unsigned int jbegin = joffset[bid];
    unsigned int jend   = joffset[bid+1];
    for(unsigned int ibase = ibegin; ibase < iend; ibase += NTHRE){
      float4 xi = xilist[ibase+tid];
      float4 apot = make_float4(0,0,0,0);
      for(unsigned int jbase = jbegin; jbase < jend; jbase += NTHRE){
	__shared__ float4 sj[NTHRE];
	sj[tid] = xjlist[jbase + tid];
	__syncthreads();
	{
	  float4 dm0 = dev_inter(xi, sj[0]);
	  float4 dm1 = dev_inter(xi, sj[1]);
	  float4 dm2 = dev_inter(xi, sj[2]);
	  float4 dm3 = dev_inter(xi, sj[3]);
	  float4 dm4 = dev_inter(xi, sj[4]);
	  float4 dm5 = dev_inter(xi, sj[5]);
	  float4 dm6 = dev_inter(xi, sj[6]);
	  float4 dm7 = dev_inter(xi, sj[7]);
	  float4 dm8 = dev_inter(xi, sj[8]);
	  float4 dm9 = dev_inter(xi, sj[9]);

	  float4 dm10 = dev_inter(xi, sj[10]);
	  float4 dm11 = dev_inter(xi, sj[11]);
	  float4 dm12 = dev_inter(xi, sj[12]);
	  float4 dm13 = dev_inter(xi, sj[13]);
	  float4 dm14 = dev_inter(xi, sj[14]);
	  float4 dm15 = dev_inter(xi, sj[15]);
	  float4 dm16 = dev_inter(xi, sj[16]);
	  float4 dm17 = dev_inter(xi, sj[17]);
	  float4 dm18 = dev_inter(xi, sj[18]);
	  float4 dm19 = dev_inter(xi, sj[19]);

	  float4 dm20 = dev_inter(xi, sj[20]);
	  float4 dm21 = dev_inter(xi, sj[21]);
	  float4 dm22 = dev_inter(xi, sj[22]);
	  float4 dm23 = dev_inter(xi, sj[23]);
	  float4 dm24 = dev_inter(xi, sj[24]);
	  float4 dm25 = dev_inter(xi, sj[25]);
	  float4 dm26 = dev_inter(xi, sj[26]);
	  float4 dm27 = dev_inter(xi, sj[27]);
	  float4 dm28 = dev_inter(xi, sj[28]);
	  float4 dm29 = dev_inter(xi, sj[29]);

	  float4 dm30 = dev_inter(xi, sj[30]);
	  float4 dm31 = dev_inter(xi, sj[31]);
	  float4 dm32 = dev_inter(xi, sj[32]);
	  float4 dm33 = dev_inter(xi, sj[33]);
	  float4 dm34 = dev_inter(xi, sj[34]);
	  float4 dm35 = dev_inter(xi, sj[35]);
	  float4 dm36 = dev_inter(xi, sj[36]);
	  float4 dm37 = dev_inter(xi, sj[37]);
	  float4 dm38 = dev_inter(xi, sj[38]);
	  float4 dm39 = dev_inter(xi, sj[39]);

	  float4 dm40 = dev_inter(xi, sj[40]);
	  float4 dm41 = dev_inter(xi, sj[41]);
	  float4 dm42 = dev_inter(xi, sj[42]);
	  float4 dm43 = dev_inter(xi, sj[43]);
	  float4 dm44 = dev_inter(xi, sj[44]);
	  float4 dm45 = dev_inter(xi, sj[45]);
	  float4 dm46 = dev_inter(xi, sj[46]);
	  float4 dm47 = dev_inter(xi, sj[47]);
	  float4 dm48 = dev_inter(xi, sj[48]);
	  float4 dm49 = dev_inter(xi, sj[49]);

	  float4 dm50 = dev_inter(xi, sj[50]);
	  float4 dm51 = dev_inter(xi, sj[51]);
	  float4 dm52 = dev_inter(xi, sj[52]);
	  float4 dm53 = dev_inter(xi, sj[53]);
	  float4 dm54 = dev_inter(xi, sj[54]);
	  float4 dm55 = dev_inter(xi, sj[55]);
	  float4 dm56 = dev_inter(xi, sj[56]);
	  float4 dm57 = dev_inter(xi, sj[57]);
	  float4 dm58 = dev_inter(xi, sj[58]);
	  float4 dm59 = dev_inter(xi, sj[59]);

	  float4 dm60 = dev_inter(xi, sj[60]);
	  float4 dm61 = dev_inter(xi, sj[61]);
	  float4 dm62 = dev_inter(xi, sj[62]);
	  float4 dm63 = dev_inter(xi, sj[63]);
	  float4 dm64 = dev_inter(xi, sj[64]);
	  float4 dm65 = dev_inter(xi, sj[65]);
	  float4 dm66 = dev_inter(xi, sj[66]);
	  float4 dm67 = dev_inter(xi, sj[67]);
	  float4 dm68 = dev_inter(xi, sj[68]);
	  float4 dm69 = dev_inter(xi, sj[69]);

	  float4 dm70 = dev_inter(xi, sj[70]);
	  float4 dm71 = dev_inter(xi, sj[71]);
	  float4 dm72 = dev_inter(xi, sj[72]);
	  float4 dm73 = dev_inter(xi, sj[73]);
	  float4 dm74 = dev_inter(xi, sj[74]);
	  float4 dm75 = dev_inter(xi, sj[75]);
	  float4 dm76 = dev_inter(xi, sj[76]);
	  float4 dm77 = dev_inter(xi, sj[77]);
	  float4 dm78 = dev_inter(xi, sj[78]);
	  float4 dm79 = dev_inter(xi, sj[79]);

	  float4 dm80 = dev_inter(xi, sj[80]);
	  float4 dm81 = dev_inter(xi, sj[81]);
	  float4 dm82 = dev_inter(xi, sj[82]);
	  float4 dm83 = dev_inter(xi, sj[83]);
	  float4 dm84 = dev_inter(xi, sj[84]);
	  float4 dm85 = dev_inter(xi, sj[85]);
	  float4 dm86 = dev_inter(xi, sj[86]);
	  float4 dm87 = dev_inter(xi, sj[87]);
	  float4 dm88 = dev_inter(xi, sj[88]);
	  float4 dm89 = dev_inter(xi, sj[89]);

	  float4 dm90 = dev_inter(xi, sj[90]);
	  float4 dm91 = dev_inter(xi, sj[91]);
	  float4 dm92 = dev_inter(xi, sj[92]);
	  float4 dm93 = dev_inter(xi, sj[93]);
	  float4 dm94 = dev_inter(xi, sj[94]);
	  float4 dm95 = dev_inter(xi, sj[95]);
	  float4 dm96 = dev_inter(xi, sj[96]);
	  float4 dm97 = dev_inter(xi, sj[97]);
	  float4 dm98 = dev_inter(xi, sj[98]);
	  float4 dm99 = dev_inter(xi, sj[99]);

	  float4 dm100 = dev_inter(xi, sj[100]);
	  float4 dm101 = dev_inter(xi, sj[101]);
	  float4 dm102 = dev_inter(xi, sj[102]);
	  float4 dm103 = dev_inter(xi, sj[103]);
	  float4 dm104 = dev_inter(xi, sj[104]);
	  float4 dm105 = dev_inter(xi, sj[105]);
	  float4 dm106 = dev_inter(xi, sj[106]);
	  float4 dm107 = dev_inter(xi, sj[107]);
	  float4 dm108 = dev_inter(xi, sj[108]);
	  float4 dm109 = dev_inter(xi, sj[109]);

	  float4 dm110 = dev_inter(xi, sj[110]);
	  float4 dm111 = dev_inter(xi, sj[111]);
	  float4 dm112 = dev_inter(xi, sj[112]);
	  float4 dm113 = dev_inter(xi, sj[113]);
	  float4 dm114 = dev_inter(xi, sj[114]);
	  float4 dm115 = dev_inter(xi, sj[115]);
	  float4 dm116 = dev_inter(xi, sj[116]);
	  float4 dm117 = dev_inter(xi, sj[117]);
	  float4 dm118 = dev_inter(xi, sj[118]);
	  float4 dm119 = dev_inter(xi, sj[119]);

	  float4 dm120 = dev_inter(xi, sj[120]);
	  float4 dm121 = dev_inter(xi, sj[121]);
	  float4 dm122 = dev_inter(xi, sj[122]);
	  float4 dm123 = dev_inter(xi, sj[123]);
	  float4 dm124 = dev_inter(xi, sj[124]);
	  float4 dm125 = dev_inter(xi, sj[125]);
	  float4 dm126 = dev_inter(xi, sj[126]);
	  float4 dm127 = dev_inter(xi, sj[127]);

	  apot.x = dm0.x * dm0.w + dm1.x * dm1.w + dm2.x * dm2.w + dm3.x * dm3.w + dm4.x * dm4.w + dm5.x * dm5.w + dm6.x * dm6.w + dm7.x * dm7.w + dm8.x * dm8.w + dm9.x * dm9.w + 
	    dm10.x * dm10.w + dm11.x * dm11.w + dm12.x * dm12.w + dm13.x * dm13.w + dm14.x * dm14.w + dm15.x * dm15.w + dm16.x * dm16.w + dm17.x * dm17.w + dm18.x * dm18.w + dm19.x * dm19.w + 
	    dm20.x * dm20.w + dm21.x * dm21.w + dm22.x * dm22.w + dm23.x * dm23.w + dm24.x * dm24.w + dm25.x * dm25.w + dm26.x * dm26.w + dm27.x * dm27.w + dm28.x * dm28.w + dm29.x * dm29.w + 
	    dm30.x * dm30.w + dm31.x * dm31.w + dm32.x * dm32.w + dm33.x * dm33.w + dm34.x * dm34.w + dm35.x * dm35.w + dm36.x * dm36.w + dm37.x * dm37.w + dm38.x * dm38.w + dm39.x * dm39.w + 
	    dm40.x * dm40.w + dm41.x * dm41.w + dm42.x * dm42.w + dm43.x * dm43.w + dm44.x * dm44.w + dm45.x * dm45.w + dm46.x * dm46.w + dm47.x * dm47.w + dm48.x * dm48.w + dm49.x * dm49.w + 
	    dm50.x * dm50.w + dm51.x * dm51.w + dm52.x * dm52.w + dm53.x * dm53.w + dm54.x * dm54.w + dm55.x * dm55.w + dm56.x * dm56.w + dm57.x * dm57.w + dm58.x * dm58.w + dm59.x * dm59.w + 
	    dm60.x * dm60.w + dm61.x * dm61.w + dm62.x * dm62.w + dm63.x * dm63.w + dm64.x * dm64.w + dm65.x * dm65.w + dm66.x * dm66.w + dm67.x * dm67.w + dm68.x * dm68.w + dm69.x * dm69.w + 
	    dm70.x * dm70.w + dm71.x * dm71.w + dm72.x * dm72.w + dm73.x * dm73.w + dm74.x * dm74.w + dm75.x * dm75.w + dm76.x * dm76.w + dm77.x * dm77.w + dm78.x * dm78.w + dm79.x * dm79.w + 
	    dm80.x * dm80.w + dm81.x * dm81.w + dm82.x * dm82.w + dm83.x * dm83.w + dm84.x * dm84.w + dm85.x * dm85.w + dm86.x * dm86.w + dm87.x * dm87.w + dm88.x * dm88.w + dm89.x * dm89.w + 
	    dm90.x * dm90.w + dm91.x * dm91.w + dm92.x * dm92.w + dm93.x * dm93.w + dm94.x * dm94.w + dm95.x * dm95.w + dm96.x * dm96.w + dm97.x * dm97.w + dm98.x * dm98.w + dm99.x * dm99.w + 
	    dm100.x * dm100.w + dm101.x * dm101.w + dm102.x * dm102.w + dm103.x * dm103.w + dm104.x * dm104.w + dm105.x * dm105.w + dm106.x * dm106.w + dm107.x * dm107.w + dm108.x * dm108.w + dm109.x * dm109.w + 
	    dm110.x * dm110.w + dm111.x * dm111.w + dm112.x * dm112.w + dm113.x * dm113.w + dm114.x * dm114.w + dm115.x * dm115.w + dm116.x * dm116.w + dm117.x * dm117.w + dm118.x * dm118.w + dm119.x * dm119.w + 
	    dm120.x * dm120.w + dm121.x * dm121.w + dm122.x * dm122.w + dm123.x * dm123.w + dm124.x * dm124.w + dm125.x * dm125.w + dm126.x * dm126.w + dm127.x * dm127.w;

	  apot.y = dm0.y * dm0.w + dm1.y * dm1.w + dm2.y * dm2.w + dm3.y * dm3.w + dm4.y * dm4.w + dm5.y * dm5.w + dm6.y * dm6.w + dm7.y * dm7.w + dm8.y * dm8.w + dm9.y * dm9.w + 
	    dm10.y * dm10.w + dm11.y * dm11.w + dm12.y * dm12.w + dm13.y * dm13.w + dm14.y * dm14.w + dm15.y * dm15.w + dm16.y * dm16.w + dm17.y * dm17.w + dm18.y * dm18.w + dm19.y * dm19.w + 
	    dm20.y * dm20.w + dm21.y * dm21.w + dm22.y * dm22.w + dm23.y * dm23.w + dm24.y * dm24.w + dm25.y * dm25.w + dm26.y * dm26.w + dm27.y * dm27.w + dm28.y * dm28.w + dm29.y * dm29.w + 
	    dm30.y * dm30.w + dm31.y * dm31.w + dm32.y * dm32.w + dm33.y * dm33.w + dm34.y * dm34.w + dm35.y * dm35.w + dm36.y * dm36.w + dm37.y * dm37.w + dm38.y * dm38.w + dm39.y * dm39.w + 
	    dm40.y * dm40.w + dm41.y * dm41.w + dm42.y * dm42.w + dm43.y * dm43.w + dm44.y * dm44.w + dm45.y * dm45.w + dm46.y * dm46.w + dm47.y * dm47.w + dm48.y * dm48.w + dm49.y * dm49.w + 
	    dm50.y * dm50.w + dm51.y * dm51.w + dm52.y * dm52.w + dm53.y * dm53.w + dm54.y * dm54.w + dm55.y * dm55.w + dm56.y * dm56.w + dm57.y * dm57.w + dm58.y * dm58.w + dm59.y * dm59.w + 
	    dm60.y * dm60.w + dm61.y * dm61.w + dm62.y * dm62.w + dm63.y * dm63.w + dm64.y * dm64.w + dm65.y * dm65.w + dm66.y * dm66.w + dm67.y * dm67.w + dm68.y * dm68.w + dm69.y * dm69.w + 
	    dm70.y * dm70.w + dm71.y * dm71.w + dm72.y * dm72.w + dm73.y * dm73.w + dm74.y * dm74.w + dm75.y * dm75.w + dm76.y * dm76.w + dm77.y * dm77.w + dm78.y * dm78.w + dm79.y * dm79.w + 
	    dm80.y * dm80.w + dm81.y * dm81.w + dm82.y * dm82.w + dm83.y * dm83.w + dm84.y * dm84.w + dm85.y * dm85.w + dm86.y * dm86.w + dm87.y * dm87.w + dm88.y * dm88.w + dm89.y * dm89.w + 
	    dm90.y * dm90.w + dm91.y * dm91.w + dm92.y * dm92.w + dm93.y * dm93.w + dm94.y * dm94.w + dm95.y * dm95.w + dm96.y * dm96.w + dm97.y * dm97.w + dm98.y * dm98.w + dm99.y * dm99.w + 
	    dm100.y * dm100.w + dm101.y * dm101.w + dm102.y * dm102.w + dm103.y * dm103.w + dm104.y * dm104.w + dm105.y * dm105.w + dm106.y * dm106.w + dm107.y * dm107.w + dm108.y * dm108.w + dm109.y * dm109.w + 
	    dm110.y * dm110.w + dm111.y * dm111.w + dm112.y * dm112.w + dm113.y * dm113.w + dm114.y * dm114.w + dm115.y * dm115.w + dm116.y * dm116.w + dm117.y * dm117.w + dm118.y * dm118.w + dm119.y * dm119.w + 
	    dm120.y * dm120.w + dm121.y * dm121.w + dm122.y * dm122.w + dm123.y * dm123.w + dm124.y * dm124.w + dm125.y * dm125.w + dm126.y * dm126.w + dm127.y * dm127.w;

	  apot.z = dm0.z * dm0.w + dm1.z * dm1.w + dm2.z * dm2.w + dm3.z * dm3.w + dm4.z * dm4.w + dm5.z * dm5.w + dm6.z * dm6.w + dm7.z * dm7.w + dm8.z * dm8.w + dm9.z * dm9.w + 
	    dm10.z * dm10.w + dm11.z * dm11.w + dm12.z * dm12.w + dm13.z * dm13.w + dm14.z * dm14.w + dm15.z * dm15.w + dm16.z * dm16.w + dm17.z * dm17.w + dm18.z * dm18.w + dm19.z * dm19.w + 
	    dm20.z * dm20.w + dm21.z * dm21.w + dm22.z * dm22.w + dm23.z * dm23.w + dm24.z * dm24.w + dm25.z * dm25.w + dm26.z * dm26.w + dm27.z * dm27.w + dm28.z * dm28.w + dm29.z * dm29.w + 
	    dm30.z * dm30.w + dm31.z * dm31.w + dm32.z * dm32.w + dm33.z * dm33.w + dm34.z * dm34.w + dm35.z * dm35.w + dm36.z * dm36.w + dm37.z * dm37.w + dm38.z * dm38.w + dm39.z * dm39.w + 
	    dm40.z * dm40.w + dm41.z * dm41.w + dm42.z * dm42.w + dm43.z * dm43.w + dm44.z * dm44.w + dm45.z * dm45.w + dm46.z * dm46.w + dm47.z * dm47.w + dm48.z * dm48.w + dm49.z * dm49.w + 
	    dm50.z * dm50.w + dm51.z * dm51.w + dm52.z * dm52.w + dm53.z * dm53.w + dm54.z * dm54.w + dm55.z * dm55.w + dm56.z * dm56.w + dm57.z * dm57.w + dm58.z * dm58.w + dm59.z * dm59.w + 
	    dm60.z * dm60.w + dm61.z * dm61.w + dm62.z * dm62.w + dm63.z * dm63.w + dm64.z * dm64.w + dm65.z * dm65.w + dm66.z * dm66.w + dm67.z * dm67.w + dm68.z * dm68.w + dm69.z * dm69.w + 
	    dm70.z * dm70.w + dm71.z * dm71.w + dm72.z * dm72.w + dm73.z * dm73.w + dm74.z * dm74.w + dm75.z * dm75.w + dm76.z * dm76.w + dm77.z * dm77.w + dm78.z * dm78.w + dm79.z * dm79.w + 
	    dm80.z * dm80.w + dm81.z * dm81.w + dm82.z * dm82.w + dm83.z * dm83.w + dm84.z * dm84.w + dm85.z * dm85.w + dm86.z * dm86.w + dm87.z * dm87.w + dm88.z * dm88.w + dm89.z * dm89.w + 
	    dm90.z * dm90.w + dm91.z * dm91.w + dm92.z * dm92.w + dm93.z * dm93.w + dm94.z * dm94.w + dm95.z * dm95.w + dm96.z * dm96.w + dm97.z * dm97.w + dm98.z * dm98.w + dm99.z * dm99.w + 
	    dm100.z * dm100.w + dm101.z * dm101.w + dm102.z * dm102.w + dm103.z * dm103.w + dm104.z * dm104.w + dm105.z * dm105.w + dm106.z * dm106.w + dm107.z * dm107.w + dm108.z * dm108.w + dm109.z * dm109.w + 
	    dm110.z * dm110.w + dm111.z * dm111.w + dm112.z * dm112.w + dm113.z * dm113.w + dm114.z * dm114.w + dm115.z * dm115.w + dm116.z * dm116.w + dm117.z * dm117.w + dm118.z * dm118.w + dm119.z * dm119.w + 
	    dm120.z * dm120.w + dm121.z * dm121.w + dm122.z * dm122.w + dm123.z * dm123.w + dm124.z * dm124.w + dm125.z * dm125.w + dm126.z * dm126.w + dm127.z * dm127.w;

	}
      }
      if(ibase+tid < iend){
	apotlist[ibase+tid] = apot;
      }
    }
  }



  __device__ 
  float4 dev_inter_001(float4 xi, float4 xj)
  {
    float dx = xj.x - xi.x;
    float dy = xj.y - xi.y;
    float dz = xj.z - xi.z;
    float eps2 = xi.w;
    float mj   = xj.w;
#if (0)
    float r2 = dx*dz + (dy*dy + (dx*dx + eps2));
#else
    float r2 = (dx*dx + eps2) + dy*dy + dz*dz;
#endif
    //	if(r2 == eps2) return (a,p)

    float r1i = 1/sqrt(r2);
    //  float r1i = rsqrt(r2);

    float r2i = r1i*r1i;
    float mr3i = mj * r1i * r2i;
    float4 retval;

    retval.x = dx; 
    retval.y = dy; 
    retval.z = dz;
    retval.w = mr3i; 

    return (retval);
  }



  __global__ 
  void cunbody_kernel_tree_001(float4 *xilist,
				float4 *xjlist,
				float4 *apotlist,
				unsigned int *ioffset,
				unsigned int *joffset)
  {
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int ibegin = ioffset[bid];
    unsigned int iend   = ioffset[bid+1];
    unsigned int jbegin = joffset[bid];
    unsigned int jend   = joffset[bid+1];
    for(unsigned int ibase = ibegin; ibase < iend; ibase += NTHRE){
      float4 xi = xilist[ibase + tid];
      float4 apot = make_float4(0,0,0,0);
      for(unsigned int jbase = jbegin; jbase < jend; jbase += NTHRE){
	__shared__ float4 sj[NTHRE];
	sj[tid] = xjlist[jbase + tid];
	__syncthreads();
	{
	  float4 dm0, dm1, dm2, dm3, dm4, dm5, dm6, dm7, dm8, dm9;
	  dm0 = dev_inter_001(xi, sj[0]);
	  dm1 = dev_inter_001(xi, sj[1]);
	  dm2 = dev_inter_001(xi, sj[2]);
	  dm3 = dev_inter_001(xi, sj[3]);
	  dm4 = dev_inter_001(xi, sj[4]);
	  dm5 = dev_inter_001(xi, sj[5]);
	  dm6 = dev_inter_001(xi, sj[6]);
	  dm7 = dev_inter_001(xi, sj[7]);
	  dm8 = dev_inter_001(xi, sj[8]);
	  dm9 = dev_inter_001(xi, sj[9]);
	  apot.x += (dm0.x*dm0.w) + (dm1.x*dm1.w) + (dm2.x*dm2.w) + (dm3.x*dm3.w) + (dm4.x*dm4.w) + (dm5.x*dm5.w) + (dm6.x*dm6.w) + (dm7.x*dm7.w) + (dm8.x*dm8.w) + (dm9.x*dm9.w);
	  apot.y += (dm0.y*dm0.w) + (dm1.y*dm1.w) + (dm2.y*dm2.w) + (dm3.y*dm3.w) + (dm4.y*dm4.w) + (dm5.y*dm5.w) + (dm6.y*dm6.w) + (dm7.y*dm7.w) + (dm8.y*dm8.w) + (dm9.y*dm9.w);
	  apot.z += (dm0.z*dm0.w) + (dm1.z*dm1.w) + (dm2.z*dm2.w) + (dm3.z*dm3.w) + (dm4.z*dm4.w) + (dm5.z*dm5.w) + (dm6.z*dm6.w) + (dm7.z*dm7.w) + (dm8.z*dm8.w) + (dm9.z*dm9.w);
	}
	{
	  float4 dm0, dm1, dm2, dm3, dm4, dm5, dm6, dm7, dm8, dm9;
	  dm0 = dev_inter_001(xi, sj[10]);
	  dm1 = dev_inter_001(xi, sj[11]);
	  dm2 = dev_inter_001(xi, sj[12]);
	  dm3 = dev_inter_001(xi, sj[13]);
	  dm4 = dev_inter_001(xi, sj[14]);
	  dm5 = dev_inter_001(xi, sj[15]);
	  dm6 = dev_inter_001(xi, sj[16]);
	  dm7 = dev_inter_001(xi, sj[17]);
	  dm8 = dev_inter_001(xi, sj[18]);
	  dm9 = dev_inter_001(xi, sj[19]);
	  apot.x += (dm0.x*dm0.w) + (dm1.x*dm1.w) + (dm2.x*dm2.w) + (dm3.x*dm3.w) + (dm4.x*dm4.w) + (dm5.x*dm5.w) + (dm6.x*dm6.w) + (dm7.x*dm7.w) + (dm8.x*dm8.w) + (dm9.x*dm9.w);
	  apot.y += (dm0.y*dm0.w) + (dm1.y*dm1.w) + (dm2.y*dm2.w) + (dm3.y*dm3.w) + (dm4.y*dm4.w) + (dm5.y*dm5.w) + (dm6.y*dm6.w) + (dm7.y*dm7.w) + (dm8.y*dm8.w) + (dm9.y*dm9.w);
	  apot.z += (dm0.z*dm0.w) + (dm1.z*dm1.w) + (dm2.z*dm2.w) + (dm3.z*dm3.w) + (dm4.z*dm4.w) + (dm5.z*dm5.w) + (dm6.z*dm6.w) + (dm7.z*dm7.w) + (dm8.z*dm8.w) + (dm9.z*dm9.w);
	}
	{
	  float4 dm0, dm1, dm2, dm3, dm4, dm5, dm6, dm7, dm8, dm9;
	  dm0 = dev_inter_001(xi, sj[20]);
	  dm1 = dev_inter_001(xi, sj[21]);
	  dm2 = dev_inter_001(xi, sj[22]);
	  dm3 = dev_inter_001(xi, sj[23]);
	  dm4 = dev_inter_001(xi, sj[24]);
	  dm5 = dev_inter_001(xi, sj[25]);
	  dm6 = dev_inter_001(xi, sj[26]);
	  dm7 = dev_inter_001(xi, sj[27]);
	  dm8 = dev_inter_001(xi, sj[28]);
	  dm9 = dev_inter_001(xi, sj[29]);
	  apot.x += (dm0.x*dm0.w) + (dm1.x*dm1.w) + (dm2.x*dm2.w) + (dm3.x*dm3.w) + (dm4.x*dm4.w) + (dm5.x*dm5.w) + (dm6.x*dm6.w) + (dm7.x*dm7.w) + (dm8.x*dm8.w) + (dm9.x*dm9.w);
	  apot.y += (dm0.y*dm0.w) + (dm1.y*dm1.w) + (dm2.y*dm2.w) + (dm3.y*dm3.w) + (dm4.y*dm4.w) + (dm5.y*dm5.w) + (dm6.y*dm6.w) + (dm7.y*dm7.w) + (dm8.y*dm8.w) + (dm9.y*dm9.w);
	  apot.z += (dm0.z*dm0.w) + (dm1.z*dm1.w) + (dm2.z*dm2.w) + (dm3.z*dm3.w) + (dm4.z*dm4.w) + (dm5.z*dm5.w) + (dm6.z*dm6.w) + (dm7.z*dm7.w) + (dm8.z*dm8.w) + (dm9.z*dm9.w);
	}
	{
	  float4 dm0, dm1, dm2, dm3, dm4, dm5, dm6, dm7, dm8, dm9;
	  dm0 = dev_inter_001(xi, sj[30]);
	  dm1 = dev_inter_001(xi, sj[31]);
	  dm2 = dev_inter_001(xi, sj[32]);
	  dm3 = dev_inter_001(xi, sj[33]);
	  dm4 = dev_inter_001(xi, sj[34]);
	  dm5 = dev_inter_001(xi, sj[35]);
	  dm6 = dev_inter_001(xi, sj[36]);
	  dm7 = dev_inter_001(xi, sj[37]);
	  dm8 = dev_inter_001(xi, sj[38]);
	  dm9 = dev_inter_001(xi, sj[39]);
	  apot.x += (dm0.x*dm0.w) + (dm1.x*dm1.w) + (dm2.x*dm2.w) + (dm3.x*dm3.w) + (dm4.x*dm4.w) + (dm5.x*dm5.w) + (dm6.x*dm6.w) + (dm7.x*dm7.w) + (dm8.x*dm8.w) + (dm9.x*dm9.w);
	  apot.y += (dm0.y*dm0.w) + (dm1.y*dm1.w) + (dm2.y*dm2.w) + (dm3.y*dm3.w) + (dm4.y*dm4.w) + (dm5.y*dm5.w) + (dm6.y*dm6.w) + (dm7.y*dm7.w) + (dm8.y*dm8.w) + (dm9.y*dm9.w);
	  apot.z += (dm0.z*dm0.w) + (dm1.z*dm1.w) + (dm2.z*dm2.w) + (dm3.z*dm3.w) + (dm4.z*dm4.w) + (dm5.z*dm5.w) + (dm6.z*dm6.w) + (dm7.z*dm7.w) + (dm8.z*dm8.w) + (dm9.z*dm9.w);
	}
	{
	  float4 dm0, dm1, dm2, dm3, dm4, dm5, dm6, dm7, dm8, dm9;
	  dm0 = dev_inter_001(xi, sj[40]);
	  dm1 = dev_inter_001(xi, sj[41]);
	  dm2 = dev_inter_001(xi, sj[42]);
	  dm3 = dev_inter_001(xi, sj[43]);
	  dm4 = dev_inter_001(xi, sj[44]);
	  dm5 = dev_inter_001(xi, sj[45]);
	  dm6 = dev_inter_001(xi, sj[46]);
	  dm7 = dev_inter_001(xi, sj[47]);
	  dm8 = dev_inter_001(xi, sj[48]);
	  dm9 = dev_inter_001(xi, sj[49]);
	  apot.x += (dm0.x*dm0.w) + (dm1.x*dm1.w) + (dm2.x*dm2.w) + (dm3.x*dm3.w) + (dm4.x*dm4.w) + (dm5.x*dm5.w) + (dm6.x*dm6.w) + (dm7.x*dm7.w) + (dm8.x*dm8.w) + (dm9.x*dm9.w);
	  apot.y += (dm0.y*dm0.w) + (dm1.y*dm1.w) + (dm2.y*dm2.w) + (dm3.y*dm3.w) + (dm4.y*dm4.w) + (dm5.y*dm5.w) + (dm6.y*dm6.w) + (dm7.y*dm7.w) + (dm8.y*dm8.w) + (dm9.y*dm9.w);
	  apot.z += (dm0.z*dm0.w) + (dm1.z*dm1.w) + (dm2.z*dm2.w) + (dm3.z*dm3.w) + (dm4.z*dm4.w) + (dm5.z*dm5.w) + (dm6.z*dm6.w) + (dm7.z*dm7.w) + (dm8.z*dm8.w) + (dm9.z*dm9.w);
	}
	{
	  float4 dm0, dm1, dm2, dm3, dm4, dm5, dm6, dm7, dm8, dm9;
	  dm0 = dev_inter_001(xi, sj[50]);
	  dm1 = dev_inter_001(xi, sj[51]);
	  dm2 = dev_inter_001(xi, sj[52]);
	  dm3 = dev_inter_001(xi, sj[53]);
	  dm4 = dev_inter_001(xi, sj[54]);
	  dm5 = dev_inter_001(xi, sj[55]);
	  dm6 = dev_inter_001(xi, sj[56]);
	  dm7 = dev_inter_001(xi, sj[57]);
	  dm8 = dev_inter_001(xi, sj[58]);
	  dm9 = dev_inter_001(xi, sj[59]);
	  apot.x += (dm0.x*dm0.w) + (dm1.x*dm1.w) + (dm2.x*dm2.w) + (dm3.x*dm3.w) + (dm4.x*dm4.w) + (dm5.x*dm5.w) + (dm6.x*dm6.w) + (dm7.x*dm7.w) + (dm8.x*dm8.w) + (dm9.x*dm9.w);
	  apot.y += (dm0.y*dm0.w) + (dm1.y*dm1.w) + (dm2.y*dm2.w) + (dm3.y*dm3.w) + (dm4.y*dm4.w) + (dm5.y*dm5.w) + (dm6.y*dm6.w) + (dm7.y*dm7.w) + (dm8.y*dm8.w) + (dm9.y*dm9.w);
	  apot.z += (dm0.z*dm0.w) + (dm1.z*dm1.w) + (dm2.z*dm2.w) + (dm3.z*dm3.w) + (dm4.z*dm4.w) + (dm5.z*dm5.w) + (dm6.z*dm6.w) + (dm7.z*dm7.w) + (dm8.z*dm8.w) + (dm9.z*dm9.w);
	}
	{
	  float4 dm0, dm1, dm2, dm3, dm4, dm5, dm6, dm7, dm8, dm9;
	  dm0 = dev_inter_001(xi, sj[60]);
	  dm1 = dev_inter_001(xi, sj[61]);
	  dm2 = dev_inter_001(xi, sj[62]);
	  dm3 = dev_inter_001(xi, sj[63]);
	  dm4 = dev_inter_001(xi, sj[64]);
	  dm5 = dev_inter_001(xi, sj[65]);
	  dm6 = dev_inter_001(xi, sj[66]);
	  dm7 = dev_inter_001(xi, sj[67]);
	  dm8 = dev_inter_001(xi, sj[68]);
	  dm9 = dev_inter_001(xi, sj[69]);
	  apot.x += (dm0.x*dm0.w) + (dm1.x*dm1.w) + (dm2.x*dm2.w) + (dm3.x*dm3.w) + (dm4.x*dm4.w) + (dm5.x*dm5.w) + (dm6.x*dm6.w) + (dm7.x*dm7.w) + (dm8.x*dm8.w) + (dm9.x*dm9.w);
	  apot.y += (dm0.y*dm0.w) + (dm1.y*dm1.w) + (dm2.y*dm2.w) + (dm3.y*dm3.w) + (dm4.y*dm4.w) + (dm5.y*dm5.w) + (dm6.y*dm6.w) + (dm7.y*dm7.w) + (dm8.y*dm8.w) + (dm9.y*dm9.w);
	  apot.z += (dm0.z*dm0.w) + (dm1.z*dm1.w) + (dm2.z*dm2.w) + (dm3.z*dm3.w) + (dm4.z*dm4.w) + (dm5.z*dm5.w) + (dm6.z*dm6.w) + (dm7.z*dm7.w) + (dm8.z*dm8.w) + (dm9.z*dm9.w);
	}
	{
	  float4 dm0, dm1, dm2, dm3, dm4, dm5, dm6, dm7, dm8, dm9;
	  dm0 = dev_inter_001(xi, sj[70]);
	  dm1 = dev_inter_001(xi, sj[71]);
	  dm2 = dev_inter_001(xi, sj[72]);
	  dm3 = dev_inter_001(xi, sj[73]);
	  dm4 = dev_inter_001(xi, sj[74]);
	  dm5 = dev_inter_001(xi, sj[75]);
	  dm6 = dev_inter_001(xi, sj[76]);
	  dm7 = dev_inter_001(xi, sj[77]);
	  dm8 = dev_inter_001(xi, sj[78]);
	  dm9 = dev_inter_001(xi, sj[79]);
	  apot.x += (dm0.x*dm0.w) + (dm1.x*dm1.w) + (dm2.x*dm2.w) + (dm3.x*dm3.w) + (dm4.x*dm4.w) + (dm5.x*dm5.w) + (dm6.x*dm6.w) + (dm7.x*dm7.w) + (dm8.x*dm8.w) + (dm9.x*dm9.w);
	  apot.y += (dm0.y*dm0.w) + (dm1.y*dm1.w) + (dm2.y*dm2.w) + (dm3.y*dm3.w) + (dm4.y*dm4.w) + (dm5.y*dm5.w) + (dm6.y*dm6.w) + (dm7.y*dm7.w) + (dm8.y*dm8.w) + (dm9.y*dm9.w);
	  apot.z += (dm0.z*dm0.w) + (dm1.z*dm1.w) + (dm2.z*dm2.w) + (dm3.z*dm3.w) + (dm4.z*dm4.w) + (dm5.z*dm5.w) + (dm6.z*dm6.w) + (dm7.z*dm7.w) + (dm8.z*dm8.w) + (dm9.z*dm9.w);
	}
	{
	  float4 dm0, dm1, dm2, dm3, dm4, dm5, dm6, dm7, dm8, dm9;
	  dm0 = dev_inter_001(xi, sj[80]);
	  dm1 = dev_inter_001(xi, sj[81]);
	  dm2 = dev_inter_001(xi, sj[82]);
	  dm3 = dev_inter_001(xi, sj[83]);
	  dm4 = dev_inter_001(xi, sj[84]);
	  dm5 = dev_inter_001(xi, sj[85]);
	  dm6 = dev_inter_001(xi, sj[86]);
	  dm7 = dev_inter_001(xi, sj[87]);
	  dm8 = dev_inter_001(xi, sj[88]);
	  dm9 = dev_inter_001(xi, sj[89]);
	  apot.x += (dm0.x*dm0.w) + (dm1.x*dm1.w) + (dm2.x*dm2.w) + (dm3.x*dm3.w) + (dm4.x*dm4.w) + (dm5.x*dm5.w) + (dm6.x*dm6.w) + (dm7.x*dm7.w) + (dm8.x*dm8.w) + (dm9.x*dm9.w);
	  apot.y += (dm0.y*dm0.w) + (dm1.y*dm1.w) + (dm2.y*dm2.w) + (dm3.y*dm3.w) + (dm4.y*dm4.w) + (dm5.y*dm5.w) + (dm6.y*dm6.w) + (dm7.y*dm7.w) + (dm8.y*dm8.w) + (dm9.y*dm9.w);
	  apot.z += (dm0.z*dm0.w) + (dm1.z*dm1.w) + (dm2.z*dm2.w) + (dm3.z*dm3.w) + (dm4.z*dm4.w) + (dm5.z*dm5.w) + (dm6.z*dm6.w) + (dm7.z*dm7.w) + (dm8.z*dm8.w) + (dm9.z*dm9.w);
	}
	{
	  float4 dm0, dm1, dm2, dm3, dm4, dm5, dm6, dm7, dm8, dm9;
	  dm0 = dev_inter_001(xi, sj[90]);
	  dm1 = dev_inter_001(xi, sj[91]);
	  dm2 = dev_inter_001(xi, sj[92]);
	  dm3 = dev_inter_001(xi, sj[93]);
	  dm4 = dev_inter_001(xi, sj[94]);
	  dm5 = dev_inter_001(xi, sj[95]);
	  dm6 = dev_inter_001(xi, sj[96]);
	  dm7 = dev_inter_001(xi, sj[97]);
	  dm8 = dev_inter_001(xi, sj[98]);
	  dm9 = dev_inter_001(xi, sj[99]);
	  apot.x += (dm0.x*dm0.w) + (dm1.x*dm1.w) + (dm2.x*dm2.w) + (dm3.x*dm3.w) + (dm4.x*dm4.w) + (dm5.x*dm5.w) + (dm6.x*dm6.w) + (dm7.x*dm7.w) + (dm8.x*dm8.w) + (dm9.x*dm9.w);
	  apot.y += (dm0.y*dm0.w) + (dm1.y*dm1.w) + (dm2.y*dm2.w) + (dm3.y*dm3.w) + (dm4.y*dm4.w) + (dm5.y*dm5.w) + (dm6.y*dm6.w) + (dm7.y*dm7.w) + (dm8.y*dm8.w) + (dm9.y*dm9.w);
	  apot.z += (dm0.z*dm0.w) + (dm1.z*dm1.w) + (dm2.z*dm2.w) + (dm3.z*dm3.w) + (dm4.z*dm4.w) + (dm5.z*dm5.w) + (dm6.z*dm6.w) + (dm7.z*dm7.w) + (dm8.z*dm8.w) + (dm9.z*dm9.w);
	}
	{
	  float4 dm0, dm1, dm2, dm3, dm4, dm5, dm6, dm7, dm8, dm9;
	  dm0 = dev_inter_001(xi, sj[100]);
	  dm1 = dev_inter_001(xi, sj[101]);
	  dm2 = dev_inter_001(xi, sj[102]);
	  dm3 = dev_inter_001(xi, sj[103]);
	  dm4 = dev_inter_001(xi, sj[104]);
	  dm5 = dev_inter_001(xi, sj[105]);
	  dm6 = dev_inter_001(xi, sj[106]);
	  dm7 = dev_inter_001(xi, sj[107]);
	  dm8 = dev_inter_001(xi, sj[108]);
	  dm9 = dev_inter_001(xi, sj[109]);
	  apot.x += (dm0.x*dm0.w) + (dm1.x*dm1.w) + (dm2.x*dm2.w) + (dm3.x*dm3.w) + (dm4.x*dm4.w) + (dm5.x*dm5.w) + (dm6.x*dm6.w) + (dm7.x*dm7.w) + (dm8.x*dm8.w) + (dm9.x*dm9.w);
	  apot.y += (dm0.y*dm0.w) + (dm1.y*dm1.w) + (dm2.y*dm2.w) + (dm3.y*dm3.w) + (dm4.y*dm4.w) + (dm5.y*dm5.w) + (dm6.y*dm6.w) + (dm7.y*dm7.w) + (dm8.y*dm8.w) + (dm9.y*dm9.w);
	  apot.z += (dm0.z*dm0.w) + (dm1.z*dm1.w) + (dm2.z*dm2.w) + (dm3.z*dm3.w) + (dm4.z*dm4.w) + (dm5.z*dm5.w) + (dm6.z*dm6.w) + (dm7.z*dm7.w) + (dm8.z*dm8.w) + (dm9.z*dm9.w);
	}
	{
	  float4 dm0, dm1, dm2, dm3, dm4, dm5, dm6, dm7, dm8, dm9;
	  dm0 = dev_inter_001(xi, sj[110]);
	  dm1 = dev_inter_001(xi, sj[111]);
	  dm2 = dev_inter_001(xi, sj[112]);
	  dm3 = dev_inter_001(xi, sj[113]);
	  dm4 = dev_inter_001(xi, sj[114]);
	  dm5 = dev_inter_001(xi, sj[115]);
	  dm6 = dev_inter_001(xi, sj[116]);
	  dm7 = dev_inter_001(xi, sj[117]);
	  dm8 = dev_inter_001(xi, sj[118]);
	  dm9 = dev_inter_001(xi, sj[119]);
	  apot.x += (dm0.x*dm0.w) + (dm1.x*dm1.w) + (dm2.x*dm2.w) + (dm3.x*dm3.w) + (dm4.x*dm4.w) + (dm5.x*dm5.w) + (dm6.x*dm6.w) + (dm7.x*dm7.w) + (dm8.x*dm8.w) + (dm9.x*dm9.w);
	  apot.y += (dm0.y*dm0.w) + (dm1.y*dm1.w) + (dm2.y*dm2.w) + (dm3.y*dm3.w) + (dm4.y*dm4.w) + (dm5.y*dm5.w) + (dm6.y*dm6.w) + (dm7.y*dm7.w) + (dm8.y*dm8.w) + (dm9.y*dm9.w);
	  apot.z += (dm0.z*dm0.w) + (dm1.z*dm1.w) + (dm2.z*dm2.w) + (dm3.z*dm3.w) + (dm4.z*dm4.w) + (dm5.z*dm5.w) + (dm6.z*dm6.w) + (dm7.z*dm7.w) + (dm8.z*dm8.w) + (dm9.z*dm9.w);
	}
	{
	  float4 dm0, dm1, dm2, dm3, dm4, dm5, dm6, dm7;//, dm8, dm9;
	  dm0 = dev_inter_001(xi, sj[120]);
	  dm1 = dev_inter_001(xi, sj[121]);
	  dm2 = dev_inter_001(xi, sj[122]);
	  dm3 = dev_inter_001(xi, sj[123]);
	  dm4 = dev_inter_001(xi, sj[124]);
	  dm5 = dev_inter_001(xi, sj[125]);
	  dm6 = dev_inter_001(xi, sj[126]);
	  dm7 = dev_inter_001(xi, sj[127]);
	  apot.x += (dm0.x*dm0.w) + (dm1.x*dm1.w) + (dm2.x*dm2.w) + (dm3.x*dm3.w) + (dm4.x*dm4.w) + (dm5.x*dm5.w) + (dm6.x*dm6.w) + (dm7.x*dm7.w);// + (dm8.x*dm8.w) + (dm9.x*dm9.w);
	  apot.y += (dm0.y*dm0.w) + (dm1.y*dm1.w) + (dm2.y*dm2.w) + (dm3.y*dm3.w) + (dm4.y*dm4.w) + (dm5.y*dm5.w) + (dm6.y*dm6.w) + (dm7.y*dm7.w);// + (dm8.y*dm8.w) + (dm9.y*dm9.w);
	  apot.z += (dm0.z*dm0.w) + (dm1.z*dm1.w) + (dm2.z*dm2.w) + (dm3.z*dm3.w) + (dm4.z*dm4.w) + (dm5.z*dm5.w) + (dm6.z*dm6.w) + (dm7.z*dm7.w);// + (dm8.z*dm8.w) + (dm9.z*dm9.w);
	}
      }
      if(ibase + tid < iend){
	apotlist[ibase + tid] = apot;
      }
    }
  }


  __global__ 
  void cunbody_kernel_tree_002(float4 *xilist,
				float4 *xjlist,
				float4 *apotlist,
				unsigned int *ioffset,
				unsigned int *joffset)
  {
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int ibegin = ioffset[bid];
    unsigned int iend   = ioffset[bid+1];
    unsigned int jbegin = joffset[bid];
    unsigned int jend   = joffset[bid+1];
    for(unsigned int ibase = ibegin; ibase < iend; ibase += NTHRE){
      float4 xi = xilist[ibase + tid];
      float4 apot = make_float4(0,0,0,0);
      for(unsigned int jbase = jbegin; jbase < jend; jbase += NTHRE){
	__shared__ float4 sj[NTHRE];
	__syncthreads(); // bugfixed 2008/09/19
	sj[tid] = xjlist[jbase + tid];
	__syncthreads();
	{
	  float4 dm0, dm1, dm2, dm3, dm4, dm5, dm6, dm7, dm8, dm9, dm10, dm11, dm12, dm13, dm14, dm15;
	  dm0 = dev_inter_001(xi, sj[0]);
	  dm1 = dev_inter_001(xi, sj[1]);
	  dm2 = dev_inter_001(xi, sj[2]);
	  dm3 = dev_inter_001(xi, sj[3]);
	  dm4 = dev_inter_001(xi, sj[4]);
	  dm5 = dev_inter_001(xi, sj[5]);
	  dm6 = dev_inter_001(xi, sj[6]);
	  dm7 = dev_inter_001(xi, sj[7]);
	  dm8 = dev_inter_001(xi, sj[8]);
	  dm9 = dev_inter_001(xi, sj[9]);
	  dm10 = dev_inter_001(xi, sj[10]);
	  dm11 = dev_inter_001(xi, sj[11]);
	  dm12 = dev_inter_001(xi, sj[12]);
	  dm13 = dev_inter_001(xi, sj[13]);
	  dm14 = dev_inter_001(xi, sj[14]);
	  dm15 = dev_inter_001(xi, sj[15]);
	  apot.x += (dm0.x*dm0.w) + (dm1.x*dm1.w) + (dm2.x*dm2.w) + (dm3.x*dm3.w) + (dm4.x*dm4.w) + (dm5.x*dm5.w) + (dm6.x*dm6.w) + (dm7.x*dm7.w) \
	    + (dm8.x*dm8.w) + (dm9.x*dm9.w) + (dm10.x*dm10.w) + (dm11.x*dm11.w) + (dm12.x*dm12.w) + (dm13.x*dm13.w) + (dm14.x*dm14.w) + (dm15.x*dm15.w);
	  apot.y += (dm0.y*dm0.w) + (dm1.y*dm1.w) + (dm2.y*dm2.w) + (dm3.y*dm3.w) + (dm4.y*dm4.w) + (dm5.y*dm5.w) + (dm6.y*dm6.w) + (dm7.y*dm7.w) \
	    + (dm8.y*dm8.w) + (dm9.y*dm9.w) + (dm10.y*dm10.w) + (dm11.y*dm11.w) + (dm12.y*dm12.w) + (dm13.y*dm13.w) + (dm14.y*dm14.w) + (dm15.y*dm15.w);
	  apot.z += (dm0.z*dm0.w) + (dm1.z*dm1.w) + (dm2.z*dm2.w) + (dm3.z*dm3.w) + (dm4.z*dm4.w) + (dm5.z*dm5.w) + (dm6.z*dm6.w) + (dm7.z*dm7.w) \
	    + (dm8.z*dm8.w) + (dm9.z*dm9.w) + (dm10.z*dm10.w) + (dm11.z*dm11.w) + (dm12.z*dm12.w) + (dm13.z*dm13.w) + (dm14.z*dm14.w) + (dm15.z*dm15.w);
	}

	{
	  float4 dm0, dm1, dm2, dm3, dm4, dm5, dm6, dm7, dm8, dm9, dm10, dm11, dm12, dm13, dm14, dm15;
	  dm0 = dev_inter_001(xi, sj[16]);
	  dm1 = dev_inter_001(xi, sj[17]);
	  dm2 = dev_inter_001(xi, sj[18]);
	  dm3 = dev_inter_001(xi, sj[19]);
	  dm4 = dev_inter_001(xi, sj[20]);
	  dm5 = dev_inter_001(xi, sj[21]);
	  dm6 = dev_inter_001(xi, sj[22]);
	  dm7 = dev_inter_001(xi, sj[23]);
	  dm8 = dev_inter_001(xi, sj[24]);
	  dm9 = dev_inter_001(xi, sj[25]);
	  dm10 = dev_inter_001(xi, sj[26]);
	  dm11 = dev_inter_001(xi, sj[27]);
	  dm12 = dev_inter_001(xi, sj[28]);
	  dm13 = dev_inter_001(xi, sj[29]);
	  dm14 = dev_inter_001(xi, sj[30]);
	  dm15 = dev_inter_001(xi, sj[31]);
	  apot.x += (dm0.x*dm0.w) + (dm1.x*dm1.w) + (dm2.x*dm2.w) + (dm3.x*dm3.w) + (dm4.x*dm4.w) + (dm5.x*dm5.w) + (dm6.x*dm6.w) + (dm7.x*dm7.w) \
	    + (dm8.x*dm8.w) + (dm9.x*dm9.w) + (dm10.x*dm10.w) + (dm11.x*dm11.w) + (dm12.x*dm12.w) + (dm13.x*dm13.w) + (dm14.x*dm14.w) + (dm15.x*dm15.w);
	  apot.y += (dm0.y*dm0.w) + (dm1.y*dm1.w) + (dm2.y*dm2.w) + (dm3.y*dm3.w) + (dm4.y*dm4.w) + (dm5.y*dm5.w) + (dm6.y*dm6.w) + (dm7.y*dm7.w) \
	    + (dm8.y*dm8.w) + (dm9.y*dm9.w) + (dm10.y*dm10.w) + (dm11.y*dm11.w) + (dm12.y*dm12.w) + (dm13.y*dm13.w) + (dm14.y*dm14.w) + (dm15.y*dm15.w);
	  apot.z += (dm0.z*dm0.w) + (dm1.z*dm1.w) + (dm2.z*dm2.w) + (dm3.z*dm3.w) + (dm4.z*dm4.w) + (dm5.z*dm5.w) + (dm6.z*dm6.w) + (dm7.z*dm7.w) \
	    + (dm8.z*dm8.w) + (dm9.z*dm9.w) + (dm10.z*dm10.w) + (dm11.z*dm11.w) + (dm12.z*dm12.w) + (dm13.z*dm13.w) + (dm14.z*dm14.w) + (dm15.z*dm15.w);
	}

	{
	  float4 dm0, dm1, dm2, dm3, dm4, dm5, dm6, dm7, dm8, dm9, dm10, dm11, dm12, dm13, dm14, dm15;
	  dm0 = dev_inter_001(xi, sj[32]);
	  dm1 = dev_inter_001(xi, sj[33]);
	  dm2 = dev_inter_001(xi, sj[34]);
	  dm3 = dev_inter_001(xi, sj[35]);
	  dm4 = dev_inter_001(xi, sj[36]);
	  dm5 = dev_inter_001(xi, sj[37]);
	  dm6 = dev_inter_001(xi, sj[38]);
	  dm7 = dev_inter_001(xi, sj[39]);
	  dm8 = dev_inter_001(xi, sj[40]);
	  dm9 = dev_inter_001(xi, sj[41]);
	  dm10 = dev_inter_001(xi, sj[42]);
	  dm11 = dev_inter_001(xi, sj[43]);
	  dm12 = dev_inter_001(xi, sj[44]);
	  dm13 = dev_inter_001(xi, sj[45]);
	  dm14 = dev_inter_001(xi, sj[46]);
	  dm15 = dev_inter_001(xi, sj[47]);
	  apot.x += (dm0.x*dm0.w) + (dm1.x*dm1.w) + (dm2.x*dm2.w) + (dm3.x*dm3.w) + (dm4.x*dm4.w) + (dm5.x*dm5.w) + (dm6.x*dm6.w) + (dm7.x*dm7.w) \
	    + (dm8.x*dm8.w) + (dm9.x*dm9.w) + (dm10.x*dm10.w) + (dm11.x*dm11.w) + (dm12.x*dm12.w) + (dm13.x*dm13.w) + (dm14.x*dm14.w) + (dm15.x*dm15.w);
	  apot.y += (dm0.y*dm0.w) + (dm1.y*dm1.w) + (dm2.y*dm2.w) + (dm3.y*dm3.w) + (dm4.y*dm4.w) + (dm5.y*dm5.w) + (dm6.y*dm6.w) + (dm7.y*dm7.w) \
	    + (dm8.y*dm8.w) + (dm9.y*dm9.w) + (dm10.y*dm10.w) + (dm11.y*dm11.w) + (dm12.y*dm12.w) + (dm13.y*dm13.w) + (dm14.y*dm14.w) + (dm15.y*dm15.w);
	  apot.z += (dm0.z*dm0.w) + (dm1.z*dm1.w) + (dm2.z*dm2.w) + (dm3.z*dm3.w) + (dm4.z*dm4.w) + (dm5.z*dm5.w) + (dm6.z*dm6.w) + (dm7.z*dm7.w) \
	    + (dm8.z*dm8.w) + (dm9.z*dm9.w) + (dm10.z*dm10.w) + (dm11.z*dm11.w) + (dm12.z*dm12.w) + (dm13.z*dm13.w) + (dm14.z*dm14.w) + (dm15.z*dm15.w);
	}

	{
	  float4 dm0, dm1, dm2, dm3, dm4, dm5, dm6, dm7, dm8, dm9, dm10, dm11, dm12, dm13, dm14, dm15;
	  dm0 = dev_inter_001(xi, sj[48]);
	  dm1 = dev_inter_001(xi, sj[49]);
	  dm2 = dev_inter_001(xi, sj[50]);
	  dm3 = dev_inter_001(xi, sj[51]);
	  dm4 = dev_inter_001(xi, sj[52]);
	  dm5 = dev_inter_001(xi, sj[53]);
	  dm6 = dev_inter_001(xi, sj[54]);
	  dm7 = dev_inter_001(xi, sj[55]);
	  dm8 = dev_inter_001(xi, sj[56]);
	  dm9 = dev_inter_001(xi, sj[57]);
	  dm10 = dev_inter_001(xi, sj[58]);
	  dm11 = dev_inter_001(xi, sj[59]);
	  dm12 = dev_inter_001(xi, sj[60]);
	  dm13 = dev_inter_001(xi, sj[61]);
	  dm14 = dev_inter_001(xi, sj[62]);
	  dm15 = dev_inter_001(xi, sj[63]);
	  apot.x += (dm0.x*dm0.w) + (dm1.x*dm1.w) + (dm2.x*dm2.w) + (dm3.x*dm3.w) + (dm4.x*dm4.w) + (dm5.x*dm5.w) + (dm6.x*dm6.w) + (dm7.x*dm7.w) \
	    + (dm8.x*dm8.w) + (dm9.x*dm9.w) + (dm10.x*dm10.w) + (dm11.x*dm11.w) + (dm12.x*dm12.w) + (dm13.x*dm13.w) + (dm14.x*dm14.w) + (dm15.x*dm15.w);
	  apot.y += (dm0.y*dm0.w) + (dm1.y*dm1.w) + (dm2.y*dm2.w) + (dm3.y*dm3.w) + (dm4.y*dm4.w) + (dm5.y*dm5.w) + (dm6.y*dm6.w) + (dm7.y*dm7.w) \
	    + (dm8.y*dm8.w) + (dm9.y*dm9.w) + (dm10.y*dm10.w) + (dm11.y*dm11.w) + (dm12.y*dm12.w) + (dm13.y*dm13.w) + (dm14.y*dm14.w) + (dm15.y*dm15.w);
	  apot.z += (dm0.z*dm0.w) + (dm1.z*dm1.w) + (dm2.z*dm2.w) + (dm3.z*dm3.w) + (dm4.z*dm4.w) + (dm5.z*dm5.w) + (dm6.z*dm6.w) + (dm7.z*dm7.w) \
	    + (dm8.z*dm8.w) + (dm9.z*dm9.w) + (dm10.z*dm10.w) + (dm11.z*dm11.w) + (dm12.z*dm12.w) + (dm13.z*dm13.w) + (dm14.z*dm14.w) + (dm15.z*dm15.w);
	}

	{
	  float4 dm0, dm1, dm2, dm3, dm4, dm5, dm6, dm7, dm8, dm9, dm10, dm11, dm12, dm13, dm14, dm15;
	  dm0 = dev_inter_001(xi, sj[64]);
	  dm1 = dev_inter_001(xi, sj[65]);
	  dm2 = dev_inter_001(xi, sj[66]);
	  dm3 = dev_inter_001(xi, sj[67]);
	  dm4 = dev_inter_001(xi, sj[68]);
	  dm5 = dev_inter_001(xi, sj[69]);
	  dm6 = dev_inter_001(xi, sj[70]);
	  dm7 = dev_inter_001(xi, sj[71]);
	  dm8 = dev_inter_001(xi, sj[72]);
	  dm9 = dev_inter_001(xi, sj[73]);
	  dm10 = dev_inter_001(xi, sj[74]);
	  dm11 = dev_inter_001(xi, sj[75]);
	  dm12 = dev_inter_001(xi, sj[76]);
	  dm13 = dev_inter_001(xi, sj[77]);
	  dm14 = dev_inter_001(xi, sj[78]);
	  dm15 = dev_inter_001(xi, sj[79]);
	  apot.x += (dm0.x*dm0.w) + (dm1.x*dm1.w) + (dm2.x*dm2.w) + (dm3.x*dm3.w) + (dm4.x*dm4.w) + (dm5.x*dm5.w) + (dm6.x*dm6.w) + (dm7.x*dm7.w) \
	    + (dm8.x*dm8.w) + (dm9.x*dm9.w) + (dm10.x*dm10.w) + (dm11.x*dm11.w) + (dm12.x*dm12.w) + (dm13.x*dm13.w) + (dm14.x*dm14.w) + (dm15.x*dm15.w);
	  apot.y += (dm0.y*dm0.w) + (dm1.y*dm1.w) + (dm2.y*dm2.w) + (dm3.y*dm3.w) + (dm4.y*dm4.w) + (dm5.y*dm5.w) + (dm6.y*dm6.w) + (dm7.y*dm7.w) \
	    + (dm8.y*dm8.w) + (dm9.y*dm9.w) + (dm10.y*dm10.w) + (dm11.y*dm11.w) + (dm12.y*dm12.w) + (dm13.y*dm13.w) + (dm14.y*dm14.w) + (dm15.y*dm15.w);
	  apot.z += (dm0.z*dm0.w) + (dm1.z*dm1.w) + (dm2.z*dm2.w) + (dm3.z*dm3.w) + (dm4.z*dm4.w) + (dm5.z*dm5.w) + (dm6.z*dm6.w) + (dm7.z*dm7.w) \
	    + (dm8.z*dm8.w) + (dm9.z*dm9.w) + (dm10.z*dm10.w) + (dm11.z*dm11.w) + (dm12.z*dm12.w) + (dm13.z*dm13.w) + (dm14.z*dm14.w) + (dm15.z*dm15.w);
	}

	{
	  float4 dm0, dm1, dm2, dm3, dm4, dm5, dm6, dm7, dm8, dm9, dm10, dm11, dm12, dm13, dm14, dm15;
	  dm0 = dev_inter_001(xi, sj[80]);
	  dm1 = dev_inter_001(xi, sj[81]);
	  dm2 = dev_inter_001(xi, sj[82]);
	  dm3 = dev_inter_001(xi, sj[83]);
	  dm4 = dev_inter_001(xi, sj[84]);
	  dm5 = dev_inter_001(xi, sj[85]);
	  dm6 = dev_inter_001(xi, sj[86]);
	  dm7 = dev_inter_001(xi, sj[87]);
	  dm8 = dev_inter_001(xi, sj[88]);
	  dm9 = dev_inter_001(xi, sj[89]);
	  dm10 = dev_inter_001(xi, sj[90]);
	  dm11 = dev_inter_001(xi, sj[91]);
	  dm12 = dev_inter_001(xi, sj[92]);
	  dm13 = dev_inter_001(xi, sj[93]);
	  dm14 = dev_inter_001(xi, sj[94]);
	  dm15 = dev_inter_001(xi, sj[95]);
	  apot.x += (dm0.x*dm0.w) + (dm1.x*dm1.w) + (dm2.x*dm2.w) + (dm3.x*dm3.w) + (dm4.x*dm4.w) + (dm5.x*dm5.w) + (dm6.x*dm6.w) + (dm7.x*dm7.w) \
	    + (dm8.x*dm8.w) + (dm9.x*dm9.w) + (dm10.x*dm10.w) + (dm11.x*dm11.w) + (dm12.x*dm12.w) + (dm13.x*dm13.w) + (dm14.x*dm14.w) + (dm15.x*dm15.w);
	  apot.y += (dm0.y*dm0.w) + (dm1.y*dm1.w) + (dm2.y*dm2.w) + (dm3.y*dm3.w) + (dm4.y*dm4.w) + (dm5.y*dm5.w) + (dm6.y*dm6.w) + (dm7.y*dm7.w) \
	    + (dm8.y*dm8.w) + (dm9.y*dm9.w) + (dm10.y*dm10.w) + (dm11.y*dm11.w) + (dm12.y*dm12.w) + (dm13.y*dm13.w) + (dm14.y*dm14.w) + (dm15.y*dm15.w);
	  apot.z += (dm0.z*dm0.w) + (dm1.z*dm1.w) + (dm2.z*dm2.w) + (dm3.z*dm3.w) + (dm4.z*dm4.w) + (dm5.z*dm5.w) + (dm6.z*dm6.w) + (dm7.z*dm7.w) \
	    + (dm8.z*dm8.w) + (dm9.z*dm9.w) + (dm10.z*dm10.w) + (dm11.z*dm11.w) + (dm12.z*dm12.w) + (dm13.z*dm13.w) + (dm14.z*dm14.w) + (dm15.z*dm15.w);
	}

	{
	  float4 dm0, dm1, dm2, dm3, dm4, dm5, dm6, dm7, dm8, dm9, dm10, dm11, dm12, dm13, dm14, dm15;
	  dm0 = dev_inter_001(xi, sj[96]);
	  dm1 = dev_inter_001(xi, sj[97]);
	  dm2 = dev_inter_001(xi, sj[98]);
	  dm3 = dev_inter_001(xi, sj[99]);
	  dm4 = dev_inter_001(xi, sj[100]);
	  dm5 = dev_inter_001(xi, sj[101]);
	  dm6 = dev_inter_001(xi, sj[102]);
	  dm7 = dev_inter_001(xi, sj[103]);
	  dm8 = dev_inter_001(xi, sj[104]);
	  dm9 = dev_inter_001(xi, sj[105]);
	  dm10 = dev_inter_001(xi, sj[106]);
	  dm11 = dev_inter_001(xi, sj[107]);
	  dm12 = dev_inter_001(xi, sj[108]);
	  dm13 = dev_inter_001(xi, sj[109]);
	  dm14 = dev_inter_001(xi, sj[110]);
	  dm15 = dev_inter_001(xi, sj[111]);
	  apot.x += (dm0.x*dm0.w) + (dm1.x*dm1.w) + (dm2.x*dm2.w) + (dm3.x*dm3.w) + (dm4.x*dm4.w) + (dm5.x*dm5.w) + (dm6.x*dm6.w) + (dm7.x*dm7.w) \
	    + (dm8.x*dm8.w) + (dm9.x*dm9.w) + (dm10.x*dm10.w) + (dm11.x*dm11.w) + (dm12.x*dm12.w) + (dm13.x*dm13.w) + (dm14.x*dm14.w) + (dm15.x*dm15.w);
	  apot.y += (dm0.y*dm0.w) + (dm1.y*dm1.w) + (dm2.y*dm2.w) + (dm3.y*dm3.w) + (dm4.y*dm4.w) + (dm5.y*dm5.w) + (dm6.y*dm6.w) + (dm7.y*dm7.w) \
	    + (dm8.y*dm8.w) + (dm9.y*dm9.w) + (dm10.y*dm10.w) + (dm11.y*dm11.w) + (dm12.y*dm12.w) + (dm13.y*dm13.w) + (dm14.y*dm14.w) + (dm15.y*dm15.w);
	  apot.z += (dm0.z*dm0.w) + (dm1.z*dm1.w) + (dm2.z*dm2.w) + (dm3.z*dm3.w) + (dm4.z*dm4.w) + (dm5.z*dm5.w) + (dm6.z*dm6.w) + (dm7.z*dm7.w) \
	    + (dm8.z*dm8.w) + (dm9.z*dm9.w) + (dm10.z*dm10.w) + (dm11.z*dm11.w) + (dm12.z*dm12.w) + (dm13.z*dm13.w) + (dm14.z*dm14.w) + (dm15.z*dm15.w);
	}

	{
	  float4 dm0, dm1, dm2, dm3, dm4, dm5, dm6, dm7, dm8, dm9, dm10, dm11, dm12, dm13, dm14, dm15;
	  dm0 = dev_inter_001(xi, sj[112]);
	  dm1 = dev_inter_001(xi, sj[113]);
	  dm2 = dev_inter_001(xi, sj[114]);
	  dm3 = dev_inter_001(xi, sj[115]);
	  dm4 = dev_inter_001(xi, sj[116]);
	  dm5 = dev_inter_001(xi, sj[117]);
	  dm6 = dev_inter_001(xi, sj[118]);
	  dm7 = dev_inter_001(xi, sj[119]);
	  dm8 = dev_inter_001(xi, sj[120]);
	  dm9 = dev_inter_001(xi, sj[121]);
	  dm10 = dev_inter_001(xi, sj[122]);
	  dm11 = dev_inter_001(xi, sj[123]);
	  dm12 = dev_inter_001(xi, sj[124]);
	  dm13 = dev_inter_001(xi, sj[125]);
	  dm14 = dev_inter_001(xi, sj[126]);
	  dm15 = dev_inter_001(xi, sj[127]);
	  apot.x += (dm0.x*dm0.w) + (dm1.x*dm1.w) + (dm2.x*dm2.w) + (dm3.x*dm3.w) + (dm4.x*dm4.w) + (dm5.x*dm5.w) + (dm6.x*dm6.w) + (dm7.x*dm7.w) \
	    + (dm8.x*dm8.w) + (dm9.x*dm9.w) + (dm10.x*dm10.w) + (dm11.x*dm11.w) + (dm12.x*dm12.w) + (dm13.x*dm13.w) + (dm14.x*dm14.w) + (dm15.x*dm15.w);
	  apot.y += (dm0.y*dm0.w) + (dm1.y*dm1.w) + (dm2.y*dm2.w) + (dm3.y*dm3.w) + (dm4.y*dm4.w) + (dm5.y*dm5.w) + (dm6.y*dm6.w) + (dm7.y*dm7.w) \
	    + (dm8.y*dm8.w) + (dm9.y*dm9.w) + (dm10.y*dm10.w) + (dm11.y*dm11.w) + (dm12.y*dm12.w) + (dm13.y*dm13.w) + (dm14.y*dm14.w) + (dm15.y*dm15.w);
	  apot.z += (dm0.z*dm0.w) + (dm1.z*dm1.w) + (dm2.z*dm2.w) + (dm3.z*dm3.w) + (dm4.z*dm4.w) + (dm5.z*dm5.w) + (dm6.z*dm6.w) + (dm7.z*dm7.w) \
	    + (dm8.z*dm8.w) + (dm9.z*dm9.w) + (dm10.z*dm10.w) + (dm11.z*dm11.w) + (dm12.z*dm12.w) + (dm13.z*dm13.w) + (dm14.z*dm14.w) + (dm15.z*dm15.w);
	}

      }
      if(ibase + tid < iend){
	apotlist[ibase + tid] = apot;
      }
    }
  }

  __global__ 
  void cunbody_kernel_tree_003(float4 *xilist,
				float4 *xjlist,
				float4 *apotlist,
				unsigned int *ioffset,
				unsigned int *joffset)
  {
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int ibegin = ioffset[bid];
    unsigned int iend   = ioffset[bid+1];
    unsigned int jbegin = joffset[bid];
    unsigned int jend   = joffset[bid+1];
    for(unsigned int ibase = ibegin; ibase < iend; ibase += NTHRE){
      float4 xi = xilist[ibase + tid];
      float4 apot = make_float4(0,0,0,0);
      for(unsigned int jbase = jbegin; jbase < jend; jbase += NTHRE){
	__shared__ float4 sj[NTHRE];
	sj[tid] = xjlist[jbase + tid];
	__syncthreads();
	{
	  float4 dm0 = dev_inter_001(xi, sj[0]);
	  float4 dm1 = dev_inter_001(xi, sj[1]);
	  float4 dm2 = dev_inter_001(xi, sj[2]);
	  float4 dm3 = dev_inter_001(xi, sj[3]);
	  float4 dm4 = dev_inter_001(xi, sj[4]);
	  float4 dm5 = dev_inter_001(xi, sj[5]);
	  float4 dm6 = dev_inter_001(xi, sj[6]);
	  float4 dm7 = dev_inter_001(xi, sj[7]);
	  float4 dm8 = dev_inter_001(xi, sj[8]);
	  float4 dm9 = dev_inter_001(xi, sj[9]);

	  float4 dm10 = dev_inter_001(xi, sj[0]);
	  float4 dm11 = dev_inter_001(xi, sj[1]);
	  float4 dm12 = dev_inter_001(xi, sj[2]);
	  float4 dm13 = dev_inter_001(xi, sj[3]);
	  float4 dm14 = dev_inter_001(xi, sj[4]);
	  float4 dm15 = dev_inter_001(xi, sj[5]);
	  float4 dm16 = dev_inter_001(xi, sj[6]);
	  float4 dm17 = dev_inter_001(xi, sj[7]);
	  float4 dm18 = dev_inter_001(xi, sj[8]);
	  float4 dm19 = dev_inter_001(xi, sj[9]);

	  float4 dm20 = dev_inter_001(xi, sj[0]);
	  float4 dm21 = dev_inter_001(xi, sj[1]);
	  float4 dm22 = dev_inter_001(xi, sj[2]);
	  float4 dm23 = dev_inter_001(xi, sj[3]);
	  float4 dm24 = dev_inter_001(xi, sj[4]);
	  float4 dm25 = dev_inter_001(xi, sj[5]);
	  float4 dm26 = dev_inter_001(xi, sj[6]);
	  float4 dm27 = dev_inter_001(xi, sj[7]);
	  float4 dm28 = dev_inter_001(xi, sj[8]);
	  float4 dm29 = dev_inter_001(xi, sj[9]);

	  float4 dm30 = dev_inter_001(xi, sj[0]);
	  float4 dm31 = dev_inter_001(xi, sj[1]);
	  float4 dm32 = dev_inter_001(xi, sj[2]);
	  float4 dm33 = dev_inter_001(xi, sj[3]);
	  float4 dm34 = dev_inter_001(xi, sj[4]);
	  float4 dm35 = dev_inter_001(xi, sj[5]);
	  float4 dm36 = dev_inter_001(xi, sj[6]);
	  float4 dm37 = dev_inter_001(xi, sj[7]);
	  float4 dm38 = dev_inter_001(xi, sj[8]);
	  float4 dm39 = dev_inter_001(xi, sj[9]);

	  float4 dm40 = dev_inter_001(xi, sj[0]);
	  float4 dm41 = dev_inter_001(xi, sj[1]);
	  float4 dm42 = dev_inter_001(xi, sj[2]);
	  float4 dm43 = dev_inter_001(xi, sj[3]);
	  float4 dm44 = dev_inter_001(xi, sj[4]);
	  float4 dm45 = dev_inter_001(xi, sj[5]);
	  float4 dm46 = dev_inter_001(xi, sj[6]);
	  float4 dm47 = dev_inter_001(xi, sj[7]);
	  float4 dm48 = dev_inter_001(xi, sj[8]);
	  float4 dm49 = dev_inter_001(xi, sj[9]);

	  float4 dm50 = dev_inter_001(xi, sj[0]);
	  float4 dm51 = dev_inter_001(xi, sj[1]);
	  float4 dm52 = dev_inter_001(xi, sj[2]);
	  float4 dm53 = dev_inter_001(xi, sj[3]);
	  float4 dm54 = dev_inter_001(xi, sj[4]);
	  float4 dm55 = dev_inter_001(xi, sj[5]);
	  float4 dm56 = dev_inter_001(xi, sj[6]);
	  float4 dm57 = dev_inter_001(xi, sj[7]);
	  float4 dm58 = dev_inter_001(xi, sj[8]);
	  float4 dm59 = dev_inter_001(xi, sj[9]);

	  float4 dm60 = dev_inter_001(xi, sj[0]);
	  float4 dm61 = dev_inter_001(xi, sj[1]);
	  float4 dm62 = dev_inter_001(xi, sj[2]);
	  float4 dm63 = dev_inter_001(xi, sj[3]);
	  float4 dm64 = dev_inter_001(xi, sj[4]);
	  float4 dm65 = dev_inter_001(xi, sj[5]);
	  float4 dm66 = dev_inter_001(xi, sj[6]);
	  float4 dm67 = dev_inter_001(xi, sj[7]);
	  float4 dm68 = dev_inter_001(xi, sj[8]);
	  float4 dm69 = dev_inter_001(xi, sj[9]);

	  float4 dm70 = dev_inter_001(xi, sj[0]);
	  float4 dm71 = dev_inter_001(xi, sj[1]);
	  float4 dm72 = dev_inter_001(xi, sj[2]);
	  float4 dm73 = dev_inter_001(xi, sj[3]);
	  float4 dm74 = dev_inter_001(xi, sj[4]);
	  float4 dm75 = dev_inter_001(xi, sj[5]);
	  float4 dm76 = dev_inter_001(xi, sj[6]);
	  float4 dm77 = dev_inter_001(xi, sj[7]);
	  float4 dm78 = dev_inter_001(xi, sj[8]);
	  float4 dm79 = dev_inter_001(xi, sj[9]);

	  float4 dm80 = dev_inter_001(xi, sj[0]);
	  float4 dm81 = dev_inter_001(xi, sj[1]);
	  float4 dm82 = dev_inter_001(xi, sj[2]);
	  float4 dm83 = dev_inter_001(xi, sj[3]);
	  float4 dm84 = dev_inter_001(xi, sj[4]);
	  float4 dm85 = dev_inter_001(xi, sj[5]);
	  float4 dm86 = dev_inter_001(xi, sj[6]);
	  float4 dm87 = dev_inter_001(xi, sj[7]);
	  float4 dm88 = dev_inter_001(xi, sj[8]);
	  float4 dm89 = dev_inter_001(xi, sj[9]);

	  float4 dm90 = dev_inter_001(xi, sj[0]);
	  float4 dm91 = dev_inter_001(xi, sj[1]);
	  float4 dm92 = dev_inter_001(xi, sj[2]);
	  float4 dm93 = dev_inter_001(xi, sj[3]);
	  float4 dm94 = dev_inter_001(xi, sj[4]);
	  float4 dm95 = dev_inter_001(xi, sj[5]);
	  float4 dm96 = dev_inter_001(xi, sj[6]);
	  float4 dm97 = dev_inter_001(xi, sj[7]);
	  float4 dm98 = dev_inter_001(xi, sj[8]);
	  float4 dm99 = dev_inter_001(xi, sj[9]);

	  float4 dm100 = dev_inter_001(xi, sj[0]);
	  float4 dm101 = dev_inter_001(xi, sj[1]);
	  float4 dm102 = dev_inter_001(xi, sj[2]);
	  float4 dm103 = dev_inter_001(xi, sj[3]);
	  float4 dm104 = dev_inter_001(xi, sj[4]);
	  float4 dm105 = dev_inter_001(xi, sj[5]);
	  float4 dm106 = dev_inter_001(xi, sj[6]);
	  float4 dm107 = dev_inter_001(xi, sj[7]);
	  float4 dm108 = dev_inter_001(xi, sj[8]);
	  float4 dm109 = dev_inter_001(xi, sj[9]);

	  float4 dm110 = dev_inter_001(xi, sj[0]);
	  float4 dm111 = dev_inter_001(xi, sj[1]);
	  float4 dm112 = dev_inter_001(xi, sj[2]);
	  float4 dm113 = dev_inter_001(xi, sj[3]);
	  float4 dm114 = dev_inter_001(xi, sj[4]);
	  float4 dm115 = dev_inter_001(xi, sj[5]);
	  float4 dm116 = dev_inter_001(xi, sj[6]);
	  float4 dm117 = dev_inter_001(xi, sj[7]);
	  float4 dm118 = dev_inter_001(xi, sj[8]);
	  float4 dm119 = dev_inter_001(xi, sj[9]);

	  float4 dm120 = dev_inter_001(xi, sj[0]);
	  float4 dm121 = dev_inter_001(xi, sj[1]);
	  float4 dm122 = dev_inter_001(xi, sj[2]);
	  float4 dm123 = dev_inter_001(xi, sj[3]);
	  float4 dm124 = dev_inter_001(xi, sj[4]);
	  float4 dm125 = dev_inter_001(xi, sj[5]);
	  float4 dm126 = dev_inter_001(xi, sj[6]);
	  float4 dm127 = dev_inter_001(xi, sj[7]);


#if (ACC_TYPE==1)
	  apot.x += \
	    (dm0.x*dm0.w) + (dm1.x*dm1.w) + (dm2.x*dm2.w) + (dm3.x*dm3.w) + (dm4.x*dm4.w) + (dm5.x*dm5.w) + (dm6.x*dm6.w) + (dm7.x*dm7.w) + (dm8.x*dm8.w) + (dm9.x*dm9.w) + \
	    (dm10.x*dm10.w) + (dm11.x*dm11.w) + (dm12.x*dm12.w) + (dm13.x*dm13.w) + (dm14.x*dm14.w) + (dm15.x*dm15.w) + (dm16.x*dm16.w) + (dm17.x*dm17.w) + (dm18.x*dm18.w) + (dm19.x*dm19.w) + \
	    (dm20.x*dm20.w) + (dm21.x*dm21.w) + (dm22.x*dm22.w) + (dm23.x*dm23.w) + (dm24.x*dm24.w) + (dm25.x*dm25.w) + (dm26.x*dm26.w) + (dm27.x*dm27.w) + (dm28.x*dm28.w) + (dm29.x*dm29.w) + \
	    (dm30.x*dm30.w) + (dm31.x*dm31.w) + (dm32.x*dm32.w) + (dm33.x*dm33.w) + (dm34.x*dm34.w) + (dm35.x*dm35.w) + (dm36.x*dm36.w) + (dm37.x*dm37.w) + (dm38.x*dm38.w) + (dm39.x*dm39.w) + \
	    (dm40.x*dm40.w) + (dm41.x*dm41.w) + (dm42.x*dm42.w) + (dm43.x*dm43.w) + (dm44.x*dm44.w) + (dm45.x*dm45.w) + (dm46.x*dm46.w) + (dm47.x*dm47.w) + (dm48.x*dm48.w) + (dm49.x*dm49.w) + \
	    (dm50.x*dm50.w) + (dm51.x*dm51.w) + (dm52.x*dm52.w) + (dm53.x*dm53.w) + (dm54.x*dm54.w) + (dm55.x*dm55.w) + (dm56.x*dm56.w) + (dm57.x*dm57.w) + (dm58.x*dm58.w) + (dm59.x*dm59.w) + \
	    (dm60.x*dm60.w) + (dm61.x*dm61.w) + (dm62.x*dm62.w) + (dm63.x*dm63.w) + (dm64.x*dm64.w) + (dm65.x*dm65.w) + (dm66.x*dm66.w) + (dm67.x*dm67.w) + (dm68.x*dm68.w) + (dm69.x*dm69.w) + \
	    (dm70.x*dm70.w) + (dm71.x*dm71.w) + (dm72.x*dm72.w) + (dm73.x*dm73.w) + (dm74.x*dm74.w) + (dm75.x*dm75.w) + (dm76.x*dm76.w) + (dm77.x*dm77.w) + (dm78.x*dm78.w) + (dm79.x*dm79.w) + \
	    (dm80.x*dm80.w) + (dm81.x*dm81.w) + (dm82.x*dm82.w) + (dm83.x*dm83.w) + (dm84.x*dm84.w) + (dm85.x*dm85.w) + (dm86.x*dm86.w) + (dm87.x*dm87.w) + (dm88.x*dm88.w) + (dm89.x*dm89.w) + \
	    (dm90.x*dm90.w) + (dm91.x*dm91.w) + (dm92.x*dm92.w) + (dm93.x*dm93.w) + (dm94.x*dm94.w) + (dm95.x*dm95.w) + (dm96.x*dm96.w) + (dm97.x*dm97.w) + (dm98.x*dm98.w) + (dm99.x*dm99.w) + \
	    (dm100.x*dm100.w) + (dm101.x*dm101.w) + (dm102.x*dm102.w) + (dm103.x*dm103.w) + (dm104.x*dm104.w) + (dm105.x*dm105.w) + (dm106.x*dm106.w) + (dm107.x*dm107.w) + (dm108.x*dm108.w) + (dm109.x*dm109.w) + \
	    (dm110.x*dm110.w) + (dm111.x*dm111.w) + (dm112.x*dm112.w) + (dm113.x*dm113.w) + (dm114.x*dm114.w) + (dm115.x*dm115.w) + (dm116.x*dm116.w) + (dm117.x*dm117.w) + (dm118.x*dm118.w) + (dm119.x*dm119.w) + \
	    (dm120.x*dm120.w) + (dm121.x*dm121.w) + (dm122.x*dm122.w) + (dm123.x*dm123.w) + (dm124.x*dm124.w) + (dm125.x*dm125.w) + (dm126.x*dm126.w) + (dm127.x*dm127.w);

	  apot.y += \
	    (dm0.y*dm0.w) + (dm1.y*dm1.w) + (dm2.y*dm2.w) + (dm3.y*dm3.w) + (dm4.y*dm4.w) + (dm5.y*dm5.w) + (dm6.y*dm6.w) + (dm7.y*dm7.w) + (dm8.y*dm8.w) + (dm9.y*dm9.w) + \
	    (dm10.y*dm10.w) + (dm11.y*dm11.w) + (dm12.y*dm12.w) + (dm13.y*dm13.w) + (dm14.y*dm14.w) + (dm15.y*dm15.w) + (dm16.y*dm16.w) + (dm17.y*dm17.w) + (dm18.y*dm18.w) + (dm19.y*dm19.w) + \
	    (dm20.y*dm20.w) + (dm21.y*dm21.w) + (dm22.y*dm22.w) + (dm23.y*dm23.w) + (dm24.y*dm24.w) + (dm25.y*dm25.w) + (dm26.y*dm26.w) + (dm27.y*dm27.w) + (dm28.y*dm28.w) + (dm29.y*dm29.w) + \
	    (dm30.y*dm30.w) + (dm31.y*dm31.w) + (dm32.y*dm32.w) + (dm33.y*dm33.w) + (dm34.y*dm34.w) + (dm35.y*dm35.w) + (dm36.y*dm36.w) + (dm37.y*dm37.w) + (dm38.y*dm38.w) + (dm39.y*dm39.w) + \
	    (dm40.y*dm40.w) + (dm41.y*dm41.w) + (dm42.y*dm42.w) + (dm43.y*dm43.w) + (dm44.y*dm44.w) + (dm45.y*dm45.w) + (dm46.y*dm46.w) + (dm47.y*dm47.w) + (dm48.y*dm48.w) + (dm49.y*dm49.w) + \
	    (dm50.y*dm50.w) + (dm51.y*dm51.w) + (dm52.y*dm52.w) + (dm53.y*dm53.w) + (dm54.y*dm54.w) + (dm55.y*dm55.w) + (dm56.y*dm56.w) + (dm57.y*dm57.w) + (dm58.y*dm58.w) + (dm59.y*dm59.w) + \
	    (dm60.y*dm60.w) + (dm61.y*dm61.w) + (dm62.y*dm62.w) + (dm63.y*dm63.w) + (dm64.y*dm64.w) + (dm65.y*dm65.w) + (dm66.y*dm66.w) + (dm67.y*dm67.w) + (dm68.y*dm68.w) + (dm69.y*dm69.w) + \
	    (dm70.y*dm70.w) + (dm71.y*dm71.w) + (dm72.y*dm72.w) + (dm73.y*dm73.w) + (dm74.y*dm74.w) + (dm75.y*dm75.w) + (dm76.y*dm76.w) + (dm77.y*dm77.w) + (dm78.y*dm78.w) + (dm79.y*dm79.w) + \
	    (dm80.y*dm80.w) + (dm81.y*dm81.w) + (dm82.y*dm82.w) + (dm83.y*dm83.w) + (dm84.y*dm84.w) + (dm85.y*dm85.w) + (dm86.y*dm86.w) + (dm87.y*dm87.w) + (dm88.y*dm88.w) + (dm89.y*dm89.w) + \
	    (dm90.y*dm90.w) + (dm91.y*dm91.w) + (dm92.y*dm92.w) + (dm93.y*dm93.w) + (dm94.y*dm94.w) + (dm95.y*dm95.w) + (dm96.y*dm96.w) + (dm97.y*dm97.w) + (dm98.y*dm98.w) + (dm99.y*dm99.w) + \
	    (dm100.y*dm100.w) + (dm101.y*dm101.w) + (dm102.y*dm102.w) + (dm103.y*dm103.w) + (dm104.y*dm104.w) + (dm105.y*dm105.w) + (dm106.y*dm106.w) + (dm107.y*dm107.w) + (dm108.y*dm108.w) + (dm109.y*dm109.w) + \
	    (dm110.y*dm110.w) + (dm111.y*dm111.w) + (dm112.y*dm112.w) + (dm113.y*dm113.w) + (dm114.y*dm114.w) + (dm115.y*dm115.w) + (dm116.y*dm116.w) + (dm117.y*dm117.w) + (dm118.y*dm118.w) + (dm119.y*dm119.w) + \
	    (dm120.y*dm120.w) + (dm121.y*dm121.w) + (dm122.y*dm122.w) + (dm123.y*dm123.w) + (dm124.y*dm124.w) + (dm125.y*dm125.w) + (dm126.y*dm126.w) + (dm127.y*dm127.w);

	  apot.z += \
	    (dm0.z*dm0.w) + (dm1.z*dm1.w) + (dm2.z*dm2.w) + (dm3.z*dm3.w) + (dm4.z*dm4.w) + (dm5.z*dm5.w) + (dm6.z*dm6.w) + (dm7.z*dm7.w) + (dm8.z*dm8.w) + (dm9.z*dm9.w) + \
	    (dm10.z*dm10.w) + (dm11.z*dm11.w) + (dm12.z*dm12.w) + (dm13.z*dm13.w) + (dm14.z*dm14.w) + (dm15.z*dm15.w) + (dm16.z*dm16.w) + (dm17.z*dm17.w) + (dm18.z*dm18.w) + (dm19.z*dm19.w) + \
	    (dm20.z*dm20.w) + (dm21.z*dm21.w) + (dm22.z*dm22.w) + (dm23.z*dm23.w) + (dm24.z*dm24.w) + (dm25.z*dm25.w) + (dm26.z*dm26.w) + (dm27.z*dm27.w) + (dm28.z*dm28.w) + (dm29.z*dm29.w) + \
	    (dm30.z*dm30.w) + (dm31.z*dm31.w) + (dm32.z*dm32.w) + (dm33.z*dm33.w) + (dm34.z*dm34.w) + (dm35.z*dm35.w) + (dm36.z*dm36.w) + (dm37.z*dm37.w) + (dm38.z*dm38.w) + (dm39.z*dm39.w) + \
	    (dm40.z*dm40.w) + (dm41.z*dm41.w) + (dm42.z*dm42.w) + (dm43.z*dm43.w) + (dm44.z*dm44.w) + (dm45.z*dm45.w) + (dm46.z*dm46.w) + (dm47.z*dm47.w) + (dm48.z*dm48.w) + (dm49.z*dm49.w) + \
	    (dm50.z*dm50.w) + (dm51.z*dm51.w) + (dm52.z*dm52.w) + (dm53.z*dm53.w) + (dm54.z*dm54.w) + (dm55.z*dm55.w) + (dm56.z*dm56.w) + (dm57.z*dm57.w) + (dm58.z*dm58.w) + (dm59.z*dm59.w) + \
	    (dm60.z*dm60.w) + (dm61.z*dm61.w) + (dm62.z*dm62.w) + (dm63.z*dm63.w) + (dm64.z*dm64.w) + (dm65.z*dm65.w) + (dm66.z*dm66.w) + (dm67.z*dm67.w) + (dm68.z*dm68.w) + (dm69.z*dm69.w) + \
	    (dm70.z*dm70.w) + (dm71.z*dm71.w) + (dm72.z*dm72.w) + (dm73.z*dm73.w) + (dm74.z*dm74.w) + (dm75.z*dm75.w) + (dm76.z*dm76.w) + (dm77.z*dm77.w) + (dm78.z*dm78.w) + (dm79.z*dm79.w) + \
	    (dm80.z*dm80.w) + (dm81.z*dm81.w) + (dm82.z*dm82.w) + (dm83.z*dm83.w) + (dm84.z*dm84.w) + (dm85.z*dm85.w) + (dm86.z*dm86.w) + (dm87.z*dm87.w) + (dm88.z*dm88.w) + (dm89.z*dm89.w) + \
	    (dm90.z*dm90.w) + (dm91.z*dm91.w) + (dm92.z*dm92.w) + (dm93.z*dm93.w) + (dm94.z*dm94.w) + (dm95.z*dm95.w) + (dm96.z*dm96.w) + (dm97.z*dm97.w) + (dm98.z*dm98.w) + (dm99.z*dm99.w) + \
	    (dm100.z*dm100.w) + (dm101.z*dm101.w) + (dm102.z*dm102.w) + (dm103.z*dm103.w) + (dm104.z*dm104.w) + (dm105.z*dm105.w) + (dm106.z*dm106.w) + (dm107.z*dm107.w) + (dm108.z*dm108.w) + (dm109.z*dm109.w) + \
	    (dm110.z*dm110.w) + (dm111.z*dm111.w) + (dm112.z*dm112.w) + (dm113.z*dm113.w) + (dm114.z*dm114.w) + (dm115.z*dm115.w) + (dm116.z*dm116.w) + (dm117.z*dm117.w) + (dm118.z*dm118.w) + (dm119.z*dm119.w) + \
	    (dm120.z*dm120.w) + (dm121.z*dm121.w) + (dm122.z*dm122.w) + (dm123.z*dm123.w) + (dm124.z*dm124.w) + (dm125.z*dm125.w) + (dm126.z*dm126.w) + (dm127.z*dm127.w);

#else 
	  apot.x += \
	    (dm0.x*dm0.w + dm1.x*dm1.w + dm2.x*dm2.w + dm3.x*dm3.w + dm4.x*dm4.w + dm5.x*dm5.w + dm6.x*dm6.w + dm7.x*dm7.w + dm8.x*dm8.w + dm9.x*dm9.w + \
	     dm10.x*dm10.w + dm11.x*dm11.w + dm12.x*dm12.w + dm13.x*dm13.w + dm14.x*dm14.w + dm15.x*dm15.w + dm16.x*dm16.w + dm17.x*dm17.w + dm18.x*dm18.w + dm19.x*dm19.w + \
	     dm20.x*dm20.w + dm21.x*dm21.w + dm22.x*dm22.w + dm23.x*dm23.w + dm24.x*dm24.w + dm25.x*dm25.w + dm26.x*dm26.w + dm27.x*dm27.w + dm28.x*dm28.w + dm29.x*dm29.w + \
	     dm30.x*dm30.w + dm31.x*dm31.w + dm32.x*dm32.w + dm33.x*dm33.w + dm34.x*dm34.w + dm35.x*dm35.w + dm36.x*dm36.w + dm37.x*dm37.w + dm38.x*dm38.w + dm39.x*dm39.w + \
	     dm40.x*dm40.w + dm41.x*dm41.w + dm42.x*dm42.w + dm43.x*dm43.w + dm44.x*dm44.w + dm45.x*dm45.w + dm46.x*dm46.w + dm47.x*dm47.w + dm48.x*dm48.w + dm49.x*dm49.w + \
	     dm50.x*dm50.w + dm51.x*dm51.w + dm52.x*dm52.w + dm53.x*dm53.w + dm54.x*dm54.w + dm55.x*dm55.w + dm56.x*dm56.w + dm57.x*dm57.w + dm58.x*dm58.w + dm59.x*dm59.w + \
	     dm60.x*dm60.w + dm61.x*dm61.w + dm62.x*dm62.w + dm63.x*dm63.w) +\
	    (dm64.x*dm64.w + dm65.x*dm65.w + dm66.x*dm66.w + dm67.x*dm67.w + dm68.x*dm68.w + dm69.x*dm69.w + \
	     dm70.x*dm70.w + dm71.x*dm71.w + dm72.x*dm72.w + dm73.x*dm73.w + dm74.x*dm74.w + dm75.x*dm75.w + dm76.x*dm76.w + dm77.x*dm77.w + dm78.x*dm78.w + dm79.x*dm79.w + \
	     dm80.x*dm80.w + dm81.x*dm81.w + dm82.x*dm82.w + dm83.x*dm83.w + dm84.x*dm84.w + dm85.x*dm85.w + dm86.x*dm86.w + dm87.x*dm87.w + dm88.x*dm88.w + dm89.x*dm89.w + \
	     dm90.x*dm90.w + dm91.x*dm91.w + dm92.x*dm92.w + dm93.x*dm93.w + dm94.x*dm94.w + dm95.x*dm95.w + dm96.x*dm96.w + dm97.x*dm97.w + dm98.x*dm98.w + dm99.x*dm99.w + \
	     dm100.x*dm100.w + dm101.x*dm101.w + dm102.x*dm102.w + dm103.x*dm103.w + dm104.x*dm104.w + dm105.x*dm105.w + dm106.x*dm106.w + dm107.x*dm107.w + dm108.x*dm108.w + dm109.x*dm109.w + \
	     dm110.x*dm110.w + dm111.x*dm111.w + dm112.x*dm112.w + dm113.x*dm113.w + dm114.x*dm114.w + dm115.x*dm115.w + dm116.x*dm116.w + dm117.x*dm117.w + dm118.x*dm118.w + dm119.x*dm119.w + \
	     dm120.x*dm120.w + dm121.x*dm121.w + dm122.x*dm122.w + dm123.x*dm123.w + dm124.x*dm124.w + dm125.x*dm125.w + dm126.x*dm126.w + dm127.x*dm127.w);

	  apot.y += \
	    (dm0.y*dm0.w + dm1.y*dm1.w + dm2.y*dm2.w + dm3.y*dm3.w + dm4.y*dm4.w + dm5.y*dm5.w + dm6.y*dm6.w + dm7.y*dm7.w + dm8.y*dm8.w + dm9.y*dm9.w + \
	     dm10.y*dm10.w + dm11.y*dm11.w + dm12.y*dm12.w + dm13.y*dm13.w + dm14.y*dm14.w + dm15.y*dm15.w + dm16.y*dm16.w + dm17.y*dm17.w + dm18.y*dm18.w + dm19.y*dm19.w + \
	     dm20.y*dm20.w + dm21.y*dm21.w + dm22.y*dm22.w + dm23.y*dm23.w + dm24.y*dm24.w + dm25.y*dm25.w + dm26.y*dm26.w + dm27.y*dm27.w + dm28.y*dm28.w + dm29.y*dm29.w + \
	     dm30.y*dm30.w + dm31.y*dm31.w + dm32.y*dm32.w + dm33.y*dm33.w + dm34.y*dm34.w + dm35.y*dm35.w + dm36.y*dm36.w + dm37.y*dm37.w + dm38.y*dm38.w + dm39.y*dm39.w + \
	     dm40.y*dm40.w + dm41.y*dm41.w + dm42.y*dm42.w + dm43.y*dm43.w + dm44.y*dm44.w + dm45.y*dm45.w + dm46.y*dm46.w + dm47.y*dm47.w + dm48.y*dm48.w + dm49.y*dm49.w + \
	     dm50.y*dm50.w + dm51.y*dm51.w + dm52.y*dm52.w + dm53.y*dm53.w + dm54.y*dm54.w + dm55.y*dm55.w + dm56.y*dm56.w + dm57.y*dm57.w + dm58.y*dm58.w + dm59.y*dm59.w + \
	     dm60.y*dm60.w + dm61.y*dm61.w + dm62.y*dm62.w + dm63.y*dm63.w) +\
	    (dm64.x*dm64.w + dm65.x*dm65.w + dm66.x*dm66.w + dm67.x*dm67.w + dm68.x*dm68.w + dm69.x*dm69.w + \
	     dm70.y*dm70.w + dm71.y*dm71.w + dm72.y*dm72.w + dm73.y*dm73.w + dm74.y*dm74.w + dm75.y*dm75.w + dm76.y*dm76.w + dm77.y*dm77.w + dm78.y*dm78.w + dm79.y*dm79.w + \
	     dm80.y*dm80.w + dm81.y*dm81.w + dm82.y*dm82.w + dm83.y*dm83.w + dm84.y*dm84.w + dm85.y*dm85.w + dm86.y*dm86.w + dm87.y*dm87.w + dm88.y*dm88.w + dm89.y*dm89.w + \
	     dm90.y*dm90.w + dm91.y*dm91.w + dm92.y*dm92.w + dm93.y*dm93.w + dm94.y*dm94.w + dm95.y*dm95.w + dm96.y*dm96.w + dm97.y*dm97.w + dm98.y*dm98.w + dm99.y*dm99.w + \
	     dm100.y*dm100.w + dm101.y*dm101.w + dm102.y*dm102.w + dm103.y*dm103.w + dm104.y*dm104.w + dm105.y*dm105.w + dm106.y*dm106.w + dm107.y*dm107.w + dm108.y*dm108.w + dm109.y*dm109.w + \
	     dm110.y*dm110.w + dm111.y*dm111.w + dm112.y*dm112.w + dm113.y*dm113.w + dm114.y*dm114.w + dm115.y*dm115.w + dm116.y*dm116.w + dm117.y*dm117.w + dm118.y*dm118.w + dm119.y*dm119.w + \
	     dm120.y*dm120.w + dm121.y*dm121.w + dm122.y*dm122.w + dm123.y*dm123.w + dm124.y*dm124.w + dm125.y*dm125.w + dm126.y*dm126.w + dm127.y*dm127.w);

	  apot.z += \
	    (dm0.z*dm0.w + dm1.z*dm1.w + dm2.z*dm2.w + dm3.z*dm3.w + dm4.z*dm4.w + dm5.z*dm5.w + dm6.z*dm6.w + dm7.z*dm7.w + dm8.z*dm8.w + dm9.z*dm9.w + \
	     dm10.z*dm10.w + dm11.z*dm11.w + dm12.z*dm12.w + dm13.z*dm13.w + dm14.z*dm14.w + dm15.z*dm15.w + dm16.z*dm16.w + dm17.z*dm17.w + dm18.z*dm18.w + dm19.z*dm19.w + \
	     dm20.z*dm20.w + dm21.z*dm21.w + dm22.z*dm22.w + dm23.z*dm23.w + dm24.z*dm24.w + dm25.z*dm25.w + dm26.z*dm26.w + dm27.z*dm27.w + dm28.z*dm28.w + dm29.z*dm29.w + \
	     dm30.z*dm30.w + dm31.z*dm31.w + dm32.z*dm32.w + dm33.z*dm33.w + dm34.z*dm34.w + dm35.z*dm35.w + dm36.z*dm36.w + dm37.z*dm37.w + dm38.z*dm38.w + dm39.z*dm39.w + \
	     dm40.z*dm40.w + dm41.z*dm41.w + dm42.z*dm42.w + dm43.z*dm43.w + dm44.z*dm44.w + dm45.z*dm45.w + dm46.z*dm46.w + dm47.z*dm47.w + dm48.z*dm48.w + dm49.z*dm49.w + \
	     dm50.z*dm50.w + dm51.z*dm51.w + dm52.z*dm52.w + dm53.z*dm53.w + dm54.z*dm54.w + dm55.z*dm55.w + dm56.z*dm56.w + dm57.z*dm57.w + dm58.z*dm58.w + dm59.z*dm59.w + \
	     dm60.z*dm60.w + dm61.z*dm61.w + dm62.z*dm62.w + dm63.z*dm63.w) +\
	    (dm64.z*dm64.w + dm65.z*dm65.w + dm66.z*dm66.w + dm67.z*dm67.w + dm68.z*dm68.w + dm69.z*dm69.w + \
	     dm70.z*dm70.w + dm71.z*dm71.w + dm72.z*dm72.w + dm73.z*dm73.w + dm74.z*dm74.w + dm75.z*dm75.w + dm76.z*dm76.w + dm77.z*dm77.w + dm78.z*dm78.w + dm79.z*dm79.w + \
	     dm80.z*dm80.w + dm81.z*dm81.w + dm82.z*dm82.w + dm83.z*dm83.w + dm84.z*dm84.w + dm85.z*dm85.w + dm86.z*dm86.w + dm87.z*dm87.w + dm88.z*dm88.w + dm89.z*dm89.w + \
	     dm90.z*dm90.w + dm91.z*dm91.w + dm92.z*dm92.w + dm93.z*dm93.w + dm94.z*dm94.w + dm95.z*dm95.w + dm96.z*dm96.w + dm97.z*dm97.w + dm98.z*dm98.w + dm99.z*dm99.w + \
	     dm100.z*dm100.w + dm101.z*dm101.w + dm102.z*dm102.w + dm103.z*dm103.w + dm104.z*dm104.w + dm105.z*dm105.w + dm106.z*dm106.w + dm107.z*dm107.w + dm108.z*dm108.w + dm109.z*dm109.w + \
	     dm110.z*dm110.w + dm111.z*dm111.w + dm112.z*dm112.w + dm113.z*dm113.w + dm114.z*dm114.w + dm115.z*dm115.w + dm116.z*dm116.w + dm117.z*dm117.w + dm118.z*dm118.w + dm119.z*dm119.w + \
	     dm120.z*dm120.w + dm121.z*dm121.w + dm122.z*dm122.w + dm123.z*dm123.w + dm124.z*dm124.w + dm125.z*dm125.w + dm126.z*dm126.w + dm127.z*dm127.w);
#endif

	}
      }
      if(ibase + tid < iend){
	apotlist[ibase + tid] = apot;
      }
    }
  }



  __device__
  float4 dev_inter_011(float4 xi, float4 xj, float* pot)
  {
    float dx = xj.x - xi.x;
    float dy = xj.y - xi.y;
    float dz = xj.z - xi.z;
    float eps2 = xi.w;
    float mj   = xj.w;
    float r2 = (dx*dx + eps2) + dy*dy + dz*dz;
    float r1i  = 1/sqrt(r2);

    if(r2 == eps2) mj = 0.0;
    float r2i = r1i*r1i;
    float mr3i = mj * r1i * r2i;
    float4 retval;

    retval.x = dx; 
    retval.y = dy; 
    retval.z = dz;
    retval.w = mr3i; 
    (*pot) -= mj * r1i;

    return (retval);
  }


  __global__ 
  void cunbody_kernel_tree_011(float4 *xilist,
				float4 *xjlist,
				float4 *apotlist,
				unsigned int *ioffset,
				unsigned int *joffset)
  {
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int ibegin = ioffset[bid];
    unsigned int iend   = ioffset[bid+1];
    unsigned int jbegin = joffset[bid];
    unsigned int jend   = joffset[bid+1];
    for(unsigned int ibase = ibegin; ibase < iend; ibase += NTHRE){
      float4 xi = xilist[ibase + tid];
      float4 apot = make_float4(0,0,0,0);
      float pot0 = 0.0;
      float pot1 = 0.0;
      float pot2 = 0.0;
      float pot3 = 0.0;
      float pot4 = 0.0;
      float pot5 = 0.0;
      float pot6 = 0.0;
      float pot7 = 0.0;

      for(unsigned int jbase = jbegin; jbase < jend; jbase += NTHRE){
	__shared__ float4 sj[NTHRE];
	sj[tid] = xjlist[jbase + tid];
	__syncthreads();
	{
	  float4 dm0, dm1, dm2, dm3, dm4, dm5, dm6, dm7, dm8, dm9, dm10, dm11, dm12, dm13, dm14, dm15;
	  dm0 = dev_inter_011(xi, sj[0], &pot0);
	  dm1 = dev_inter_011(xi, sj[1], &pot0);
	  dm2 = dev_inter_011(xi, sj[2], &pot0);
	  dm3 = dev_inter_011(xi, sj[3], &pot0);
	  dm4 = dev_inter_011(xi, sj[4], &pot0);
	  dm5 = dev_inter_011(xi, sj[5], &pot0);
	  dm6 = dev_inter_011(xi, sj[6], &pot0);
	  dm7 = dev_inter_011(xi, sj[7], &pot0);
	  dm8 = dev_inter_011(xi, sj[8], &pot0);
	  dm9 = dev_inter_011(xi, sj[9], &pot0);
	  dm10 = dev_inter_011(xi, sj[10], &pot0);
	  dm11 = dev_inter_011(xi, sj[11], &pot0);
	  dm12 = dev_inter_011(xi, sj[12], &pot0);
	  dm13 = dev_inter_011(xi, sj[13], &pot0);
	  dm14 = dev_inter_011(xi, sj[14], &pot0);
	  dm15 = dev_inter_011(xi, sj[15], &pot0);
	  apot.x += (dm0.x*dm0.w) + (dm1.x*dm1.w) + (dm2.x*dm2.w) + (dm3.x*dm3.w) + (dm4.x*dm4.w) + (dm5.x*dm5.w) + (dm6.x*dm6.w) + (dm7.x*dm7.w) \
	    + (dm8.x*dm8.w) + (dm9.x*dm9.w) + (dm10.x*dm10.w) + (dm11.x*dm11.w) + (dm12.x*dm12.w) + (dm13.x*dm13.w) + (dm14.x*dm14.w) + (dm15.x*dm15.w);
	  apot.y += (dm0.y*dm0.w) + (dm1.y*dm1.w) + (dm2.y*dm2.w) + (dm3.y*dm3.w) + (dm4.y*dm4.w) + (dm5.y*dm5.w) + (dm6.y*dm6.w) + (dm7.y*dm7.w) \
	    + (dm8.y*dm8.w) + (dm9.y*dm9.w) + (dm10.y*dm10.w) + (dm11.y*dm11.w) + (dm12.y*dm12.w) + (dm13.y*dm13.w) + (dm14.y*dm14.w) + (dm15.y*dm15.w);
	  apot.z += (dm0.z*dm0.w) + (dm1.z*dm1.w) + (dm2.z*dm2.w) + (dm3.z*dm3.w) + (dm4.z*dm4.w) + (dm5.z*dm5.w) + (dm6.z*dm6.w) + (dm7.z*dm7.w) \
	    + (dm8.z*dm8.w) + (dm9.z*dm9.w) + (dm10.z*dm10.w) + (dm11.z*dm11.w) + (dm12.z*dm12.w) + (dm13.z*dm13.w) + (dm14.z*dm14.w) + (dm15.z*dm15.w);
	}

	{
	  float4 dm0, dm1, dm2, dm3, dm4, dm5, dm6, dm7, dm8, dm9, dm10, dm11, dm12, dm13, dm14, dm15;
	  dm0 = dev_inter_011(xi, sj[16], &pot1);
	  dm1 = dev_inter_011(xi, sj[17], &pot1);
	  dm2 = dev_inter_011(xi, sj[18], &pot1);
	  dm3 = dev_inter_011(xi, sj[19], &pot1);
	  dm4 = dev_inter_011(xi, sj[20], &pot1);
	  dm5 = dev_inter_011(xi, sj[21], &pot1);
	  dm6 = dev_inter_011(xi, sj[22], &pot1);
	  dm7 = dev_inter_011(xi, sj[23], &pot1);
	  dm8 = dev_inter_011(xi, sj[24], &pot1);
	  dm9 = dev_inter_011(xi, sj[25], &pot1);
	  dm10 = dev_inter_011(xi, sj[26], &pot1);
	  dm11 = dev_inter_011(xi, sj[27], &pot1);
	  dm12 = dev_inter_011(xi, sj[28], &pot1);
	  dm13 = dev_inter_011(xi, sj[29], &pot1);
	  dm14 = dev_inter_011(xi, sj[30], &pot1);
	  dm15 = dev_inter_011(xi, sj[31], &pot1);
	  apot.x += (dm0.x*dm0.w) + (dm1.x*dm1.w) + (dm2.x*dm2.w) + (dm3.x*dm3.w) + (dm4.x*dm4.w) + (dm5.x*dm5.w) + (dm6.x*dm6.w) + (dm7.x*dm7.w) \
	    + (dm8.x*dm8.w) + (dm9.x*dm9.w) + (dm10.x*dm10.w) + (dm11.x*dm11.w) + (dm12.x*dm12.w) + (dm13.x*dm13.w) + (dm14.x*dm14.w) + (dm15.x*dm15.w);
	  apot.y += (dm0.y*dm0.w) + (dm1.y*dm1.w) + (dm2.y*dm2.w) + (dm3.y*dm3.w) + (dm4.y*dm4.w) + (dm5.y*dm5.w) + (dm6.y*dm6.w) + (dm7.y*dm7.w) \
	    + (dm8.y*dm8.w) + (dm9.y*dm9.w) + (dm10.y*dm10.w) + (dm11.y*dm11.w) + (dm12.y*dm12.w) + (dm13.y*dm13.w) + (dm14.y*dm14.w) + (dm15.y*dm15.w);
	  apot.z += (dm0.z*dm0.w) + (dm1.z*dm1.w) + (dm2.z*dm2.w) + (dm3.z*dm3.w) + (dm4.z*dm4.w) + (dm5.z*dm5.w) + (dm6.z*dm6.w) + (dm7.z*dm7.w) \
	    + (dm8.z*dm8.w) + (dm9.z*dm9.w) + (dm10.z*dm10.w) + (dm11.z*dm11.w) + (dm12.z*dm12.w) + (dm13.z*dm13.w) + (dm14.z*dm14.w) + (dm15.z*dm15.w);
	}

	{
	  float4 dm0, dm1, dm2, dm3, dm4, dm5, dm6, dm7, dm8, dm9, dm10, dm11, dm12, dm13, dm14, dm15;
	  dm0 = dev_inter_011(xi, sj[32], &pot2);
	  dm1 = dev_inter_011(xi, sj[33], &pot2);
	  dm2 = dev_inter_011(xi, sj[34], &pot2);
	  dm3 = dev_inter_011(xi, sj[35], &pot2);
	  dm4 = dev_inter_011(xi, sj[36], &pot2);
	  dm5 = dev_inter_011(xi, sj[37], &pot2);
	  dm6 = dev_inter_011(xi, sj[38], &pot2);
	  dm7 = dev_inter_011(xi, sj[39], &pot2);
	  dm8 = dev_inter_011(xi, sj[40], &pot2);
	  dm9 = dev_inter_011(xi, sj[41], &pot2);
	  dm10 = dev_inter_011(xi, sj[42], &pot2);
	  dm11 = dev_inter_011(xi, sj[43], &pot2);
	  dm12 = dev_inter_011(xi, sj[44], &pot2);
	  dm13 = dev_inter_011(xi, sj[45], &pot2);
	  dm14 = dev_inter_011(xi, sj[46], &pot2);
	  dm15 = dev_inter_011(xi, sj[47], &pot2);
	  apot.x += (dm0.x*dm0.w) + (dm1.x*dm1.w) + (dm2.x*dm2.w) + (dm3.x*dm3.w) + (dm4.x*dm4.w) + (dm5.x*dm5.w) + (dm6.x*dm6.w) + (dm7.x*dm7.w) \
	    + (dm8.x*dm8.w) + (dm9.x*dm9.w) + (dm10.x*dm10.w) + (dm11.x*dm11.w) + (dm12.x*dm12.w) + (dm13.x*dm13.w) + (dm14.x*dm14.w) + (dm15.x*dm15.w);
	  apot.y += (dm0.y*dm0.w) + (dm1.y*dm1.w) + (dm2.y*dm2.w) + (dm3.y*dm3.w) + (dm4.y*dm4.w) + (dm5.y*dm5.w) + (dm6.y*dm6.w) + (dm7.y*dm7.w) \
	    + (dm8.y*dm8.w) + (dm9.y*dm9.w) + (dm10.y*dm10.w) + (dm11.y*dm11.w) + (dm12.y*dm12.w) + (dm13.y*dm13.w) + (dm14.y*dm14.w) + (dm15.y*dm15.w);
	  apot.z += (dm0.z*dm0.w) + (dm1.z*dm1.w) + (dm2.z*dm2.w) + (dm3.z*dm3.w) + (dm4.z*dm4.w) + (dm5.z*dm5.w) + (dm6.z*dm6.w) + (dm7.z*dm7.w) \
	    + (dm8.z*dm8.w) + (dm9.z*dm9.w) + (dm10.z*dm10.w) + (dm11.z*dm11.w) + (dm12.z*dm12.w) + (dm13.z*dm13.w) + (dm14.z*dm14.w) + (dm15.z*dm15.w);
	}

	{
	  float4 dm0, dm1, dm2, dm3, dm4, dm5, dm6, dm7, dm8, dm9, dm10, dm11, dm12, dm13, dm14, dm15;
	  dm0 = dev_inter_011(xi, sj[48], &pot3);
	  dm1 = dev_inter_011(xi, sj[49], &pot3);
	  dm2 = dev_inter_011(xi, sj[50], &pot3);
	  dm3 = dev_inter_011(xi, sj[51], &pot3);
	  dm4 = dev_inter_011(xi, sj[52], &pot3);
	  dm5 = dev_inter_011(xi, sj[53], &pot3);
	  dm6 = dev_inter_011(xi, sj[54], &pot3);
	  dm7 = dev_inter_011(xi, sj[55], &pot3);
	  dm8 = dev_inter_011(xi, sj[56], &pot3);
	  dm9 = dev_inter_011(xi, sj[57], &pot3);
	  dm10 = dev_inter_011(xi, sj[58], &pot3);
	  dm11 = dev_inter_011(xi, sj[59], &pot3);
	  dm12 = dev_inter_011(xi, sj[60], &pot3);
	  dm13 = dev_inter_011(xi, sj[61], &pot3);
	  dm14 = dev_inter_011(xi, sj[62], &pot3);
	  dm15 = dev_inter_011(xi, sj[63], &pot3);
	  apot.x += (dm0.x*dm0.w) + (dm1.x*dm1.w) + (dm2.x*dm2.w) + (dm3.x*dm3.w) + (dm4.x*dm4.w) + (dm5.x*dm5.w) + (dm6.x*dm6.w) + (dm7.x*dm7.w) \
	    + (dm8.x*dm8.w) + (dm9.x*dm9.w) + (dm10.x*dm10.w) + (dm11.x*dm11.w) + (dm12.x*dm12.w) + (dm13.x*dm13.w) + (dm14.x*dm14.w) + (dm15.x*dm15.w);
	  apot.y += (dm0.y*dm0.w) + (dm1.y*dm1.w) + (dm2.y*dm2.w) + (dm3.y*dm3.w) + (dm4.y*dm4.w) + (dm5.y*dm5.w) + (dm6.y*dm6.w) + (dm7.y*dm7.w) \
	    + (dm8.y*dm8.w) + (dm9.y*dm9.w) + (dm10.y*dm10.w) + (dm11.y*dm11.w) + (dm12.y*dm12.w) + (dm13.y*dm13.w) + (dm14.y*dm14.w) + (dm15.y*dm15.w);
	  apot.z += (dm0.z*dm0.w) + (dm1.z*dm1.w) + (dm2.z*dm2.w) + (dm3.z*dm3.w) + (dm4.z*dm4.w) + (dm5.z*dm5.w) + (dm6.z*dm6.w) + (dm7.z*dm7.w) \
	    + (dm8.z*dm8.w) + (dm9.z*dm9.w) + (dm10.z*dm10.w) + (dm11.z*dm11.w) + (dm12.z*dm12.w) + (dm13.z*dm13.w) + (dm14.z*dm14.w) + (dm15.z*dm15.w);
	}

	{
	  float4 dm0, dm1, dm2, dm3, dm4, dm5, dm6, dm7, dm8, dm9, dm10, dm11, dm12, dm13, dm14, dm15;
	  dm0 = dev_inter_011(xi, sj[64], &pot4);
	  dm1 = dev_inter_011(xi, sj[65], &pot4);
	  dm2 = dev_inter_011(xi, sj[66], &pot4);
	  dm3 = dev_inter_011(xi, sj[67], &pot4);
	  dm4 = dev_inter_011(xi, sj[68], &pot4);
	  dm5 = dev_inter_011(xi, sj[69], &pot4);
	  dm6 = dev_inter_011(xi, sj[70], &pot4);
	  dm7 = dev_inter_011(xi, sj[71], &pot4);
	  dm8 = dev_inter_011(xi, sj[72], &pot4);
	  dm9 = dev_inter_011(xi, sj[73], &pot4);
	  dm10 = dev_inter_011(xi, sj[74], &pot4);
	  dm11 = dev_inter_011(xi, sj[75], &pot4);
	  dm12 = dev_inter_011(xi, sj[76], &pot4);
	  dm13 = dev_inter_011(xi, sj[77], &pot4);
	  dm14 = dev_inter_011(xi, sj[78], &pot4);
	  dm15 = dev_inter_011(xi, sj[79], &pot4);
	  apot.x += (dm0.x*dm0.w) + (dm1.x*dm1.w) + (dm2.x*dm2.w) + (dm3.x*dm3.w) + (dm4.x*dm4.w) + (dm5.x*dm5.w) + (dm6.x*dm6.w) + (dm7.x*dm7.w) \
	    + (dm8.x*dm8.w) + (dm9.x*dm9.w) + (dm10.x*dm10.w) + (dm11.x*dm11.w) + (dm12.x*dm12.w) + (dm13.x*dm13.w) + (dm14.x*dm14.w) + (dm15.x*dm15.w);
	  apot.y += (dm0.y*dm0.w) + (dm1.y*dm1.w) + (dm2.y*dm2.w) + (dm3.y*dm3.w) + (dm4.y*dm4.w) + (dm5.y*dm5.w) + (dm6.y*dm6.w) + (dm7.y*dm7.w) \
	    + (dm8.y*dm8.w) + (dm9.y*dm9.w) + (dm10.y*dm10.w) + (dm11.y*dm11.w) + (dm12.y*dm12.w) + (dm13.y*dm13.w) + (dm14.y*dm14.w) + (dm15.y*dm15.w);
	  apot.z += (dm0.z*dm0.w) + (dm1.z*dm1.w) + (dm2.z*dm2.w) + (dm3.z*dm3.w) + (dm4.z*dm4.w) + (dm5.z*dm5.w) + (dm6.z*dm6.w) + (dm7.z*dm7.w) \
	    + (dm8.z*dm8.w) + (dm9.z*dm9.w) + (dm10.z*dm10.w) + (dm11.z*dm11.w) + (dm12.z*dm12.w) + (dm13.z*dm13.w) + (dm14.z*dm14.w) + (dm15.z*dm15.w);
	}

	{
	  float4 dm0, dm1, dm2, dm3, dm4, dm5, dm6, dm7, dm8, dm9, dm10, dm11, dm12, dm13, dm14, dm15;
	  dm0 = dev_inter_011(xi, sj[80], &pot5);
	  dm1 = dev_inter_011(xi, sj[81], &pot5);
	  dm2 = dev_inter_011(xi, sj[82], &pot5);
	  dm3 = dev_inter_011(xi, sj[83], &pot5);
	  dm4 = dev_inter_011(xi, sj[84], &pot5);
	  dm5 = dev_inter_011(xi, sj[85], &pot5);
	  dm6 = dev_inter_011(xi, sj[86], &pot5);
	  dm7 = dev_inter_011(xi, sj[87], &pot5);
	  dm8 = dev_inter_011(xi, sj[88], &pot5);
	  dm9 = dev_inter_011(xi, sj[89], &pot5);
	  dm10 = dev_inter_011(xi, sj[90], &pot5);
	  dm11 = dev_inter_011(xi, sj[91], &pot5);
	  dm12 = dev_inter_011(xi, sj[92], &pot5);
	  dm13 = dev_inter_011(xi, sj[93], &pot5);
	  dm14 = dev_inter_011(xi, sj[94], &pot5);
	  dm15 = dev_inter_011(xi, sj[95], &pot5);
	  apot.x += (dm0.x*dm0.w) + (dm1.x*dm1.w) + (dm2.x*dm2.w) + (dm3.x*dm3.w) + (dm4.x*dm4.w) + (dm5.x*dm5.w) + (dm6.x*dm6.w) + (dm7.x*dm7.w) \
	    + (dm8.x*dm8.w) + (dm9.x*dm9.w) + (dm10.x*dm10.w) + (dm11.x*dm11.w) + (dm12.x*dm12.w) + (dm13.x*dm13.w) + (dm14.x*dm14.w) + (dm15.x*dm15.w);
	  apot.y += (dm0.y*dm0.w) + (dm1.y*dm1.w) + (dm2.y*dm2.w) + (dm3.y*dm3.w) + (dm4.y*dm4.w) + (dm5.y*dm5.w) + (dm6.y*dm6.w) + (dm7.y*dm7.w) \
	    + (dm8.y*dm8.w) + (dm9.y*dm9.w) + (dm10.y*dm10.w) + (dm11.y*dm11.w) + (dm12.y*dm12.w) + (dm13.y*dm13.w) + (dm14.y*dm14.w) + (dm15.y*dm15.w);
	  apot.z += (dm0.z*dm0.w) + (dm1.z*dm1.w) + (dm2.z*dm2.w) + (dm3.z*dm3.w) + (dm4.z*dm4.w) + (dm5.z*dm5.w) + (dm6.z*dm6.w) + (dm7.z*dm7.w) \
	    + (dm8.z*dm8.w) + (dm9.z*dm9.w) + (dm10.z*dm10.w) + (dm11.z*dm11.w) + (dm12.z*dm12.w) + (dm13.z*dm13.w) + (dm14.z*dm14.w) + (dm15.z*dm15.w);
	}

	{
	  float4 dm0, dm1, dm2, dm3, dm4, dm5, dm6, dm7, dm8, dm9, dm10, dm11, dm12, dm13, dm14, dm15;
	  dm0 = dev_inter_011(xi, sj[96], &pot6);
	  dm1 = dev_inter_011(xi, sj[97], &pot6);
	  dm2 = dev_inter_011(xi, sj[98], &pot6);
	  dm3 = dev_inter_011(xi, sj[99], &pot6);
	  dm4 = dev_inter_011(xi, sj[100], &pot6);
	  dm5 = dev_inter_011(xi, sj[101], &pot6);
	  dm6 = dev_inter_011(xi, sj[102], &pot6);
	  dm7 = dev_inter_011(xi, sj[103], &pot6);
	  dm8 = dev_inter_011(xi, sj[104], &pot6);
	  dm9 = dev_inter_011(xi, sj[105], &pot6);
	  dm10 = dev_inter_011(xi, sj[106], &pot6);
	  dm11 = dev_inter_011(xi, sj[107], &pot6);
	  dm12 = dev_inter_011(xi, sj[108], &pot6);
	  dm13 = dev_inter_011(xi, sj[109], &pot6);
	  dm14 = dev_inter_011(xi, sj[110], &pot6);
	  dm15 = dev_inter_011(xi, sj[111], &pot6);
	  apot.x += (dm0.x*dm0.w) + (dm1.x*dm1.w) + (dm2.x*dm2.w) + (dm3.x*dm3.w) + (dm4.x*dm4.w) + (dm5.x*dm5.w) + (dm6.x*dm6.w) + (dm7.x*dm7.w) \
	    + (dm8.x*dm8.w) + (dm9.x*dm9.w) + (dm10.x*dm10.w) + (dm11.x*dm11.w) + (dm12.x*dm12.w) + (dm13.x*dm13.w) + (dm14.x*dm14.w) + (dm15.x*dm15.w);
	  apot.y += (dm0.y*dm0.w) + (dm1.y*dm1.w) + (dm2.y*dm2.w) + (dm3.y*dm3.w) + (dm4.y*dm4.w) + (dm5.y*dm5.w) + (dm6.y*dm6.w) + (dm7.y*dm7.w) \
	    + (dm8.y*dm8.w) + (dm9.y*dm9.w) + (dm10.y*dm10.w) + (dm11.y*dm11.w) + (dm12.y*dm12.w) + (dm13.y*dm13.w) + (dm14.y*dm14.w) + (dm15.y*dm15.w);
	  apot.z += (dm0.z*dm0.w) + (dm1.z*dm1.w) + (dm2.z*dm2.w) + (dm3.z*dm3.w) + (dm4.z*dm4.w) + (dm5.z*dm5.w) + (dm6.z*dm6.w) + (dm7.z*dm7.w) \
	    + (dm8.z*dm8.w) + (dm9.z*dm9.w) + (dm10.z*dm10.w) + (dm11.z*dm11.w) + (dm12.z*dm12.w) + (dm13.z*dm13.w) + (dm14.z*dm14.w) + (dm15.z*dm15.w);
	}

	{
	  float4 dm0, dm1, dm2, dm3, dm4, dm5, dm6, dm7, dm8, dm9, dm10, dm11, dm12, dm13, dm14, dm15;
	  dm0 = dev_inter_011(xi, sj[112], &pot7);
	  dm1 = dev_inter_011(xi, sj[113], &pot7);
	  dm2 = dev_inter_011(xi, sj[114], &pot7);
	  dm3 = dev_inter_011(xi, sj[115], &pot7);
	  dm4 = dev_inter_011(xi, sj[116], &pot7);
	  dm5 = dev_inter_011(xi, sj[117], &pot7);
	  dm6 = dev_inter_011(xi, sj[118], &pot7);
	  dm7 = dev_inter_011(xi, sj[119], &pot7);
	  dm8 = dev_inter_011(xi, sj[120], &pot7);
	  dm9 = dev_inter_011(xi, sj[121], &pot7);
	  dm10 = dev_inter_011(xi, sj[122], &pot7);
	  dm11 = dev_inter_011(xi, sj[123], &pot7);
	  dm12 = dev_inter_011(xi, sj[124], &pot7);
	  dm13 = dev_inter_011(xi, sj[125], &pot7);
	  dm14 = dev_inter_011(xi, sj[126], &pot7);
	  dm15 = dev_inter_011(xi, sj[127], &pot7);
	  apot.x += (dm0.x*dm0.w) + (dm1.x*dm1.w) + (dm2.x*dm2.w) + (dm3.x*dm3.w) + (dm4.x*dm4.w) + (dm5.x*dm5.w) + (dm6.x*dm6.w) + (dm7.x*dm7.w) \
	    + (dm8.x*dm8.w) + (dm9.x*dm9.w) + (dm10.x*dm10.w) + (dm11.x*dm11.w) + (dm12.x*dm12.w) + (dm13.x*dm13.w) + (dm14.x*dm14.w) + (dm15.x*dm15.w);
	  apot.y += (dm0.y*dm0.w) + (dm1.y*dm1.w) + (dm2.y*dm2.w) + (dm3.y*dm3.w) + (dm4.y*dm4.w) + (dm5.y*dm5.w) + (dm6.y*dm6.w) + (dm7.y*dm7.w) \
	    + (dm8.y*dm8.w) + (dm9.y*dm9.w) + (dm10.y*dm10.w) + (dm11.y*dm11.w) + (dm12.y*dm12.w) + (dm13.y*dm13.w) + (dm14.y*dm14.w) + (dm15.y*dm15.w);
	  apot.z += (dm0.z*dm0.w) + (dm1.z*dm1.w) + (dm2.z*dm2.w) + (dm3.z*dm3.w) + (dm4.z*dm4.w) + (dm5.z*dm5.w) + (dm6.z*dm6.w) + (dm7.z*dm7.w) \
	    + (dm8.z*dm8.w) + (dm9.z*dm9.w) + (dm10.z*dm10.w) + (dm11.z*dm11.w) + (dm12.z*dm12.w) + (dm13.z*dm13.w) + (dm14.z*dm14.w) + (dm15.z*dm15.w);
	}

      }

      apot.w -= pot0+pot1+pot2+pot3+pot4+pot5+pot6+pot7;

      if(ibase + tid < iend){
	apotlist[ibase + tid] = apot;
      }
    }
  }


  // 2008/04/23
  __device__ float4 dev_apot(float4 xi, float4 xj, float4 apot)
  {
    float dx = xj.x - xi.x;
    float dy = xj.y - xi.y;
    float dz = xj.z - xi.z;
    float eps2 = xi.w;
    float mj   = xj.w;
    float r2   = (dx*dx + eps2) + dy*dy + dz*dz;
    float r1i  = rsqrt(r2);
    float r2i  = r1i*r1i;
    float mr1i = mj * r1i;
    float mr3i = mr1i * r2i;
    apot.x += dx * mr3i;
    apot.y += dy * mr3i;
    apot.z += dz * mr3i;
    apot.w -= mr1i;
    return (apot);
  }

  // 2008/04/23
  __global__ 
  void cunbody_kernel_tree_012(float4 *xilist,
			       float4 *xjlist,
			       float4 *apotlist,
			       unsigned int *ioffset,
			       unsigned int *joffset)
  {
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int ibegin = ioffset[bid];
    unsigned int iend   = ioffset[bid+1];
    unsigned int jbegin = joffset[bid];
    unsigned int jend   = joffset[bid+1];
    for(unsigned int ibase = ibegin; ibase < iend; ibase += NTHRE){
      float4 xi = xilist[ibase + tid];
      float4 apot = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

      for(unsigned int jbase = jbegin; jbase < jend; jbase += NTHRE){
	__shared__ float4 sj[NTHRE];
	sj[tid] = xjlist[jbase + tid];
	__syncthreads();
	apot = dev_apot(xi, sj[0], apot); apot = dev_apot(xi, sj[1], apot); apot = dev_apot(xi, sj[2], apot); apot = dev_apot(xi, sj[3], apot);	apot = dev_apot(xi, sj[4], apot);
	apot = dev_apot(xi, sj[5], apot); apot = dev_apot(xi, sj[6], apot); apot = dev_apot(xi, sj[7], apot); apot = dev_apot(xi, sj[8], apot); apot = dev_apot(xi, sj[9], apot);
	apot = dev_apot(xi, sj[10], apot); apot = dev_apot(xi, sj[11], apot); apot = dev_apot(xi, sj[12], apot); apot = dev_apot(xi, sj[13], apot); apot = dev_apot(xi, sj[14], apot);
	apot = dev_apot(xi, sj[15], apot); apot = dev_apot(xi, sj[16], apot); apot = dev_apot(xi, sj[17], apot); apot = dev_apot(xi, sj[18], apot); apot = dev_apot(xi, sj[19], apot);
	apot = dev_apot(xi, sj[20], apot); apot = dev_apot(xi, sj[21], apot); apot = dev_apot(xi, sj[22], apot); apot = dev_apot(xi, sj[23], apot); apot = dev_apot(xi, sj[24], apot);
	apot = dev_apot(xi, sj[25], apot); apot = dev_apot(xi, sj[26], apot); apot = dev_apot(xi, sj[27], apot); apot = dev_apot(xi, sj[28], apot); apot = dev_apot(xi, sj[29], apot);
	apot = dev_apot(xi, sj[30], apot); apot = dev_apot(xi, sj[31], apot); apot = dev_apot(xi, sj[32], apot); apot = dev_apot(xi, sj[33], apot); apot = dev_apot(xi, sj[34], apot);
	apot = dev_apot(xi, sj[35], apot); apot = dev_apot(xi, sj[36], apot); apot = dev_apot(xi, sj[37], apot); apot = dev_apot(xi, sj[38], apot); apot = dev_apot(xi, sj[39], apot);
	apot = dev_apot(xi, sj[40], apot); apot = dev_apot(xi, sj[41], apot); apot = dev_apot(xi, sj[42], apot); apot = dev_apot(xi, sj[43], apot); apot = dev_apot(xi, sj[44], apot);
	apot = dev_apot(xi, sj[45], apot); apot = dev_apot(xi, sj[46], apot); apot = dev_apot(xi, sj[47], apot); apot = dev_apot(xi, sj[48], apot); apot = dev_apot(xi, sj[49], apot);
	apot = dev_apot(xi, sj[50], apot); apot = dev_apot(xi, sj[51], apot); apot = dev_apot(xi, sj[52], apot); apot = dev_apot(xi, sj[53], apot); apot = dev_apot(xi, sj[54], apot);
	apot = dev_apot(xi, sj[55], apot); apot = dev_apot(xi, sj[56], apot); apot = dev_apot(xi, sj[57], apot); apot = dev_apot(xi, sj[58], apot); apot = dev_apot(xi, sj[59], apot);
	apot = dev_apot(xi, sj[60], apot); apot = dev_apot(xi, sj[61], apot); apot = dev_apot(xi, sj[62], apot); apot = dev_apot(xi, sj[63], apot); apot = dev_apot(xi, sj[64], apot);
	apot = dev_apot(xi, sj[65], apot); apot = dev_apot(xi, sj[66], apot); apot = dev_apot(xi, sj[67], apot); apot = dev_apot(xi, sj[68], apot); apot = dev_apot(xi, sj[69], apot);
	apot = dev_apot(xi, sj[70], apot); apot = dev_apot(xi, sj[71], apot); apot = dev_apot(xi, sj[72], apot); apot = dev_apot(xi, sj[73], apot); apot = dev_apot(xi, sj[74], apot);
	apot = dev_apot(xi, sj[75], apot); apot = dev_apot(xi, sj[76], apot); apot = dev_apot(xi, sj[77], apot); apot = dev_apot(xi, sj[78], apot); apot = dev_apot(xi, sj[79], apot);
	apot = dev_apot(xi, sj[80], apot); apot = dev_apot(xi, sj[81], apot); apot = dev_apot(xi, sj[82], apot); apot = dev_apot(xi, sj[83], apot); apot = dev_apot(xi, sj[84], apot);
	apot = dev_apot(xi, sj[85], apot); apot = dev_apot(xi, sj[86], apot); apot = dev_apot(xi, sj[87], apot); apot = dev_apot(xi, sj[88], apot); apot = dev_apot(xi, sj[89], apot);
	apot = dev_apot(xi, sj[90], apot); apot = dev_apot(xi, sj[91], apot); apot = dev_apot(xi, sj[92], apot); apot = dev_apot(xi, sj[93], apot); apot = dev_apot(xi, sj[94], apot);
	apot = dev_apot(xi, sj[95], apot); apot = dev_apot(xi, sj[96], apot); apot = dev_apot(xi, sj[97], apot); apot = dev_apot(xi, sj[98], apot); apot = dev_apot(xi, sj[99], apot);
	apot = dev_apot(xi, sj[100], apot); apot = dev_apot(xi, sj[101], apot); apot = dev_apot(xi, sj[102], apot); apot = dev_apot(xi, sj[103], apot); apot = dev_apot(xi, sj[104], apot);
	apot = dev_apot(xi, sj[105], apot); apot = dev_apot(xi, sj[106], apot); apot = dev_apot(xi, sj[107], apot); apot = dev_apot(xi, sj[108], apot); apot = dev_apot(xi, sj[109], apot);
	apot = dev_apot(xi, sj[110], apot); apot = dev_apot(xi, sj[111], apot); apot = dev_apot(xi, sj[112], apot); apot = dev_apot(xi, sj[113], apot); apot = dev_apot(xi, sj[114], apot);
	apot = dev_apot(xi, sj[115], apot); apot = dev_apot(xi, sj[116], apot); apot = dev_apot(xi, sj[117], apot); apot = dev_apot(xi, sj[118], apot); apot = dev_apot(xi, sj[119], apot);
	apot = dev_apot(xi, sj[120], apot); apot = dev_apot(xi, sj[121], apot); apot = dev_apot(xi, sj[122], apot); apot = dev_apot(xi, sj[123], apot); apot = dev_apot(xi, sj[124], apot);
	apot = dev_apot(xi, sj[125], apot); apot = dev_apot(xi, sj[126], apot); apot = dev_apot(xi, sj[127], apot);
	__syncthreads();
      }

      if(ibase + tid < iend){
	apotlist[ibase + tid] = apot;
      }
    }
  }

  // 2008/04/24
  __global__ 
  void cunbody_kernel_tree_013(float4 *xilist,
			       float4 *xjlist,
			       float4 *apotlist,
			       unsigned int *ioffset,
			       unsigned int *joffset)
  {
    __syncthreads(); // iranai

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int ibegin = ioffset[bid];
    int iend   = ioffset[bid+1];
    int jbegin = joffset[bid];
    int jend   = joffset[bid+1];

    __syncthreads(); // iranai
    for(int ibase = ibegin; ibase < iend; ibase += NTHRE){

      __syncthreads(); // iranai
      float4 xi = xilist[ibase + tid];
      float4 apot = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      __syncthreads(); // iranai
      __shared__ float4 sj[NTHRE];
      __syncthreads(); // iranai

      for(int jbase = jbegin; jbase < jend; jbase += NTHRE){
	sj[tid] = xjlist[jbase + tid];
	__syncthreads();
	for(int j=0; j<128; j++){ apot = dev_apot(xi, sj[j], apot); }
	__syncthreads();
      }

      __syncthreads(); // iranai
      if(ibase + tid < iend){
	apotlist[ibase + tid] = apot;
      }
      __syncthreads(); // iranai
    }

    __syncthreads(); // irania
  }


}; // namespace libcunbody __END__


