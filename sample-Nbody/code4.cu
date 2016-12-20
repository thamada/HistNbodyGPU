#include "cutil.h"

__device__ float4 
inter(float4 xj, float4 xi, float4 apot)
{
  float mj    = xj.w;        // Mass Mj
  float ieps2 = xi.w;        // epsilon^2
  float dx = xj.x - xi.x;    // Coordinates Xj - Xi
  float dy = xj.y - xi.y;    // Coordinates Yj - Yi
  float dz = xj.z - xi.z;    // Coordinates Zj - Zi
  float r2 = (dx*dx+ieps2)+dy*dy+dz*dz;
  float r1i = 1/sqrt(r2);
  float r2i = r1i * r1i;
  float mr3i = mj * r2i * r1i;
  apot.x += dx * mr3i;       // Accel AXi
  apot.y += dy * mr3i;       // Accel AYi
  apot.z += dz * mr3i;       // Accel AZi
  return (apot);
}
#define NTHRE (128) // blockDim.x§»∆±√Õ
__global__ void
kernel(float4* g_xj,
       float* g_xi,
       float* g_fi,
       int ni,
       int nj)
{
  int tid      = threadIdx.x;
  int i = blockIdx.x*NTHRE+tid;
  float4 ai = make_float4(0.0, 0.0, 0.0, 0.0);
  float4 xi;
  xi.x = g_xi[i];
  xi.y = g_xi[i+ni];
  xi.z = g_xi[i+ni*2];
  xi.w = g_xi[i+ni*3];
  __shared__ float4 s_xj[NTHRE];
  for(int j = 0; j<nj; j+=NTHRE){
    __syncthreads();
    s_xj[tid] = g_xj[j+tid];
    __syncthreads();
    ai = inter(s_xj[0], xi, ai); ai = inter(s_xj[1], xi, ai); ai = inter(s_xj[2], xi, ai); ai = inter(s_xj[3], xi, ai);
    ai = inter(s_xj[4], xi, ai); ai = inter(s_xj[5], xi, ai); ai = inter(s_xj[6], xi, ai); ai = inter(s_xj[7], xi, ai);
    ai = inter(s_xj[8], xi, ai); ai = inter(s_xj[9], xi, ai);

    ai = inter(s_xj[10], xi, ai); ai = inter(s_xj[11], xi, ai); ai = inter(s_xj[12], xi, ai); ai = inter(s_xj[13], xi, ai);
    ai = inter(s_xj[14], xi, ai); ai = inter(s_xj[15], xi, ai); ai = inter(s_xj[16], xi, ai); ai = inter(s_xj[17], xi, ai);
    ai = inter(s_xj[18], xi, ai); ai = inter(s_xj[19], xi, ai);

    ai = inter(s_xj[20], xi, ai); ai = inter(s_xj[21], xi, ai); ai = inter(s_xj[22], xi, ai); ai = inter(s_xj[23], xi, ai);
    ai = inter(s_xj[24], xi, ai); ai = inter(s_xj[25], xi, ai); ai = inter(s_xj[26], xi, ai); ai = inter(s_xj[27], xi, ai);
    ai = inter(s_xj[28], xi, ai); ai = inter(s_xj[29], xi, ai);

    ai = inter(s_xj[30], xi, ai); ai = inter(s_xj[31], xi, ai); ai = inter(s_xj[32], xi, ai); ai = inter(s_xj[33], xi, ai);
    ai = inter(s_xj[34], xi, ai); ai = inter(s_xj[35], xi, ai); ai = inter(s_xj[36], xi, ai); ai = inter(s_xj[37], xi, ai);
    ai = inter(s_xj[38], xi, ai); ai = inter(s_xj[39], xi, ai);

    ai = inter(s_xj[40], xi, ai); ai = inter(s_xj[41], xi, ai); ai = inter(s_xj[42], xi, ai); ai = inter(s_xj[43], xi, ai);
    ai = inter(s_xj[44], xi, ai); ai = inter(s_xj[45], xi, ai); ai = inter(s_xj[46], xi, ai); ai = inter(s_xj[47], xi, ai);
    ai = inter(s_xj[48], xi, ai); ai = inter(s_xj[49], xi, ai);

    ai = inter(s_xj[50], xi, ai); ai = inter(s_xj[51], xi, ai); ai = inter(s_xj[52], xi, ai); ai = inter(s_xj[53], xi, ai);
    ai = inter(s_xj[54], xi, ai); ai = inter(s_xj[55], xi, ai); ai = inter(s_xj[56], xi, ai); ai = inter(s_xj[57], xi, ai);
    ai = inter(s_xj[58], xi, ai); ai = inter(s_xj[59], xi, ai);

    ai = inter(s_xj[60], xi, ai); ai = inter(s_xj[61], xi, ai); ai = inter(s_xj[62], xi, ai); ai = inter(s_xj[63], xi, ai);
    ai = inter(s_xj[64], xi, ai); ai = inter(s_xj[65], xi, ai); ai = inter(s_xj[66], xi, ai); ai = inter(s_xj[67], xi, ai);
    ai = inter(s_xj[68], xi, ai); ai = inter(s_xj[69], xi, ai);

    ai = inter(s_xj[70], xi, ai); ai = inter(s_xj[71], xi, ai); ai = inter(s_xj[72], xi, ai); ai = inter(s_xj[73], xi, ai);
    ai = inter(s_xj[74], xi, ai); ai = inter(s_xj[75], xi, ai); ai = inter(s_xj[76], xi, ai); ai = inter(s_xj[77], xi, ai);
    ai = inter(s_xj[78], xi, ai); ai = inter(s_xj[79], xi, ai);

    ai = inter(s_xj[80], xi, ai); ai = inter(s_xj[81], xi, ai); ai = inter(s_xj[82], xi, ai); ai = inter(s_xj[83], xi, ai);
    ai = inter(s_xj[84], xi, ai); ai = inter(s_xj[85], xi, ai); ai = inter(s_xj[86], xi, ai); ai = inter(s_xj[87], xi, ai);
    ai = inter(s_xj[88], xi, ai); ai = inter(s_xj[89], xi, ai);

    ai = inter(s_xj[90], xi, ai); ai = inter(s_xj[91], xi, ai); ai = inter(s_xj[92], xi, ai); ai = inter(s_xj[93], xi, ai);
    ai = inter(s_xj[94], xi, ai); ai = inter(s_xj[95], xi, ai); ai = inter(s_xj[96], xi, ai); ai = inter(s_xj[97], xi, ai);
    ai = inter(s_xj[98], xi, ai); ai = inter(s_xj[99], xi, ai);

    ai = inter(s_xj[100], xi, ai); ai = inter(s_xj[101], xi, ai); ai = inter(s_xj[102], xi, ai); ai = inter(s_xj[103], xi, ai);
    ai = inter(s_xj[104], xi, ai); ai = inter(s_xj[105], xi, ai); ai = inter(s_xj[106], xi, ai); ai = inter(s_xj[107], xi, ai);
    ai = inter(s_xj[108], xi, ai); ai = inter(s_xj[109], xi, ai);

    ai = inter(s_xj[110], xi, ai); ai = inter(s_xj[111], xi, ai); ai = inter(s_xj[112], xi, ai); ai = inter(s_xj[113], xi, ai);
    ai = inter(s_xj[114], xi, ai); ai = inter(s_xj[115], xi, ai); ai = inter(s_xj[116], xi, ai); ai = inter(s_xj[117], xi, ai);
    ai = inter(s_xj[118], xi, ai); ai = inter(s_xj[119], xi, ai);

    ai = inter(s_xj[120], xi, ai); ai = inter(s_xj[121], xi, ai); ai = inter(s_xj[122], xi, ai); ai = inter(s_xj[123], xi, ai);
    ai = inter(s_xj[124], xi, ai); ai = inter(s_xj[125], xi, ai); ai = inter(s_xj[126], xi, ai); ai = inter(s_xj[127], xi, ai);
  }
  if(i<ni){
    g_fi[i]      = ai.x;
    g_fi[i+ni]   = ai.y;
    g_fi[i+ni*2] = ai.z;
  }
}

#include "api.h"
