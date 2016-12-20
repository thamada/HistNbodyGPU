#include "cutil.h"

__device__ float4
inter(float4 xj,  float4 xi,  float4 apot)
{
  float mj    = xj.w;        // Mass Mj
  float ieps2 = xi.w;        // epsilon^2
  float dx = xj.x - xi.x;    // Coordinates Xj - Xi
  float dy = xj.y - xi.y;    // Coordinates Yj - Yi
  float dz = xj.z - xi.z;    // Coordinates Zj - Zi
  float r2 = dx*dx+dy*dy+dz*dz+ieps2;
  float r1i = 1/sqrt(r2);
  float r2i = r1i * r1i;
  float mr3i = mj * r2i * r1i;
  apot.x += dx * mr3i;       // Accel AXi
  apot.y += dy * mr3i;       // Accel AYi
  apot.z += dz * mr3i;       // Accel AZi
  return (apot);
}

__global__ void
kernel(float4* g_xj,
       float* g_xi,
       float* g_fi,
       int ni,
       int nj)
{
  int i = blockIdx.x*gridDim.x+threadIdx.x;
  float4 ai = make_float4(0.0, 0.0, 0.0, 0.0);
  float4 xi;
  xi.x = g_xi[i];       // Coordinates Xi
  xi.y = g_xi[i+ni];    // Coordinates Yi
  xi.z = g_xi[i+ni*2];  // Coordinates Zi
  xi.w = g_xi[i+ni*3];  // epsilon ^ 2
  for(int j = 0; j<nj; j++)  ai = inter(g_xj[j], xi, ai);
  if(i<ni){
    g_fi[i]      = ai.x;
    g_fi[i+ni]   = ai.y;
    g_fi[i+ni*2] = ai.z;
  }
}

#define NTHRE (128)
#include "api.h"
