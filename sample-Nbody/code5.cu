#include "cutil.h"

__device__ float4 
inter(float4 xj, float4 xi)
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

  float4 output;
  output.x = dx;
  output.y = dy;
  output.z = dz;
  output.w = mr3i;
  return (output);
  /*
  apot.x += dx * mr3i;       // Accel AXi
  apot.y += dy * mr3i;       // Accel AYi
  apot.z += dz * mr3i;       // Accel AZi
  return (apot);
  */
}
#define NTHRE (128) // blockDim.x¤ÈÆ±ÃÍ
__global__ void
kernel(float4* g_xj,
       float* g_xi,
       float* g_fi,
       int ni,
       int nj)
{
  int tid = threadIdx.x;
  int i = blockIdx.x*NTHRE+tid;
  float4 ai = make_float4(0.0, 0.0, 0.0, 0.0);
  float4 xi;
  xi.x = g_xi[i];
  xi.y = g_xi[i+ni];
  xi.z = g_xi[i+ni*2];
  xi.w = g_xi[i+ni*3];
  __shared__ float4 s_xj[NTHRE];

  float4 mf000;float4 mf001;float4 mf002;float4 mf003;float4 mf004;float4 mf005;float4 mf006;float4 mf007;float4 mf008;float4 mf009;
  float4 mf010;float4 mf011;float4 mf012;float4 mf013;float4 mf014;float4 mf015;float4 mf016;float4 mf017;float4 mf018;float4 mf019;
  float4 mf020;float4 mf021;float4 mf022;float4 mf023;float4 mf024;float4 mf025;float4 mf026;float4 mf027;float4 mf028;float4 mf029;
  float4 mf030;float4 mf031;float4 mf032;float4 mf033;float4 mf034;float4 mf035;float4 mf036;float4 mf037;float4 mf038;float4 mf039;
  float4 mf040;float4 mf041;float4 mf042;float4 mf043;float4 mf044;float4 mf045;float4 mf046;float4 mf047;float4 mf048;float4 mf049;
  float4 mf050;float4 mf051;float4 mf052;float4 mf053;float4 mf054;float4 mf055;float4 mf056;float4 mf057;float4 mf058;float4 mf059;
  float4 mf060;float4 mf061;float4 mf062;float4 mf063;float4 mf064;float4 mf065;float4 mf066;float4 mf067;float4 mf068;float4 mf069;
  float4 mf070;float4 mf071;float4 mf072;float4 mf073;float4 mf074;float4 mf075;float4 mf076;float4 mf077;float4 mf078;float4 mf079;
  float4 mf080;float4 mf081;float4 mf082;float4 mf083;float4 mf084;float4 mf085;float4 mf086;float4 mf087;float4 mf088;float4 mf089;
  float4 mf090;float4 mf091;float4 mf092;float4 mf093;float4 mf094;float4 mf095;float4 mf096;float4 mf097;float4 mf098;float4 mf099;
  float4 mf100;float4 mf101;float4 mf102;float4 mf103;float4 mf104;float4 mf105;float4 mf106;float4 mf107;float4 mf108;float4 mf109;
  float4 mf110;float4 mf111;float4 mf112;float4 mf113;float4 mf114;float4 mf115;float4 mf116;float4 mf117;float4 mf118;float4 mf119;
  float4 mf120;float4 mf121;float4 mf122;float4 mf123;float4 mf124;float4 mf125;float4 mf126;float4 mf127;

  for(int j = 0; j<nj; j+=NTHRE){
    __syncthreads();
    s_xj[tid] = g_xj[j+tid];
    __syncthreads();
    mf000=inter(s_xj[0], xi);
    mf001=inter(s_xj[1], xi);mf002=inter(s_xj[2], xi);mf003=inter(s_xj[3], xi);mf004=inter(s_xj[4], xi);
    mf005=inter(s_xj[5], xi);mf006=inter(s_xj[6], xi);mf007=inter(s_xj[7], xi);mf008=inter(s_xj[8], xi);
    mf009=inter(s_xj[9], xi);mf010=inter(s_xj[10], xi);mf011=inter(s_xj[11], xi);mf012=inter(s_xj[12], xi);
    mf013=inter(s_xj[13], xi);mf014=inter(s_xj[14], xi);mf015=inter(s_xj[15], xi);mf016=inter(s_xj[16], xi);
    mf017=inter(s_xj[17], xi);mf018=inter(s_xj[18], xi);mf019=inter(s_xj[19], xi);mf020=inter(s_xj[20], xi);
    mf021=inter(s_xj[21], xi);mf022=inter(s_xj[22], xi);mf023=inter(s_xj[23], xi);mf024=inter(s_xj[24], xi);
    mf025=inter(s_xj[25], xi);mf026=inter(s_xj[26], xi);mf027=inter(s_xj[27], xi);mf028=inter(s_xj[28], xi);
    mf029=inter(s_xj[29], xi);mf030=inter(s_xj[30], xi);mf031=inter(s_xj[31], xi);mf032=inter(s_xj[32], xi);
    mf033=inter(s_xj[33], xi);mf034=inter(s_xj[34], xi);mf035=inter(s_xj[35], xi);mf036=inter(s_xj[36], xi);
    mf037=inter(s_xj[37], xi);mf038=inter(s_xj[38], xi);mf039=inter(s_xj[39], xi);mf040=inter(s_xj[40], xi);
    mf041=inter(s_xj[41], xi);mf042=inter(s_xj[42], xi);mf043=inter(s_xj[43], xi);mf044=inter(s_xj[44], xi);
    mf045=inter(s_xj[45], xi);mf046=inter(s_xj[46], xi);mf047=inter(s_xj[47], xi);mf048=inter(s_xj[48], xi);
    mf049=inter(s_xj[49], xi);mf050=inter(s_xj[50], xi);mf051=inter(s_xj[51], xi);mf052=inter(s_xj[52], xi);
    mf053=inter(s_xj[53], xi);mf054=inter(s_xj[54], xi);mf055=inter(s_xj[55], xi);mf056=inter(s_xj[56], xi);
    mf057=inter(s_xj[57], xi);mf058=inter(s_xj[58], xi);mf059=inter(s_xj[59], xi);mf060=inter(s_xj[60], xi);
    mf061=inter(s_xj[61], xi);mf062=inter(s_xj[62], xi);mf063=inter(s_xj[63], xi);mf064=inter(s_xj[64], xi);
    mf065=inter(s_xj[65], xi);mf066=inter(s_xj[66], xi);mf067=inter(s_xj[67], xi);mf068=inter(s_xj[68], xi);
    mf069=inter(s_xj[69], xi);mf070=inter(s_xj[70], xi);mf071=inter(s_xj[71], xi);mf072=inter(s_xj[72], xi);
    mf073=inter(s_xj[73], xi);mf074=inter(s_xj[74], xi);mf075=inter(s_xj[75], xi);mf076=inter(s_xj[76], xi);
    mf077=inter(s_xj[77], xi);mf078=inter(s_xj[78], xi);mf079=inter(s_xj[79], xi);mf080=inter(s_xj[80], xi);
    mf081=inter(s_xj[81], xi);mf082=inter(s_xj[82], xi);mf083=inter(s_xj[83], xi);mf084=inter(s_xj[84], xi);
    mf085=inter(s_xj[85], xi);mf086=inter(s_xj[86], xi);mf087=inter(s_xj[87], xi);mf088=inter(s_xj[88], xi);
    mf089=inter(s_xj[89], xi);mf090=inter(s_xj[90], xi);mf091=inter(s_xj[91], xi);mf092=inter(s_xj[92], xi);
    mf093=inter(s_xj[93], xi);mf094=inter(s_xj[94], xi);mf095=inter(s_xj[95], xi);mf096=inter(s_xj[96], xi);
    mf097=inter(s_xj[97], xi);mf098=inter(s_xj[98], xi);mf099=inter(s_xj[99], xi);mf100=inter(s_xj[100], xi);
    mf101=inter(s_xj[101], xi);mf102=inter(s_xj[102], xi);mf103=inter(s_xj[103], xi);mf104=inter(s_xj[104], xi);
    mf105=inter(s_xj[105], xi);mf106=inter(s_xj[106], xi);mf107=inter(s_xj[107], xi);mf108=inter(s_xj[108], xi);
    mf109=inter(s_xj[109], xi);mf110=inter(s_xj[110], xi);mf111=inter(s_xj[111], xi);mf112=inter(s_xj[112], xi);
    mf113=inter(s_xj[113], xi);mf114=inter(s_xj[114], xi);mf115=inter(s_xj[115], xi);mf116=inter(s_xj[116], xi);
    mf117=inter(s_xj[117], xi);mf118=inter(s_xj[118], xi);mf119=inter(s_xj[119], xi);mf120=inter(s_xj[120], xi);
    mf121=inter(s_xj[121], xi);mf122=inter(s_xj[122], xi);mf123=inter(s_xj[123], xi);mf124=inter(s_xj[124], xi);
    mf125=inter(s_xj[125], xi);mf126=inter(s_xj[126], xi);mf127=inter(s_xj[127], xi);

    ai.x += (mf000.w * mf000.x)+
      (mf001.w * mf001.x)+(mf002.w * mf002.x)+(mf003.w * mf003.x)+(mf004.w * mf004.x)+(mf005.w * mf005.x)+(mf006.w * mf006.x)+(mf007.w * mf007.x)+(mf008.w * mf008.x)+(mf009.w * mf009.x)+
      (mf010.w * mf010.x)+(mf011.w * mf011.x)+(mf012.w * mf012.x)+(mf013.w * mf013.x)+(mf014.w * mf014.x)+(mf015.w * mf015.x)+(mf016.w * mf016.x)+(mf017.w * mf017.x)+(mf018.w * mf018.x)+
      (mf019.w * mf019.x)+(mf020.w * mf020.x)+(mf021.w * mf021.x)+(mf022.w * mf022.x)+(mf023.w * mf023.x)+(mf024.w * mf024.x)+(mf025.w * mf025.x)+(mf026.w * mf026.x)+(mf027.w * mf027.x)+
      (mf028.w * mf028.x)+(mf029.w * mf029.x)+(mf030.w * mf030.x)+(mf031.w * mf031.x)+(mf032.w * mf032.x)+(mf033.w * mf033.x)+(mf034.w * mf034.x)+(mf035.w * mf035.x)+(mf036.w * mf036.x)+
      (mf037.w * mf037.x)+(mf038.w * mf038.x)+(mf039.w * mf039.x)+(mf040.w * mf040.x)+(mf041.w * mf041.x)+(mf042.w * mf042.x)+(mf043.w * mf043.x)+(mf044.w * mf044.x)+(mf045.w * mf045.x)+
      (mf046.w * mf046.x)+(mf047.w * mf047.x)+(mf048.w * mf048.x)+(mf049.w * mf049.x)+(mf050.w * mf050.x)+(mf051.w * mf051.x)+(mf052.w * mf052.x)+(mf053.w * mf053.x)+(mf054.w * mf054.x)+
      (mf055.w * mf055.x)+(mf056.w * mf056.x)+(mf057.w * mf057.x)+(mf058.w * mf058.x)+(mf059.w * mf059.x)+(mf060.w * mf060.x)+(mf061.w * mf061.x)+(mf062.w * mf062.x)+(mf063.w * mf063.x)+
      (mf064.w * mf064.x)+(mf065.w * mf065.x)+(mf066.w * mf066.x)+(mf067.w * mf067.x)+(mf068.w * mf068.x)+(mf069.w * mf069.x)+(mf070.w * mf070.x)+(mf071.w * mf071.x)+(mf072.w * mf072.x)+
      (mf073.w * mf073.x)+(mf074.w * mf074.x)+(mf075.w * mf075.x)+(mf076.w * mf076.x)+(mf077.w * mf077.x)+(mf078.w * mf078.x)+(mf079.w * mf079.x)+(mf080.w * mf080.x)+(mf081.w * mf081.x)+
      (mf082.w * mf082.x)+(mf083.w * mf083.x)+(mf084.w * mf084.x)+(mf085.w * mf085.x)+(mf086.w * mf086.x)+(mf087.w * mf087.x)+(mf088.w * mf088.x)+(mf089.w * mf089.x)+(mf090.w * mf090.x)+
      (mf091.w * mf091.x)+(mf092.w * mf092.x)+(mf093.w * mf093.x)+(mf094.w * mf094.x)+(mf095.w * mf095.x)+(mf096.w * mf096.x)+(mf097.w * mf097.x)+(mf098.w * mf098.x)+(mf099.w * mf099.x)+
      (mf100.w * mf100.x)+(mf101.w * mf101.x)+(mf102.w * mf102.x)+(mf103.w * mf103.x)+(mf104.w * mf104.x)+(mf105.w * mf105.x)+(mf106.w * mf106.x)+(mf107.w * mf107.x)+(mf108.w * mf108.x)+
      (mf109.w * mf109.x)+(mf110.w * mf110.x)+(mf111.w * mf111.x)+(mf112.w * mf112.x)+(mf113.w * mf113.x)+(mf114.w * mf114.x)+(mf115.w * mf115.x)+(mf116.w * mf116.x)+(mf117.w * mf117.x)+
      (mf118.w * mf118.x)+(mf119.w * mf119.x)+(mf120.w * mf120.x)+(mf121.w * mf121.x)+(mf122.w * mf122.x)+(mf123.w * mf123.x)+(mf124.w * mf124.x)+(mf125.w * mf125.x)+(mf126.w * mf126.x)+
      (mf127.w * mf127.x);

    ai.y += (mf000.w * mf000.y)+
      (mf001.w * mf001.y)+(mf002.w * mf002.y)+(mf003.w * mf003.y)+(mf004.w * mf004.y)+(mf005.w * mf005.y)+(mf006.w * mf006.y)+(mf007.w * mf007.y)+(mf008.w * mf008.y)+(mf009.w * mf009.y)+
      (mf010.w * mf010.y)+(mf011.w * mf011.y)+(mf012.w * mf012.y)+(mf013.w * mf013.y)+(mf014.w * mf014.y)+(mf015.w * mf015.y)+(mf016.w * mf016.y)+(mf017.w * mf017.y)+(mf018.w * mf018.y)+
      (mf019.w * mf019.y)+(mf020.w * mf020.y)+(mf021.w * mf021.y)+(mf022.w * mf022.y)+(mf023.w * mf023.y)+(mf024.w * mf024.y)+(mf025.w * mf025.y)+(mf026.w * mf026.y)+(mf027.w * mf027.y)+
      (mf028.w * mf028.y)+(mf029.w * mf029.y)+(mf030.w * mf030.y)+(mf031.w * mf031.y)+(mf032.w * mf032.y)+(mf033.w * mf033.y)+(mf034.w * mf034.y)+(mf035.w * mf035.y)+(mf036.w * mf036.y)+
      (mf037.w * mf037.y)+(mf038.w * mf038.y)+(mf039.w * mf039.y)+(mf040.w * mf040.y)+(mf041.w * mf041.y)+(mf042.w * mf042.y)+(mf043.w * mf043.y)+(mf044.w * mf044.y)+(mf045.w * mf045.y)+
      (mf046.w * mf046.y)+(mf047.w * mf047.y)+(mf048.w * mf048.y)+(mf049.w * mf049.y)+(mf050.w * mf050.y)+(mf051.w * mf051.y)+(mf052.w * mf052.y)+(mf053.w * mf053.y)+(mf054.w * mf054.y)+
      (mf055.w * mf055.y)+(mf056.w * mf056.y)+(mf057.w * mf057.y)+(mf058.w * mf058.y)+(mf059.w * mf059.y)+(mf060.w * mf060.y)+(mf061.w * mf061.y)+(mf062.w * mf062.y)+(mf063.w * mf063.y)+
      (mf064.w * mf064.y)+(mf065.w * mf065.y)+(mf066.w * mf066.y)+(mf067.w * mf067.y)+(mf068.w * mf068.y)+(mf069.w * mf069.y)+(mf070.w * mf070.y)+(mf071.w * mf071.y)+(mf072.w * mf072.y)+
      (mf073.w * mf073.y)+(mf074.w * mf074.y)+(mf075.w * mf075.y)+(mf076.w * mf076.y)+(mf077.w * mf077.y)+(mf078.w * mf078.y)+(mf079.w * mf079.y)+(mf080.w * mf080.y)+(mf081.w * mf081.y)+
      (mf082.w * mf082.y)+(mf083.w * mf083.y)+(mf084.w * mf084.y)+(mf085.w * mf085.y)+(mf086.w * mf086.y)+(mf087.w * mf087.y)+(mf088.w * mf088.y)+(mf089.w * mf089.y)+(mf090.w * mf090.y)+
      (mf091.w * mf091.y)+(mf092.w * mf092.y)+(mf093.w * mf093.y)+(mf094.w * mf094.y)+(mf095.w * mf095.y)+(mf096.w * mf096.y)+(mf097.w * mf097.y)+(mf098.w * mf098.y)+(mf099.w * mf099.y)+
      (mf100.w * mf100.y)+(mf101.w * mf101.y)+(mf102.w * mf102.y)+(mf103.w * mf103.y)+(mf104.w * mf104.y)+(mf105.w * mf105.y)+(mf106.w * mf106.y)+(mf107.w * mf107.y)+(mf108.w * mf108.y)+
      (mf109.w * mf109.y)+(mf110.w * mf110.y)+(mf111.w * mf111.y)+(mf112.w * mf112.y)+(mf113.w * mf113.y)+(mf114.w * mf114.y)+(mf115.w * mf115.y)+(mf116.w * mf116.y)+(mf117.w * mf117.y)+
      (mf118.w * mf118.y)+(mf119.w * mf119.y)+(mf120.w * mf120.y)+(mf121.w * mf121.y)+(mf122.w * mf122.y)+(mf123.w * mf123.y)+(mf124.w * mf124.y)+(mf125.w * mf125.y)+(mf126.w * mf126.y)+
      (mf127.w * mf127.y);

    ai.z += (mf000.w * mf000.z)+
      (mf001.w * mf001.z)+(mf002.w * mf002.z)+(mf003.w * mf003.z)+(mf004.w * mf004.z)+(mf005.w * mf005.z)+(mf006.w * mf006.z)+(mf007.w * mf007.z)+(mf008.w * mf008.z)+(mf009.w * mf009.z)+
      (mf010.w * mf010.z)+(mf011.w * mf011.z)+(mf012.w * mf012.z)+(mf013.w * mf013.z)+(mf014.w * mf014.z)+(mf015.w * mf015.z)+(mf016.w * mf016.z)+(mf017.w * mf017.z)+(mf018.w * mf018.z)+
      (mf019.w * mf019.z)+(mf020.w * mf020.z)+(mf021.w * mf021.z)+(mf022.w * mf022.z)+(mf023.w * mf023.z)+(mf024.w * mf024.z)+(mf025.w * mf025.z)+(mf026.w * mf026.z)+(mf027.w * mf027.z)+
      (mf028.w * mf028.z)+(mf029.w * mf029.z)+(mf030.w * mf030.z)+(mf031.w * mf031.z)+(mf032.w * mf032.z)+(mf033.w * mf033.z)+(mf034.w * mf034.z)+(mf035.w * mf035.z)+(mf036.w * mf036.z)+
      (mf037.w * mf037.z)+(mf038.w * mf038.z)+(mf039.w * mf039.z)+(mf040.w * mf040.z)+(mf041.w * mf041.z)+(mf042.w * mf042.z)+(mf043.w * mf043.z)+(mf044.w * mf044.z)+(mf045.w * mf045.z)+
      (mf046.w * mf046.z)+(mf047.w * mf047.z)+(mf048.w * mf048.z)+(mf049.w * mf049.z)+(mf050.w * mf050.z)+(mf051.w * mf051.z)+(mf052.w * mf052.z)+(mf053.w * mf053.z)+(mf054.w * mf054.z)+
      (mf055.w * mf055.z)+(mf056.w * mf056.z)+(mf057.w * mf057.z)+(mf058.w * mf058.z)+(mf059.w * mf059.z)+(mf060.w * mf060.z)+(mf061.w * mf061.z)+(mf062.w * mf062.z)+(mf063.w * mf063.z)+
      (mf064.w * mf064.z)+(mf065.w * mf065.z)+(mf066.w * mf066.z)+(mf067.w * mf067.z)+(mf068.w * mf068.z)+(mf069.w * mf069.z)+(mf070.w * mf070.z)+(mf071.w * mf071.z)+(mf072.w * mf072.z)+
      (mf073.w * mf073.z)+(mf074.w * mf074.z)+(mf075.w * mf075.z)+(mf076.w * mf076.z)+(mf077.w * mf077.z)+(mf078.w * mf078.z)+(mf079.w * mf079.z)+(mf080.w * mf080.z)+(mf081.w * mf081.z)+
      (mf082.w * mf082.z)+(mf083.w * mf083.z)+(mf084.w * mf084.z)+(mf085.w * mf085.z)+(mf086.w * mf086.z)+(mf087.w * mf087.z)+(mf088.w * mf088.z)+(mf089.w * mf089.z)+(mf090.w * mf090.z)+
      (mf091.w * mf091.z)+(mf092.w * mf092.z)+(mf093.w * mf093.z)+(mf094.w * mf094.z)+(mf095.w * mf095.z)+(mf096.w * mf096.z)+(mf097.w * mf097.z)+(mf098.w * mf098.z)+(mf099.w * mf099.z)+
      (mf100.w * mf100.z)+(mf101.w * mf101.z)+(mf102.w * mf102.z)+(mf103.w * mf103.z)+(mf104.w * mf104.z)+(mf105.w * mf105.z)+(mf106.w * mf106.z)+(mf107.w * mf107.z)+(mf108.w * mf108.z)+
      (mf109.w * mf109.z)+(mf110.w * mf110.z)+(mf111.w * mf111.z)+(mf112.w * mf112.z)+(mf113.w * mf113.z)+(mf114.w * mf114.z)+(mf115.w * mf115.z)+(mf116.w * mf116.z)+(mf117.w * mf117.z)+
      (mf118.w * mf118.z)+(mf119.w * mf119.z)+(mf120.w * mf120.z)+(mf121.w * mf121.z)+(mf122.w * mf122.z)+(mf123.w * mf123.z)+(mf124.w * mf124.z)+(mf125.w * mf125.z)+(mf126.w * mf126.z)+
      (mf127.w * mf127.z);
  }

  if(i<ni){
    g_fi[i]      = ai.x;
    g_fi[i+ni]   = ai.y;
    g_fi[i+ni*2] = ai.z;
  }
}

#include "api.h"
