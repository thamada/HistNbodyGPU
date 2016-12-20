#define ERROR_CHECK

#include <iostream>
using namespace std;
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define NMAX (1<<17) // 131072

extern "C" void force(double xj[][3], double mj[], double xi[][3], double eps2, double a[][3], int ni, int nj);

#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>
double get_time(void)
{
  struct rusage r;
  double sec, usec, z;
  getrusage(RUSAGE_SELF, &r);
  sec   = r.ru_utime.tv_sec;
  usec  = r.ru_utime.tv_usec;
  z = sec + usec*1.0e-6;
  return (z);
}

static double r[NMAX][3];
static double m[NMAX];
static double eps2 = 0.0001;
static double a[NMAX][3];
static double v[NMAX][3];
static double a_h[NMAX][3];

void readfile(char* fname, int* nbody, double* m, double r[][3], double v[][3])
{
  ifstream f;
  f.open(fname, ios::in);
  if(!f) {
    cerr << "Can't open file: " <<  fname << endl;
    exit(-1);
  }
  char* buf= new char [2048];
  f.getline(buf, 511); // *ignore*
  int n;
  f >> n;
  (*nbody) = n;
  f.getline(buf, 511); // *ignore*

  f.scientific;
  for(int i=0; i<n; i++){
    f.getline(buf, 1024);
    char ii[256];
    double mj, x, y, z, vx, vy, vz;
    f >> ii >> mj >> x >> y >> z >> vx >> vy >> vz;
    //    printf("%s   %.16E   % .16E % .16E % .16E   % .16E % .16E % .16E \n", ii, m, x, y, z, vx, vy, vz);
    m[i]    = mj;
    r[i][0] = x;
    r[i][1] = y;
    r[i][2] = z;
    v[i][0] = vx;
    v[i][1] = vy;
    v[i][2] = vz;
  }
  f.close();

  delete[] buf;
}

void force_host(double xj[][3],
								double m[],
								double xi[][3],
								double eps2,
								double a[][3],
								int ni,
								int nj)
{
  int i,j,d;
  double dx[3];
  for(i=0;i<ni;i++) for(d=0;d<3;d++) a[i][d] = 0.0;

  for(i=0;i<ni;i++){
    for(j=0;j<nj;j++){
      double r2,r3;
      r2 = eps2;
      for(d=0;d<3;d++){
				dx[d] = xj[j][d] - xi[i][d];
				r2 += dx[d] * dx[d];
      }
      r3 = sqrt(r2)*r2;
      for(d=0;d<3;d++){
				a[i][d] +=  m[j]*dx[d]/r3;
      }
    }
  }
}

int main(int argc,char** argv)
{
  int n=0;
  int nstep = 1;

  if((argc == 2)||(argc == 3)){
    char ifile[256];
    strcpy(ifile,argv[1]);
    readfile(ifile, &n, m, r, v);
    if(argc == 3)  nstep = atoi(argv[2]);
  }else{
    fprintf(stderr, "argc = %d\n",argc);
    fprintf(stderr, "cmd <init file> <nstep>\n");
    exit(-1);
  }

  int ni=n;
  int nj=n;

  force(r, m, r, eps2, a, ni, nj);

  double eps2 = (1./256.);
  double tt = get_time();
  for(int step = 0; step < nstep; step++) force(r, m, r, eps2, a, ni, nj);
  tt = get_time() - tt;

  double nfloat = 38.0 * ((double)ni)*((double)nj) * ((double)nstep);
  double gflops = nfloat * (1.e-9) / tt;
  //  fprintf(stdout, "n=%d\t%g sec/gpucall\t%g Gflops\n", n, tt/(double)nstep, gflops );
  fprintf(stdout, "n=%d\n", n);
  fprintf(stdout, "%g sec/gpucall\n", tt/(double)nstep);
  fprintf(stdout, "%g Gflops\n",  gflops );


  if(0){
    cout << "Start Error check: " << n <<endl;
    force_host(r, m, r, eps2, a_h, ni, nj);
    double err_max = 0.0;
    for(int i=0;i<ni;i++){
      double err=0.0;
      for(int d=0;d<3;d++){
				double diff = fabs(a[i][d] - a_h[i][d]);
				err += (diff*diff);
      }
      err = sqrt(err);
      err = err/sqrt(a_h[i][0]*a_h[i][0]+a_h[i][1]*a_h[i][1]+a_h[i][2]*a_h[i][2]);
      if (err > err_max){
				err_max = err;
				printf("[%d]:\t", i);
				printf("%.7e:\t", err);
				printf("%e,%e,%e:\t%e,%e,%e\n",a_h[i][0], a_h[i][1], a_h[i][2],a[i][0], a[i][1], a[i][2]);
      }
    }
    printf("Max Err %.7e\n", err_max);
  }

  return (0);
}
