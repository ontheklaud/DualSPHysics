//HEAD_DSPH
/*
 <DUALSPHYSICS>  Copyright (c) 2025 by Dr Jose M. Dominguez et al. (see http://dual.sphysics.org/index.php/developers/). 

 EPHYSLAB Environmental Physics Laboratory, Universidade de Vigo, Ourense, Spain.
 School of Mechanical, Aerospace and Civil Engineering, University of Manchester, Manchester, U.K.

 This file is part of DualSPHysics. 

 DualSPHysics is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License 
 as published by the Free Software Foundation; either version 2.1 of the License, or (at your option) any later version.
 
 DualSPHysics is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details. 

 You should have received a copy of the GNU Lesser General Public License along with DualSPHysics. If not, see <http://www.gnu.org/licenses/>. 
*/

/// \file JSphGpu_ker.cu \brief Implements functions and CUDA kernels for the Particle Interaction and System Update.


#include <hip/hip_runtime.h>
#include "JSphGpu_ker.h"
#include "Functions.h"
#include "FunctionsCuda.h"
#include "JLog2.h"
#include <cfloat>
#include <hip/hip_math_constants.h>
//:#include "JDgKerPrint.h"
//:#include "JDgKerPrint_ker.h"

#pragma warning(disable : 4267) //Cancels "warning C4267: conversion from 'size_t' to 'int', possible loss of data"
#pragma warning(disable : 4244) //Cancels "warning C4244: conversion from 'unsigned __int64' to 'unsigned int', possible loss of data"
#pragma warning(disable : 4503) //Cancels "warning C4503: decorated name length exceeded, name was truncated"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/logical.h>
#include <thrust/count.h>

__constant__ StCteInteraction CTE;
#define CTE_AVAILABLE

namespace cusph{
#include "FunctionsBasic_iker.h"
#include "FunctionsMath_iker.h"
#include "FunctionsGeo3d_iker.h"
#include "FunSphKernel_iker.h"
#include "FunSphEos_iker.h"
#include "JCellSearch_iker.h"
#include "JSphGpu_InOut_iker.h"


//==============================================================================
/// Reduction using maximum of float values in shared memory for a warp.
/// Reduccion mediante maximo de valores float en memoria shared para un warp.
//==============================================================================
template <unsigned blockSize> __device__ void KerReduMaxFloatWarp(
  volatile float* sdat,unsigned tid)
{
  if(blockSize>=64)sdat[tid]=max(sdat[tid],sdat[tid+32]);
  if(blockSize>=32)sdat[tid]=max(sdat[tid],sdat[tid+16]);
  if(blockSize>=16)sdat[tid]=max(sdat[tid],sdat[tid+8]);
  if(blockSize>=8)sdat[tid]=max(sdat[tid],sdat[tid+4]);
  if(blockSize>=4)sdat[tid]=max(sdat[tid],sdat[tid+2]);
  if(blockSize>=2)sdat[tid]=max(sdat[tid],sdat[tid+1]);
}

//==============================================================================
/// Accumulates the maximum of n values of array dat[], storing the result in 
/// the beginning of res[].(Many positions of res[] are used as blocks, 
/// storing the final result in res[0]).
///
/// Acumula el maximo de n valores del vector dat[], guardando el resultado al 
/// principio de res[] (Se usan tantas posiciones del res[] como bloques, 
/// quedando el resultado final en res[0]).
//==============================================================================
template <unsigned blockSize> __global__ void KerReduMaxFloat(unsigned n
  ,unsigned ini,const float* dat,float* res)
{
  extern __shared__ float sdat[];
  unsigned tid=threadIdx.x;
  unsigned c=blockIdx.x*blockDim.x + threadIdx.x;
  sdat[tid]=(c<n? dat[c+ini]: -FLT_MAX);
  __syncthreads();
  if(blockSize>=512){ if(tid<256)sdat[tid]=max(sdat[tid],sdat[tid+256]);  __syncthreads(); }
  if(blockSize>=256){ if(tid<128)sdat[tid]=max(sdat[tid],sdat[tid+128]);  __syncthreads(); }
  if(blockSize>=128){ if(tid<64) sdat[tid]=max(sdat[tid],sdat[tid+64]);   __syncthreads(); }
  if(tid<32)KerReduMaxFloatWarp<blockSize>(sdat,tid);
  if(tid==0)res[blockIdx.x]=sdat[0];
}

//==============================================================================
/// Returns the maximum of an array, using resu[] as auxiliar array.
/// Size of resu[] must be >= a (N/SPHBSIZE+1)+(N/(SPHBSIZE*SPHBSIZE)+SPHBSIZE)
///
/// Devuelve el maximo de un vector, usando resu[] como vector auxiliar. El tamanho
/// de resu[] debe ser >= a (N/SPHBSIZE+1)+(N/(SPHBSIZE*SPHBSIZE)+SPHBSIZE)
//==============================================================================
float ReduMaxFloat(unsigned ndata,unsigned inidata,float* data,float* resu){
  float resf=0;
  if(ndata>=1){
    unsigned n=ndata,ini=inidata;
    unsigned smemSize=SPHBSIZE*sizeof(float);
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    unsigned n_blocks=sgrid.x*sgrid.y;
    float* dat=data;
    float* resu1=resu;
    float* resu2=resu+n_blocks;
    float* res=resu1;
    while(n>1){
      KerReduMaxFloat<SPHBSIZE><<<sgrid,SPHBSIZE,smemSize>>>(n,ini,dat,res);
      n=n_blocks; ini=0;
      sgrid=GetSimpleGridSize(n,SPHBSIZE);  
      n_blocks=sgrid.x*sgrid.y;
      if(n>1){
        dat=res; res=(dat==resu1? resu2: resu1); 
      }
    }
    if(ndata>1)hipMemcpy(&resf,res,sizeof(float),hipMemcpyDeviceToHost);
    else hipMemcpy(&resf,data,sizeof(float),hipMemcpyDeviceToHost);
  }
  //else{//-Using Thrust library is slower than ReduMasFloat() with ndata < 5M.
  //  thrust::device_ptr<float> dev_ptr(data);
  //  resf=thrust::reduce(dev_ptr,dev_ptr+ndata,-FLT_MAX,thrust::maximum<float>());
  //}
  return(resf);
}

//==============================================================================
/// Accumulates the sum of n values of array dat[], storing the result in 
/// the beginning of res[].(Many positions of res[] are used as blocks, 
/// storing the final result in res[0]).
///
/// Acumula la suma de n valores del vector dat[].w, guardando el resultado al 
/// principio de res[] (Se usan tantas posiciones del res[] como bloques, 
/// quedando el resultado final en res[0]).
//==============================================================================
template <unsigned blockSize> __global__ void KerReduMaxFloat_w(unsigned n
  ,unsigned ini,const float4* dat,float* res)
{
  extern __shared__ float sdat[];
  unsigned tid=threadIdx.x;
  unsigned c=blockIdx.x*blockDim.x + threadIdx.x;
  sdat[tid]=(c<n? dat[c+ini].w: -FLT_MAX);
  __syncthreads();
  if(blockSize>=512){ if(tid<256)sdat[tid]=max(sdat[tid],sdat[tid+256]);  __syncthreads(); }
  if(blockSize>=256){ if(tid<128)sdat[tid]=max(sdat[tid],sdat[tid+128]);  __syncthreads(); }
  if(blockSize>=128){ if(tid<64) sdat[tid]=max(sdat[tid],sdat[tid+64]);   __syncthreads(); }
  if(tid<32)KerReduMaxFloatWarp<blockSize>(sdat,tid);
  if(tid==0)res[blockIdx.x]=sdat[0];
}

//==============================================================================
/// Returns the maximum of an array, using resu[] as auxiliar array.
/// Size of resu[] must be >= a (N/SPHBSIZE+1)+(N/(SPHBSIZE*SPHBSIZE)+SPHBSIZE).
///
/// Devuelve el maximo de la componente w de un vector float4, usando resu[] como 
/// vector auxiliar. El tamanho de resu[] debe ser >= a (N/SPHBSIZE+1)+(N/(SPHBSIZE*SPHBSIZE)+SPHBSIZE).
//==============================================================================
float ReduMaxFloat_w(unsigned ndata,unsigned inidata,float4* data,float* resu){
  unsigned n=ndata,ini=inidata;
  unsigned smemSize=SPHBSIZE*sizeof(float);
  dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
  unsigned n_blocks=sgrid.x*sgrid.y;
  float* dat=NULL;
  float* resu1=resu;
  float* resu2=resu+n_blocks;
  float* res=resu1;
  while(n>1){
    if(!dat)KerReduMaxFloat_w<SPHBSIZE><<<sgrid,SPHBSIZE,smemSize>>>(n,ini,data,res);
    else KerReduMaxFloat<SPHBSIZE><<<sgrid,SPHBSIZE,smemSize>>>(n,ini,dat,res);
    n=n_blocks; ini=0;
    sgrid=GetSimpleGridSize(n,SPHBSIZE);  
    n_blocks=sgrid.x*sgrid.y;
    if(n>1){
      dat=res; res=(dat==resu1? resu2: resu1); 
    }
  }
  float resf;
  if(ndata>1)hipMemcpy(&resf,res,sizeof(float),hipMemcpyDeviceToHost);
  else{
    float4 resf4;
    hipMemcpy(&resf4,data,sizeof(float4),hipMemcpyDeviceToHost);
    resf=resf4.w;
  }
  return(resf);
}

//==============================================================================
/// Stores constants for the GPU interaction.
/// Graba constantes para la interaccion a la GPU.
//==============================================================================
void CteInteractionUp(const StCteInteraction* cte){
  hipMemcpyToSymbol(HIP_SYMBOL(CTE),cte,sizeof(StCteInteraction));
}

//------------------------------------------------------------------------------
/// Initialises array with the indicated value.
/// Inicializa array con el valor indicado.
//------------------------------------------------------------------------------
__global__ void KerInitArray(unsigned n,float3* v,float3 value)
{
  unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n)v[p]=value;
}

//==============================================================================
/// Initialises array with the indicated value.
/// Inicializa array con el valor indicado.
//==============================================================================
void InitArray(unsigned n,float3* v,tfloat3 value){
  if(n){
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    KerInitArray <<<sgrid,SPHBSIZE>>> (n,v,Float3(value));
  }
}

//------------------------------------------------------------------------------
/// Sets v[].y to zero.
/// Pone v[].y a cero.
//------------------------------------------------------------------------------
__global__ void KerResety(unsigned n,unsigned ini,float3* v)
{
  unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n)v[p+ini].y=0;
}

//==============================================================================
/// Sets v[].y to zero.
/// Pone v[].y a cero.
//==============================================================================
void Resety(unsigned n,unsigned ini,float3* v){
  if(n){
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    KerResety <<<sgrid,SPHBSIZE>>> (n,ini,v);
  }
}

//------------------------------------------------------------------------------
/// Calculates module^2 of ace.
//------------------------------------------------------------------------------
__global__ void KerComputeAceMod(unsigned n,const float3* ace,float* acemod)
{
  unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    const float3 r=ace[p];
    acemod[p]=r.x*r.x+r.y*r.y+r.z*r.z;
  }
}

//==============================================================================
/// Calculates module^2 of ace.
//==============================================================================
void ComputeAceMod(unsigned n,const float3* ace,float* acemod){
  if(n){
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    KerComputeAceMod <<<sgrid,SPHBSIZE>>> (n,ace,acemod);
  }
}

//------------------------------------------------------------------------------
/// Calculates module^2 of ace, comprobando que la particula sea normal.
/// Uses zero for periodic particles.
//------------------------------------------------------------------------------
__global__ void KerComputeAceMod(unsigned n,const typecode* code
  ,const float3* ace,float* acemod)
{
  unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    const typecode rcod=code[p];
    const float3 r=(CODE_IsNormal(rcod) && !CODE_IsFluidInout(rcod) && !CODE_IsFluidBuffer(rcod)? ace[p]: make_float3(0,0,0));  //<vs_vrres
    acemod[p]=r.x*r.x+r.y*r.y+r.z*r.z;
  }
}

//==============================================================================
/// Calculates module^2 of ace, comprobando que la particula sea normal.
/// Uses zero for periodic particles.
//==============================================================================
void ComputeAceMod(unsigned n,const typecode* code,const float3* ace
  ,float* acemod)
{
  if(n){
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    KerComputeAceMod <<<sgrid,SPHBSIZE>>> (n,code,ace,acemod);
  }
}


//##############################################################################
//# Other kernels...
//# Otros kernels...
//##############################################################################
//------------------------------------------------------------------------------
/// Calculates module^2 of vel.
//------------------------------------------------------------------------------
__global__ void KerComputeVelMod(unsigned n,const float4* vel,float* velmod)
{
  unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    const float4 r=vel[p];
    velmod[p]=r.x*r.x+r.y*r.y+r.z*r.z;
  }
}

//==============================================================================
/// Calculates module^2 of vel.
//==============================================================================
void ComputeVelMod(unsigned n,const float4* vel,float* velmod){
  if(n){
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    KerComputeVelMod <<<sgrid,SPHBSIZE>>> (n,vel,velmod);
  }
}


//##############################################################################
//# Kernels para cambiar la posicion.
//# Kernels for changing the position.
//##############################################################################
//------------------------------------------------------------------------------
/// Updates pos, dcell and code from the indicated displacement.
/// The code may be CODE_OUTRHO because in ComputeStepVerlet / Symplectic this is evaluated
/// and is executed before ComputeStepPos.
/// Checks limits depending on maprealposmin and maprealsize, this is valid 
/// for single-GPU because maprealpos and domrealpos are equal. For multi-gpu it is
/// important to mark particles that leave the domain without leaving the map.
///
/// Actualiza pos, dcell y code a partir del desplazamiento indicado.
/// Code puede ser CODE_OUTRHO pq en ComputeStepVerlet/Symplectic se evalua esto 
/// y se ejecuta antes que ComputeStepPos.
/// Comprueba los limites en funcion de maprealposmin y maprealsize esto es valido
/// para single-gpu pq domrealpos y maprealpos son iguales. Para multi-gpu seria 
/// necesario marcar las particulas q salgan del dominio sin salir del mapa.
//------------------------------------------------------------------------------
template<bool periactive> __device__ void KerUpdatePos(
  double2 rxy,double rz,double movx,double movy,double movz
  ,bool outrho,unsigned p,double2* posxy,double* posz,unsigned* dcell
  ,typecode* code)
{
  //-Checks validity of displacement. | Comprueba validez del desplazamiento.
  const bool outmov=(fmaxf(fabsf(float(movx)),fmaxf(fabsf(float(movy)),fabsf(float(movz))))>CTE.movlimit);
  //-Applies diplacement.
  double3 rpos=make_double3(rxy.x,rxy.y,rz);
  rpos.x+=movx; rpos.y+=movy; rpos.z+=movz;
  //-Checks limits of real domain. | Comprueba limites del dominio reales.
  double dx=rpos.x-CTE.maprealposminx;
  double dy=rpos.y-CTE.maprealposminy;
  double dz=rpos.z-CTE.maprealposminz;
  bool out=(dx!=dx || dy!=dy || dz!=dz || dx<0 || dy<0 || dz<0 || dx>=CTE.maprealsizex || dy>=CTE.maprealsizey || dz>=CTE.maprealsizez);
  if(periactive && out){
    bool xperi=(CTE.periactive&1),yperi=(CTE.periactive&2),zperi=(CTE.periactive&4);
    if(xperi){
      if(dx<0)                { dx-=CTE.xperincx; dy-=CTE.xperincy; dz-=CTE.xperincz; }
      if(dx>=CTE.maprealsizex){ dx+=CTE.xperincx; dy+=CTE.xperincy; dz+=CTE.xperincz; }
    }
    if(yperi){
      if(dy<0)                { dx-=CTE.yperincx; dy-=CTE.yperincy; dz-=CTE.yperincz; }
      if(dy>=CTE.maprealsizey){ dx+=CTE.yperincx; dy+=CTE.yperincy; dz+=CTE.yperincz; }
    }
    if(zperi){
      if(dz<0)                { dx-=CTE.zperincx; dy-=CTE.zperincy; dz-=CTE.zperincz; }
      if(dz>=CTE.maprealsizez){ dx+=CTE.zperincx; dy+=CTE.zperincy; dz+=CTE.zperincz; }
    }
    bool outx=!xperi && (dx<0 || dx>=CTE.maprealsizex);
    bool outy=!yperi && (dy<0 || dy>=CTE.maprealsizey);
    bool outz=!zperi && (dz<0 || dz>=CTE.maprealsizez);
    out=(outx||outy||outz);
    rpos=make_double3(dx+CTE.maprealposminx,dy+CTE.maprealposminy,dz+CTE.maprealposminz);
  }
  //-Stores updated position.
  posxy[p]=make_double2(rpos.x,rpos.y);
  posz[p]=rpos.z;
  //-Stores cell and check. | Guarda celda y check.
  if(outrho || outmov || out){//-Particle out. Only brands as excluded normal particles (not periodic). | Particle out. Solo las particulas normales (no periodicas) se pueden marcar como excluidas.
    typecode rcode=code[p];
    if(out)rcode=CODE_SetOutPos(rcode);
    else if(outrho)rcode=CODE_SetOutRho(rcode);
    else rcode=CODE_SetOutMov(rcode);
    code[p]=rcode;
    dcell[p]=DCEL_CodeMapOut;
  }
  else{//-Particle in.
    if(periactive){
      dx=rpos.x-CTE.domposminx;
      dy=rpos.y-CTE.domposminy;
      dz=rpos.z-CTE.domposminz;
    }
    const unsigned cx=unsigned(dx/CTE.scell);
    const unsigned cy=unsigned(dy/CTE.scell);
    const unsigned cz=unsigned(dz/CTE.scell);
    dcell[p]=DCEL_Cell(CTE.cellcode,cx,cy,cz);
  }
}

//------------------------------------------------------------------------------
/// Returns the corrected position after applying periodic conditions.
/// Devuelve la posicion corregida tras aplicar condiciones periodicas.
//------------------------------------------------------------------------------
__device__ double3 KerUpdatePeriodicPos(double3 ps)
{
  double dx=ps.x-CTE.maprealposminx;
  double dy=ps.y-CTE.maprealposminy;
  double dz=ps.z-CTE.maprealposminz;
  const bool out=(dx!=dx || dy!=dy || dz!=dz || dx<0 || dy<0 || dz<0 || dx>=CTE.maprealsizex || dy>=CTE.maprealsizey || dz>=CTE.maprealsizez);
  //-Adjusts position according to periodic conditions and rechecks domain limits.
  //-Ajusta posicion segun condiciones periodicas y vuelve a comprobar los limites del dominio.
  if(out){
    bool xperi=(CTE.periactive&1),yperi=(CTE.periactive&2),zperi=(CTE.periactive&4);
    if(xperi){
      if(dx<0)                { dx-=CTE.xperincx; dy-=CTE.xperincy; dz-=CTE.xperincz; }
      if(dx>=CTE.maprealsizex){ dx+=CTE.xperincx; dy+=CTE.xperincy; dz+=CTE.xperincz; }
    }
    if(yperi){
      if(dy<0)                { dx-=CTE.yperincx; dy-=CTE.yperincy; dz-=CTE.yperincz; }
      if(dy>=CTE.maprealsizey){ dx+=CTE.yperincx; dy+=CTE.yperincy; dz+=CTE.yperincz; }
    }
    if(zperi){
      if(dz<0)                { dx-=CTE.zperincx; dy-=CTE.zperincy; dz-=CTE.zperincz; }
      if(dz>=CTE.maprealsizez){ dx+=CTE.zperincx; dy+=CTE.zperincy; dz+=CTE.zperincz; }
    }
    ps=make_double3(dx+CTE.maprealposminx,dy+CTE.maprealposminy,dz+CTE.maprealposminz);
  }
  return(ps);
}

//------------------------------------------------------------------------------
/// Helper function for No penetration algorithm.
//------------------------------------------------------------------------------
__device__ void ComputeNoPenVel(const float dv,const float norm,const float dr
  ,float& nopencount,float& nopenshift)
{
  const float vfc=dv*norm;
  if(vfc<0.f){//-fluid particle moving towards boundary?
    const float ratio=max(abs(dr/norm),0.25f);
    const float factor=-4.f*ratio+3.f;
    nopencount+=1.f; //-boundary particle counter for average
                //-delta v = sum uij dot (nj cross nj)
    nopenshift-=factor*dv*norm*norm;
  }
}


//##############################################################################
//# Kernels for calculating forces (Pos-Double).
//# Kernels para calculo de fuerzas (Pos-Double).
//##############################################################################
//------------------------------------------------------------------------------
/// Interaction of a particle with a set of particles. Bound-Fluid/Float
/// Realiza la interaccion de una particula con un conjunto de ellas. Bound-Fluid/Float
//------------------------------------------------------------------------------
template<TpKernel tker,TpFtMode ftmode>
  __device__ void KerInteractionForcesBoundBox
  (unsigned p1,const unsigned& pini,const unsigned& pfin
  ,const float* ftomassp
  ,const float4* poscell,const float4* velrho,const typecode* code,const unsigned* idp
  ,float massf,const float4& pscellp1,const float4& velrhop1,float& arp1,float& visc)
{
  for(int p2=pini;p2<pfin;p2++){
    const float4 pscellp2=poscell[p2];
    const float drx=pscellp1.x-pscellp2.x + CTE.poscellsize*(PSCEL_GetfX(pscellp1.w)-PSCEL_GetfX(pscellp2.w));
    const float dry=pscellp1.y-pscellp2.y + CTE.poscellsize*(PSCEL_GetfY(pscellp1.w)-PSCEL_GetfY(pscellp2.w));
    const float drz=pscellp1.z-pscellp2.z + CTE.poscellsize*(PSCEL_GetfZ(pscellp1.w)-PSCEL_GetfZ(pscellp2.w));
    const float rr2=drx*drx+dry*dry+drz*drz;
    if(rr2<=CTE.kernelsize2 && rr2>=ALMOSTZERO){
      //-Computes kernel.
      const float fac=cufsph::GetKernel_Fac<tker>(rr2);
      const float frx=fac*drx,fry=fac*dry,frz=fac*drz; //-Gradients.

      const float4 velrhop2=velrho[p2];
      //-Obtains particle mass p2 if there are floating bodies.
      //-Obtiene masa de particula p2 en caso de existir floatings.
      float ftmassp2;    //-Contains mass of floating body or massf if fluid. | Contiene masa de particula floating o massf si es fluid.
      bool compute=true; //-Deactivated when DEM is used and is float-float or float-bound. | Se desactiva cuando se usa DEM y es float-float o float-bound.
      if(USE_FLOATING){
        const typecode cod=code[p2];
        bool ftp2=CODE_IsFloating(cod);
        ftmassp2=(ftp2? ftomassp[CODE_GetTypeValue(cod)]: massf);
        compute=!(USE_FTEXTERNAL && ftp2); //-Deactivated when DEM or Chrono is used and is bound-float. | Se desactiva cuando se usa DEM o Chrono y es bound-float.
      }

      if(compute){
        //-Density derivative (Continuity equation).
        const float dvx=velrhop1.x-velrhop2.x, dvy=velrhop1.y-velrhop2.y, dvz=velrhop1.z-velrhop2.z;
        arp1+=(USE_FLOATING? ftmassp2: massf)*(dvx*frx+dvy*fry+dvz*frz)*(velrhop1.w/velrhop2.w);

        {//===== Viscosity ===== 
          const float dot=drx*dvx + dry*dvy + drz*dvz;
          const float dot_rr2=dot/(rr2+CTE.eta2);
          visc=max(dot_rr2,visc); 
        }
      }
    }
  }
}

//------------------------------------------------------------------------------
/// Particle interaction. Bound-Fluid/Float
/// Realiza interaccion entre particulas. Bound-Fluid/Float
//------------------------------------------------------------------------------
template<TpKernel tker,TpFtMode ftmode> 
  __global__ void KerInteractionForcesBound(unsigned n,unsigned pinit
  ,int scelldiv,int4 nc,int3 cellzero,const int2* beginendcellfluid,const unsigned* dcell
  ,const float* ftomassp
  ,const float4* poscell,const float4* velrho,const typecode* code,const unsigned* idp
  ,float* viscdt,float* ar)
{
  const unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of thread.
  if(p<n){
    const unsigned p1=p+pinit;      //-Number of particle.
    float visc=0,arp1=0;

    //-Loads particle p1 data.
    const float4 pscellp1=poscell[p1];
    const float4 velrhop1=velrho[p1];
    
    //-Obtains neighborhood search limits.
    int ini1,fin1,ini2,fin2,ini3,fin3;
    cunsearch::InitCte(dcell[p1],scelldiv,nc,cellzero,ini1,fin1,ini2,fin2,ini3,fin3);

    //-Boundary-Fluid interaction.
    for(int c3=ini3;c3<fin3;c3+=nc.w)for(int c2=ini2;c2<fin2;c2+=nc.x){
      unsigned pini,pfin=0;  cunsearch::ParticleRange(c2,c3,ini1,fin1,beginendcellfluid,pini,pfin);
      if(pfin){
        KerInteractionForcesBoundBox<tker,ftmode> (p1,pini,pfin,ftomassp,poscell
          ,velrho,code,idp,CTE.massf,pscellp1,velrhop1,arp1,visc);
      }
    }
    //-Stores results.
    if(arp1 || visc){
      ar[p1]+=arp1;
      if(visc>viscdt[p1])viscdt[p1]=visc;
    }
  }
}

//------------------------------------------------------------------------------
/// Interaction of a particle with a set of particles. (Fluid/Float-Fluid/Float/Bound)
/// Realiza la interaccion de una particula con un conjunto de ellas. (Fluid/Float-Fluid/Float/Bound)
//------------------------------------------------------------------------------
template<TpKernel tker,TpFtMode ftmode,TpVisco tvisco,TpDensity tdensity
  ,bool shift,TpMdbc2Mode mdbc2
  ,bool shiftadv,bool aleform,bool ncpress,bool divclean> //<vs_advshift>
  __device__ void KerInteractionForcesFluidBox
  (bool boundp2,unsigned p1,const unsigned& pini,const unsigned& pfin,float visco
  ,const float* ftomassp,const float2* tauff,const float3* dengradcorr
  ,const float4* poscell,const float4* velrho,const typecode* code,const unsigned* idp
  ,const byte* boundmode,const float3* tangenvel,const float3* motionvel,const float3* boundnorm //<vs_m2dbc>
  ,float massp2,bool ftp1
  ,const float4& pscellp1,const float4& velrhop1,float pressp1
  ,const float2& taup1_xx_xy,const float2& taup1_xz_yy,const float2& taup1_yz_zz
  ,float2& two_strainp1_xx_xy,float2& two_strainp1_xz_yy,float2& two_strainp1_yz_zz
  ,float3& acep1,float& arp1,float& visc,float& deltap1
  ,TpShifting shiftmode,float4& shiftposfsp1,float3& nopenshift, float3& nopencount
  ,float& fs_treshold,unsigned& neigh,float& pou //<vs_advshift>
  ,const float4* shiftvel,float3& presssym,float3& pressasym,tmatrix3f& lcorr,const float3 shiftp1 //<vs_advshift>
  ,const float* psiclean,float& psicleanr,const float psicleanp1)     //<vs_divclean>
{
  for(int p2=pini;p2<pfin;p2++){
    const float4 pscellp2=poscell[p2];
    const float drx=pscellp1.x-pscellp2.x + CTE.poscellsize*(PSCEL_GetfX(pscellp1.w)-PSCEL_GetfX(pscellp2.w));
    const float dry=pscellp1.y-pscellp2.y + CTE.poscellsize*(PSCEL_GetfY(pscellp1.w)-PSCEL_GetfY(pscellp2.w));
    const float drz=pscellp1.z-pscellp2.z + CTE.poscellsize*(PSCEL_GetfZ(pscellp1.w)-PSCEL_GetfZ(pscellp2.w));
    const float rr2=drx*drx+dry*dry+drz*drz;
    if(rr2<=CTE.kernelsize2 && rr2>=ALMOSTZERO){
      //-Computes kernel.
      const float fac=cufsph::GetKernel_Fac<tker>(rr2);
      const float frx=fac*drx,fry=fac*dry,frz=fac*drz; //-Gradients.

      //-Obtains mass of particle p2 if any floating bodies exist.
      //-Obtiene masa de particula p2 en caso de existir floatings.
      bool ftp2=false;         //-Indicates if it is floating. | Indica si es floating.
      float ftmassp2;    //-Contains mass of floating body or massf if fluid. | Contiene masa de particula floating o massp2 si es bound o fluid.
      bool compute=true; //-Deactivated when DEM is used and is float-float or float-bound. | Se desactiva cuando se usa DEM y es float-float o float-bound.
      if(USE_FLOATING){
        const typecode cod=code[p2];
        ftp2=CODE_IsFloating(cod);
        ftmassp2=(ftp2? ftomassp[CODE_GetTypeValue(cod)]: massp2);
        #ifdef DELTA_HEAVYFLOATING
          if(ftp2 && tdensity==DDT_DDT && ftmassp2<=(massp2*1.2f))deltap1=FLT_MAX;
        #else
          if(ftp2 && tdensity==DDT_DDT)deltap1=FLT_MAX;
        #endif
        if(ftp2 && shift && shiftmode==SHIFT_NoBound)shiftposfsp1.x=FLT_MAX; //-Cancels shifting with floating bodies. | Con floatings anula shifting.
        compute=!(USE_FTEXTERNAL && ftp1 && (boundp2 || ftp2)); //-Deactivated when DEM or Chrono is used and is float-float or float-bound. | Se desactiva cuando se usa DEM o Chrono y es float-float o float-bound.
      }
      //-Changing the mass of boundary particle with boundmode. //<vs_m2dbc_ini>
      if(mdbc2>=MDBC2_Std && boundp2 && !ftp2){
        if(boundmode[p2]==BMODE_MDBC2OFF)massp2=0;
      } //<vs_m2dbc_end>

      const float4 velrhop2=velrho[p2];
      //-Velocity derivative (Momentum equation).
      if(compute && !ncpress){
        const float pressp2=cufsph::ComputePressCte(velrhop2.w);
        const float prs=(pressp1+pressp2)/(velrhop1.w*velrhop2.w)
          +(tker==KERNEL_Cubic? cufsph::GetKernelCubic_Tensil(rr2,velrhop1.w,pressp1,velrhop2.w,pressp2): 0);
        const float p_vpm=-prs*(USE_FLOATING? ftmassp2: massp2);
        acep1.x+=p_vpm*frx; acep1.y+=p_vpm*fry; acep1.z+=p_vpm*frz;
      }

      if(ncpress && compute){ //<vs_advshift_ini>
        const float pressp2=cufsph::ComputePressCte(velrhop2.w);
        const float prs=(pressp1+pressp2)/(velrhop1.w*velrhop2.w)
          +(tker==KERNEL_Cubic? cufsph::GetKernelCubic_Tensil(rr2,velrhop1.w,pressp1,velrhop2.w,pressp2): 0);
        const float p_vpm=-prs*(USE_FLOATING? ftmassp2: massp2);
        const float ncprs=(-pressp1+pressp2)/(velrhop1.w*velrhop2.w);
        const float ncp_vpm=-ncprs*(USE_FLOATING? ftmassp2: massp2);
        presssym.x+=p_vpm*frx; presssym.y+=p_vpm*fry; presssym.z+=p_vpm*frz;
        pressasym.x+=ncp_vpm*frx; pressasym.y+=ncp_vpm*fry; pressasym.z+=ncp_vpm*frz;
      } //<vs_advshift_end>    

      //-Density derivative (Continuity equation).
      float dvx=velrhop1.x-velrhop2.x, dvy=velrhop1.y-velrhop2.y, dvz=velrhop1.z-velrhop2.z;
      #ifndef MDBC2_KEEPVEL
      if(mdbc2>=MDBC2_Std && boundp2 && !ftp2){ //<vs_m2dbc_ini>
        const float3 movvelp2=motionvel[p2];
        dvx=velrhop1.x-movvelp2.x; //-mDBC2 no slip.
        dvy=velrhop1.y-movvelp2.y;
        dvz=velrhop1.z-movvelp2.z;
      } //<vs_m2dbc_end>
      #endif
      if(compute)arp1+=(USE_FLOATING? ftmassp2: massp2)*(dvx*frx+dvy*fry+dvz*frz)*(velrhop1.w/velrhop2.w);

      #ifdef AVAILABLE_DIVCLEAN
      if(divclean && compute){
        float psicleanp2=psiclean[p2];
        if(boundp2)psicleanp2=psicleanp1;
        float dvpsiclean=-(psicleanp1+psicleanp2)*massp2/(velrhop2.w);
        acep1.x+=dvpsiclean*frx; acep1.y+=dvpsiclean*fry; acep1.z+=dvpsiclean*frz;
        psicleanr+=CTE.cs0*CTE.cs0*(USE_FLOATING? ftmassp2: massp2)*(dvx*frx+dvy*fry+dvz*frz)/(velrhop2.w);
      }
      #endif

      if(aleform && compute){ //<vs_advshift_ini>
        float4 shiftp2=make_float4(0.f,0.f,0.f,0.f);
        if(!boundp2 && !ftp2 && !ftp1)shiftp2=shiftvel[p2];      

        float massrhop=(USE_FLOATING? ftmassp2: massp2)/velrhop2.w;
        float rhozeroover1=CTE.rhopzero/velrhop1.w;
        float divshiftp1=shiftp1.x*frx+shiftp1.y*fry+shiftp1.z*frz;
        float divshiftp2=shiftp2.x*frx+shiftp2.y*fry+shiftp2.z*frz;
        float div_pm1=divshiftp1*massrhop*rhozeroover1;
        float div_pm2=divshiftp2*massrhop*rhozeroover1;
        float dvx=shiftp1.x-shiftp2.x, dvy=shiftp1.y-shiftp2.y, dvz=shiftp1.z-shiftp2.z;

        acep1.x+=velrhop1.x*div_pm1;  acep1.y+=velrhop1.y*div_pm1;  acep1.z+=velrhop1.z*div_pm1;
        acep1.x+=velrhop2.x*div_pm2;  acep1.y+=velrhop2.y*div_pm2;  acep1.z+=velrhop2.z*div_pm2;

        float dotdv=massrhop*(-dvx*frx-dvy*fry-dvz*frz);
        acep1.x-= velrhop1.x*dotdv;   acep1.y-= velrhop1.y*dotdv; acep1.z-= velrhop1.z*dotdv;

        float dvx1=shiftp1.x*velrhop1.w+shiftp2.x*velrhop2.w, dvy1=shiftp1.y*velrhop1.w+shiftp2.y*velrhop2.w, dvz1=shiftp1.z*velrhop1.w+shiftp2.z*velrhop2.w;
        arp1+=massrhop*(dvx1*frx+dvy1*fry+dvz1*frz);
        // //-Density derivative (Continuity equation).
        arp1+=massrhop*(dvx*frx+dvy*fry+dvz*frz)*velrhop1.w;
      } //<vs_advshift_end>

      const float cbar=CTE.cs0;
      const float dot3=(tdensity!=DDT_None || shift? drx*frx+dry*fry+drz*frz: 0);
      //-Density Diffusion Term (Molteni and Colagrossi 2009).
      if(tdensity==DDT_DDT && deltap1!=FLT_MAX){
        const float rhop1over2=velrhop1.w/velrhop2.w;
        const float visc_densi=CTE.ddtkh*cbar*(rhop1over2-1.f)/(rr2+CTE.eta2);
        const float delta=visc_densi*dot3*(USE_FLOATING? ftmassp2: massp2);
        //deltap1=(boundp2? FLT_MAX: deltap1+delta);
        deltap1=(boundp2 && CTE.tboundary==BC_DBC? FLT_MAX: deltap1+delta);
      }
      //-Density Diffusion Term (Fourtakas et al 2019).
      if((tdensity==DDT_DDT2 || (tdensity==DDT_DDT2Full && !boundp2)) && deltap1!=FLT_MAX && !ftp2){
        const float rh=1.f+CTE.ddtgz*drz;
        const float drho=CTE.rhopzero*pow(rh,1.f/CTE.gamma)-CTE.rhopzero;  
        const float visc_densi=CTE.ddtkh*cbar*((velrhop2.w-velrhop1.w)-drho)/(rr2+CTE.eta2);
        const float delta=visc_densi*dot3*massp2/velrhop2.w;
        deltap1=(boundp2? FLT_MAX: deltap1-delta); //-blocks it makes it boil - bloody DBC
      }

      //-Shifting correction.
      if(shift && shiftposfsp1.x!=FLT_MAX){
        const float massrho=(USE_FLOATING? ftmassp2: massp2)/velrhop2.w;
        const bool noshift=(boundp2 && (shiftmode==SHIFT_NoBound || (shiftmode==SHIFT_NoFixed && CODE_IsFixed(code[p2]))));
        shiftposfsp1.x=(noshift? FLT_MAX: shiftposfsp1.x+massrho*frx); //-Removes shifting for the boundaries. | Con boundary anula shifting.
        shiftposfsp1.y+=massrho*fry;
        shiftposfsp1.z+=massrho*frz;
        shiftposfsp1.w-=massrho*dot3;
      }

      //-No-Penetration correction SHABA
      if(boundp2 && mdbc2==MDBC2_NoPen && !ftp2){//<vs_m2dbcNP_ini>
        const float rrmag=sqrt(rr2);
        if(rrmag<1.25f*CTE.dp){ //-if fluid particle is less than 1.25dp from a boundary particle
          const float norm=sqrt(boundnorm[p2].x*boundnorm[p2].x+boundnorm[p2].y*boundnorm[p2].y+boundnorm[p2].z*boundnorm[p2].z);
          const float normx=boundnorm[p2].x/norm; float normy=boundnorm[p2].y/norm; float normz=boundnorm[p2].z/norm;
          const float normdist=(normx*drx+normy*dry+normz*drz);
          if(normdist<0.75f*norm && norm<1.75f*float(CTE.dp)) {//-if normal distance is less than 0.75 boundary normal size and only first layer of bound
            const float3 movvelp2=motionvel[p2];
            float absx=abs(normx);
            float absy=abs(normy);
            float absz=abs(normz);
            // decompose the normal and apply correction in each direction separately
            if(drx*normx<0.75f && absx>0.001f*CTE.dp)cusph::ComputeNoPenVel(velrhop1.x-movvelp2.x,normx,drx,
                                                              nopencount.x,nopenshift.x);
            if(dry*normy<0.75f && absy>0.001f*CTE.dp)cusph::ComputeNoPenVel(velrhop1.y-movvelp2.y,normy,dry,
                                                              nopencount.y,nopenshift.y);
            if(drz*normz<0.75f && absz>0.001f*CTE.dp)cusph::ComputeNoPenVel(velrhop1.z-movvelp2.z,normz,drz,
                                                            nopencount.z,nopenshift.z);
          }
        }
      }//<vs_m2dbcNP_end>

      //-Advanced shifting. //<vs_advshift_ini>
      if(shiftadv && compute){
        const float massrho=(USE_FLOATING? ftmassp2: massp2)/velrhop2.w;        
        const float wab=cufsph::GetKernel_Wab<KERNEL_Wendland>(rr2);
        pou+=wab*massrho;
        fs_treshold-=massrho*(drx*frx+dry*fry+drz*frz);
        if(ncpress && compute){
          float vfrx=frx*massrho; float vfry=fry*massrho; float vfrz=frz*massrho;
          lcorr.a11+=-drx*vfrx; lcorr.a12+=-drx*vfry; lcorr.a13+=-drx*vfrz;
          lcorr.a21+=-dry*vfrx; lcorr.a22+=-dry*vfry; lcorr.a23+=-dry*vfrz;
          lcorr.a31+=-drz*vfrx; lcorr.a32+=-drz*vfry; lcorr.a33+=-drz*vfrz;
        }
      } //<vs_advshift_end>

      //===== Viscosity ===== 
      if(compute){
        if(mdbc2>=MDBC2_Std && boundp2 && !ftp2){ //<vs_m2dbc_ini>
          const float3 tangentvelp2=tangenvel[p2];
          dvx=velrhop1.x-tangentvelp2.x;
          dvy=velrhop1.y-tangentvelp2.y;
          dvz=velrhop1.z-tangentvelp2.z;
        } //<vs_m2dbc_end>
        const float dot=drx*dvx + dry*dvy + drz*dvz;
        const float dot_rr2=dot/(rr2+CTE.eta2);
        visc=max(dot_rr2,visc);  //ViscDt=max(dot/(rr2+Eta2),ViscDt);
        if(tvisco==VISCO_Artificial){//-Artificial viscosity.
          if(dot<0){
            const float amubar=CTE.kernelh*dot_rr2;  //amubar=CTE.kernelh*dot/(rr2+CTE.eta2);
            const float robar=(velrhop1.w+velrhop2.w)*0.5f;
            const float pi_visc=(-visco*cbar*amubar/robar)*(USE_FLOATING? ftmassp2: massp2);
            acep1.x-=pi_visc*frx; acep1.y-=pi_visc*fry; acep1.z-=pi_visc*frz;
          }
        }
        if(tvisco==VISCO_Laminar || tvisco==VISCO_LaminarSPS){//-Laminar and Laminar+SPS viscosity.
          const float robar2=(velrhop1.w+velrhop2.w);
          const float temp=4.f*visco/((rr2+CTE.eta2)*robar2);  //-Simplication of temp=2.0f*visco/((rr2+CTE.eta2)*robar); robar=(rhopp1+velrhop2.w)*0.5f;
          const float vtemp=(USE_FLOATING? ftmassp2: massp2)*temp*(drx*frx+dry*fry+drz*frz);  
          acep1.x+=vtemp*dvx; acep1.y+=vtemp*dvy; acep1.z+=vtemp*dvz;
        }
        if(tvisco==VISCO_LaminarSPS){//-SPS contribution for Laminar viscosity. 
          //-SPS turbulence model.
          //-Note that taup1 is tau_a/rho_a^2 for interaction of particle a and b.
          //-And taup1 is always zero when p1 is not a fluid particle.
          float2 stau_xx_xy=taup1_xx_xy;
          float2 stau_xz_yy=taup1_xz_yy;
          float2 stau_yz_zz=taup1_yz_zz;
          if(!boundp2 && (USE_NOFLOATING || !ftp2)){//-When p2 is fluid.
            //-Note that taup2 is tau_b/rho_b^2 for interaction of particle a and b.
            float2 taup2=tauff[p2*3];     stau_xx_xy.x+=taup2.x; stau_xx_xy.y+=taup2.y;
                   taup2=tauff[p2*3+1];   stau_xz_yy.x+=taup2.x; stau_xz_yy.y+=taup2.y;
                   taup2=tauff[p2*3+2];   stau_yz_zz.x+=taup2.x; stau_yz_zz.y+=taup2.y;
          }
          acep1.x+=(USE_FLOATING? ftmassp2: massp2)*(stau_xx_xy.x*frx + stau_xx_xy.y*fry + stau_xz_yy.x*frz);
          acep1.y+=(USE_FLOATING? ftmassp2: massp2)*(stau_xx_xy.y*frx + stau_xz_yy.y*fry + stau_yz_zz.x*frz);
          acep1.z+=(USE_FLOATING? ftmassp2: massp2)*(stau_xz_yy.x*frx + stau_yz_zz.x*fry + stau_yz_zz.y*frz);
          //-Velocity gradients.
          if(USE_NOFLOATING || !ftp1){//-When p1 is fluid.
            const float volp2=-(USE_FLOATING? ftmassp2: massp2)/velrhop2.w;
            float dv=dvx*volp2; two_strainp1_xx_xy.x+=dv*frx; two_strainp1_xx_xy.y+=dv*fry; two_strainp1_xz_yy.x+=dv*frz;
                  dv=dvy*volp2; two_strainp1_xx_xy.y+=dv*frx; two_strainp1_xz_yy.y+=dv*fry; two_strainp1_yz_zz.x+=dv*frz;
                  dv=dvz*volp2; two_strainp1_xz_yy.x+=dv*frx; two_strainp1_yz_zz.x+=dv*fry; two_strainp1_yz_zz.y+=dv*frz;
            // to compute tau terms we assume that two_strain.xy=dudy+dvdx, two_strain.xz=dudz+dwdx, two_strain.yz=dvdz+dwdy
            // so only 6 elements are needed instead of 3x3.
          }
        }
      }
    }
  }
}

//------------------------------------------------------------------------------
/// Interaction between particles. Fluid/Float-Fluid/Float or Fluid/Float-Bound.
/// Includes artificial/laminar viscosity and normal/DEM floating bodies.
///
/// Realiza interaccion entre particulas. Fluid/Float-Fluid/Float or Fluid/Float-Bound
/// Incluye visco artificial/laminar y floatings normales/dem.
//------------------------------------------------------------------------------
template<TpKernel tker,TpFtMode ftmode,TpVisco tvisco,TpDensity tdensity
  ,bool shift,TpMdbc2Mode mdbc2
  ,bool shiftadv,bool aleform,bool ncpress,bool divclean> //<vs_advshift> //<vs_divclean>
  __global__ void KerInteractionForcesFluid
  (unsigned n,unsigned pinit,float viscob,float viscof
  ,int scelldiv,int4 nc,int3 cellzero,const int2* begincell,unsigned cellfluid
  ,const unsigned* dcell,const float* ftomassp,const float2* tauff,float2* two_strain
  ,const float3* dengradcorr,const float4* poscell,const float4* velrho
  ,const typecode* code,const unsigned* idp
  ,const byte* boundmode,const float3* tangenvel,const float3* motionvel,const float3* boundnormal //<vs_m2dbc>
  ,float* viscdt,float* ar,float3* ace,float* delta
  ,TpShifting shiftmode,float4* shiftposfs, float4* nopenshift                                 //<vs_advshift>
  ,unsigned* fstype,const float4* shiftvel,bool corrector,bool simulate2d //<vs_advshift>
  ,const float* psiclean,float* psicleanrhs,float* cspsiclean,float divcleankp)  //<vs_divclean>
{
  const unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    const unsigned p1=p+pinit;      //-Number of particle.
    float visc=0,arp1=0,deltap1=0;
    float3 acep1=make_float3(0,0,0);

    //-Variables for Shifting.
    float4 shiftposfsp1;
    if(shift)shiftposfsp1=shiftposfs[p1];
    float3 nopenshiftp1 = make_float3(0,0,0); //-no-penetration array
    float3 nopencountp1 = make_float3(0, 0, 0); //-no-penetration array

    //-Variables for Advanced Shifting. //<vs_advshift_ini>
    unsigned neigh=0;
    float fs_treshold=0;
    float pou=0;
    float3 presssym=make_float3(0,0,0);
    float3 pressasym=make_float3(0,0,0);
    tmatrix3f LCorr;      cumath::Tmatrix3fReset(LCorr);
    tmatrix3f LCorr_inv;  cumath::Tmatrix3fReset(LCorr_inv);
    float3 shiftp1=make_float3(0,0,0);
    if(aleform && !CODE_IsFloating(code[p1]))shiftp1=make_float3(shiftvel[p1].x,shiftvel[p1].y,shiftvel[p1].z);

    //-Variable for divergence cleaning.
    float psicleanp1;
    float psicleanr=0;
    if(divclean)psicleanp1=psiclean[p1];

    float Nzero=0;
    if(simulate2d)Nzero=float((3.141592)*CTE.kernelsize2/(CTE.dp*CTE.dp));
    else          Nzero=float((4.f/3.f)*(3.141592)*CTE.kernelsize2*CTE.kernelsize/(CTE.dp*CTE.dp*CTE.dp));
    //<vs_advshift_end>

    //-Obtains data of particle p1 in case there are floating bodies.
    bool ftp1;       //-Indicates if it is floating. | Indica si es floating.
    if(USE_FLOATING){
      const typecode cod=code[p1];
      ftp1=CODE_IsFloating(cod);
      if(ftp1 && tdensity!=DDT_None)deltap1=FLT_MAX; //-DDT is not applied to floating particles.
      if(ftp1 && shift)shiftposfsp1.x=FLT_MAX; //-Shifting is not calculated for floating bodies. | Para floatings no se calcula shifting.
    }

    //-Obtains basic data of particle p1.
    const float4 pscellp1=poscell[p1];
    const float4 velrhop1=velrho[p1];
    const float pressp1=cufsph::ComputePressCte(velrhop1.w);

    //-Variables for Laminar+SPS.
    float2 taup1_xx_xy,taup1_xz_yy,taup1_yz_zz; //-Note that taup1 is tau_a/rho_a^2.
    if(tvisco==VISCO_LaminarSPS){
      taup1_xx_xy=tauff[p1*3];
      taup1_xz_yy=tauff[p1*3+1];
      taup1_yz_zz=tauff[p1*3+2];
    }
    //-Variables for Laminar+SPS (computation).
    float2 two_strainp1_xx_xy,two_strainp1_xz_yy,two_strainp1_yz_zz;
    if(tvisco==VISCO_LaminarSPS){
      two_strainp1_xx_xy=make_float2(0,0);
      two_strainp1_xz_yy=make_float2(0,0);
      two_strainp1_yz_zz=make_float2(0,0);
    }

    //-Obtains neighborhood search limits.
    int ini1,fin1,ini2,fin2,ini3,fin3;
    cunsearch::InitCte(dcell[p1],scelldiv,nc,cellzero,ini1,fin1,ini2,fin2,ini3,fin3);

    //-Interaction with fluids.
    ini3+=cellfluid; fin3+=cellfluid;
    for(int c3=ini3;c3<fin3;c3+=nc.w)for(int c2=ini2;c2<fin2;c2+=nc.x){
      unsigned pini,pfin=0;  cunsearch::ParticleRange(c2,c3,ini1,fin1,begincell,pini,pfin);
      if(pfin){
        KerInteractionForcesFluidBox<tker,ftmode,tvisco,tdensity,shift,mdbc2,shiftadv,aleform,ncpress,divclean>
          (false,p1,pini,pfin,viscof,ftomassp,tauff,dengradcorr,poscell,velrho,code,idp
          ,boundmode,tangenvel,motionvel,boundnormal //<vs_m2dbc>
          ,CTE.massf,ftp1,pscellp1,velrhop1,pressp1,taup1_xx_xy,taup1_xz_yy,taup1_yz_zz
          ,two_strainp1_xx_xy,two_strainp1_xz_yy,two_strainp1_yz_zz,acep1,arp1,visc
          ,deltap1,shiftmode,shiftposfsp1,nopenshiftp1,nopencountp1
          ,fs_treshold,neigh,pou,shiftvel,presssym,pressasym,LCorr,shiftp1 //<vs_advshift>
          ,psiclean,psicleanr,psicleanp1); //<vs_divclean>
      }
    }
    //-Interaction with boundaries.
    ini3-=cellfluid; fin3-=cellfluid;
    for(int c3=ini3;c3<fin3;c3+=nc.w)for(int c2=ini2;c2<fin2;c2+=nc.x){
      unsigned pini,pfin=0;  cunsearch::ParticleRange(c2,c3,ini1,fin1,begincell,pini,pfin);
      if(pfin){
        KerInteractionForcesFluidBox<tker,ftmode,tvisco,tdensity,shift,mdbc2,shiftadv,aleform,ncpress,divclean>
          (true,p1,pini,pfin,viscob,ftomassp,tauff,NULL,poscell,velrho,code,idp
          ,boundmode,tangenvel,motionvel,boundnormal //<vs_m2dbc>
          ,CTE.massb,ftp1,pscellp1,velrhop1,pressp1,taup1_xx_xy,taup1_xz_yy,taup1_yz_zz
          ,two_strainp1_xx_xy,two_strainp1_xz_yy,two_strainp1_yz_zz,acep1,arp1,visc
          ,deltap1,shiftmode,shiftposfsp1,nopenshiftp1,nopencountp1
          ,fs_treshold,neigh,pou,shiftvel,presssym,pressasym,LCorr,shiftp1 //<vs_advshift>
          ,psiclean,psicleanr,psicleanp1);   //<vs_divclean>
      }
    }

    //<vs_advshift_ini>
    pou+=cufsph::GetKernel_Wab<tker>(0.f)*CTE.massf/velrhop1.w;
    if(ncpress){
      if(fstype[p1]==0 && pou>0.95f){
        if(simulate2d){
          tmatrix2f Lcorr2D;
          tmatrix2f Lcorr2D_inv;
          Lcorr2D.a11=LCorr.a11; Lcorr2D.a12=LCorr.a13;
          Lcorr2D.a21=LCorr.a31; Lcorr2D.a22=LCorr.a33;
          float lcorr_det=(Lcorr2D.a11*Lcorr2D.a22-Lcorr2D.a12*Lcorr2D.a21);
          Lcorr2D_inv.a11=Lcorr2D.a22/lcorr_det; Lcorr2D_inv.a12=-Lcorr2D.a12/lcorr_det; Lcorr2D_inv.a22=Lcorr2D.a11/lcorr_det; Lcorr2D_inv.a21=-Lcorr2D.a21/lcorr_det;
          LCorr_inv.a11=Lcorr2D_inv.a11;  LCorr_inv.a13=Lcorr2D_inv.a12;
          LCorr_inv.a31=Lcorr2D_inv.a21;  LCorr_inv.a33=Lcorr2D_inv.a22;
        }
        else{
          const float determ = cumath::Determinant3x3(LCorr);
          LCorr_inv = cumath::InverseMatrix3x3(LCorr, determ);
        }
        acep1.x+=pressasym.x*LCorr_inv.a11 + pressasym.y*LCorr_inv.a12 + pressasym.z*LCorr_inv.a13;
        acep1.y+=pressasym.x*LCorr_inv.a21 + pressasym.y*LCorr_inv.a22 + pressasym.z*LCorr_inv.a23;
        acep1.z+=pressasym.x*LCorr_inv.a31 + pressasym.y*LCorr_inv.a32 + pressasym.z*LCorr_inv.a33;
      }
      else{
        acep1.x+=presssym.x; acep1.y+=presssym.y; acep1.z+=presssym.z;
      }
    }

    if(corrector && shiftadv){
      unsigned fstypep1=0;
      if(simulate2d){
        if(fs_treshold<1.7) fstypep1=2;
        if(fs_treshold<1.1 && Nzero/float(neigh)<0.4f) fstypep1=3;
      } else {
        if(fs_treshold<2.75) fstypep1=2;
        if(fs_treshold<1.8 && Nzero/float(neigh)<0.4f) fstypep1=3;
      }
      fstype[p1]=fstypep1;
    }
    //<vs_advshift_end>

    //-Stores results.
    if(shift||arp1||acep1.x||acep1.y||acep1.z||visc){
      if(tdensity!=DDT_None){
        if(delta){
          const float rdelta=delta[p1];
          delta[p1]=(rdelta==FLT_MAX || deltap1==FLT_MAX? FLT_MAX: rdelta+deltap1);
        }
        else if(deltap1!=FLT_MAX)arp1+=deltap1;
      }
      ar[p1]+=arp1;
      float3 r=ace[p1]; r.x+=acep1.x; r.y+=acep1.y; r.z+=acep1.z; ace[p1]=r;
      if(visc>viscdt[p1])viscdt[p1]=visc;
      if(tvisco==VISCO_LaminarSPS){
        float2 rg;
        rg=two_strain[p1*3  ];  rg=make_float2(rg.x+two_strainp1_xx_xy.x,rg.y+two_strainp1_xx_xy.y);  two_strain[p1*3  ]=rg;
        rg=two_strain[p1*3+1];  rg=make_float2(rg.x+two_strainp1_xz_yy.x,rg.y+two_strainp1_xz_yy.y);  two_strain[p1*3+1]=rg;
        rg=two_strain[p1*3+2];  rg=make_float2(rg.x+two_strainp1_yz_zz.x,rg.y+two_strainp1_yz_zz.y);  two_strain[p1*3+2]=rg;
      }
      if(shift)shiftposfs[p1]=shiftposfsp1;
    }
    //-No-Penetration correction SHABA
    if (mdbc2==MDBC2_NoPen){ //<vs_m2dbcNP_end>
      if(nopencountp1.x>0.f|| nopencountp1.y>0.f|| nopencountp1.z>0.f){
        //-Average correction velocity over number of boundary particles
        if (nopencountp1.x>0.f){//-if correction required x
        nopenshift[p1].x=nopenshiftp1.x/ nopencountp1.x;
        nopenshift[p1].w=10; //-correction needed? yes
        }
        if (nopencountp1.y>0.f){//-if correction required y
            nopenshift[p1].y=nopenshiftp1.y/ nopencountp1.y;
            nopenshift[p1].w=10; //-correction needed? yes
        }
        if (nopencountp1.z>0.f){//-if correction required z
            nopenshift[p1].z=nopenshiftp1.z/ nopencountp1.z;
            nopenshift[p1].w=10; //-correction needed? yes
        }
      }
      else{
        nopenshift[p1].w=0;//-correction needed? no
      }
    }//<vs_m2dbcNP_end>
    
    #ifdef AVAILABLE_DIVCLEAN
    //<vs_divclean_ini>
    if(divclean){
      float cs1=divcleankp*sqrt(fabs(pressp1/velrhop1.w));
      cspsiclean[p1]=cs1;
      psicleanrhs[p1]+=psicleanr-psiclean[p1]*max(cs1,CTE.cs0)/CTE.kernelh;
    }
    //<vs_divclean_end>
    #endif
  }
}

#ifndef DISABLE_BSMODES
//==============================================================================
/// Collects kernel information.
//==============================================================================
template<TpKernel tker,TpFtMode ftmode,TpVisco tvisco,TpDensity tdensity
  ,bool shift,TpMdbc2Mode mdbc2,bool shiftadv,bool aleform,bool ncpress,bool divclean> 
  void Interaction_ForcesT_KerInfo(StKerInfo* kerinfo)
{
 #if CUDART_VERSION >= 6050
  {
    typedef void (*fun_ptr)(unsigned,unsigned,float,float,int,int4,int3
      ,const int2*,unsigned,const unsigned*,const float*,const float2*
      ,float2*,const float3*,const float4*,const float4*,const typecode*
      ,const unsigned*
      ,const byte*,const float3*,const float3*,const float3* //<vs_m2dbc>
      ,float*,float*,float3*,float*,TpShifting,float4*,float4*
      ,unsigned*,const float4*,bool,bool    //<vs_advshift>
      ,const float*,float*,float*,float); //<vs_divclean>
    fun_ptr ptr=&KerInteractionForcesFluid<tker,ftmode,tvisco,tdensity,shift,mdbc2,shiftadv,aleform,ncpress,divclean>;
    int qblocksize=0,mingridsize=0;
    hipOccupancyMaxPotentialBlockSize(&mingridsize,&qblocksize,(void*)ptr,0,0);
    struct hipFuncAttributes attr;
    hipFuncGetAttributes(&attr,reinterpret_cast<const void*>((void*)ptr));
    kerinfo->forcesfluid_bs=qblocksize;
    kerinfo->forcesfluid_rg=attr.numRegs;
    kerinfo->forcesfluid_bsmax=attr.maxThreadsPerBlock;
    //printf(">> KerInteractionForcesFluid  blocksize:%u (%u)\n",qblocksize,0);
  }
  {
    typedef void (*fun_ptr)(unsigned,unsigned,int,int4,int3,const int2*
      ,const unsigned*,const float*,const float4*,const float4*
      ,const typecode*,const unsigned*,float*,float*);
    fun_ptr ptr=&KerInteractionForcesBound<tker,ftmode>;
    int qblocksize=0,mingridsize=0;
    hipOccupancyMaxPotentialBlockSize(&mingridsize,&qblocksize,(void*)ptr,0,0);
    struct hipFuncAttributes attr;
    hipFuncGetAttributes(&attr,reinterpret_cast<const void*>((void*)ptr));
    kerinfo->forcesbound_bs=qblocksize;
    kerinfo->forcesbound_rg=attr.numRegs;
    kerinfo->forcesbound_bsmax=attr.maxThreadsPerBlock;
    //printf(">> KerInteractionForcesBound  blocksize:%u (%u)\n",qblocksize,0);
  }
  fcuda::Check_CudaErroorFun("Error collecting kernel information.");
 #endif
}
#endif

//==============================================================================
/// Interaction for the force computation.
/// Interaccion para el calculo de fuerzas.
//==============================================================================
template<TpKernel tker,TpFtMode ftmode,TpVisco tvisco,TpDensity tdensity
  ,bool shift,TpMdbc2Mode mdbc2,bool shiftadv,bool aleform,bool ncpress,bool divclean> 
  void Interaction_ForcesGpuT(const StInterParmsg& t)
{
  //-Collects kernel information.
#ifndef DISABLE_BSMODES
  if(t.kerinfo){
    Interaction_ForcesT_KerInfo<tker,ftmode,tvisco,tdensity,shift,mdbc2,shiftadv,aleform,ncpress,divclean>(t.kerinfo);
    return;
  }
#endif
  const StDivDataGpu& dvd=t.divdatag;
  //-Interaction Fluid-Fluid & Fluid-Bound.
  if(t.fluidnum){
    //printf("[ns:%u  id:%d] halo:%d fini:%d(%d) bini:%d(%d)\n",t.nstep,t.id,t.halo,t.fluidini,t.fluidnum,t.boundini,t.boundnum);
    dim3 sgridf=GetSimpleGridSize(t.fluidnum,t.bsfluid);
    KerInteractionForcesFluid<tker,ftmode,tvisco,tdensity,shift,mdbc2,shiftadv,aleform,ncpress,divclean> <<<sgridf,t.bsfluid,0,t.stm>>> 
      (t.fluidnum,t.fluidini,t.viscob,t.viscof,dvd.scelldiv,dvd.nc,dvd.cellzero,dvd.beginendcell,dvd.cellfluid,t.dcell
      ,t.ftomassp,(const float2*)t.spstaurho2,(float2*)t.sps2strain,t.dengradcorr,t.poscell,t.velrho,t.code,t.idp
      ,t.boundmode,t.tangenvel,t.motionvel,t.boundnormal //<vs_m2dbc>
      ,t.viscdt,t.ar,t.ace,t.delta,t.shiftmode,t.shiftposfs,t.nopenshift
      ,t.fstype,t.shiftvel,t.corrector,t.simulate2d  //<vs_advshift>
      ,t.psiclean,t.psicleanrhs,t.cspsiclean,t.divcleankp);        //<vs_divclean>
  }
  //-Interaction Boundary-Fluid.
  if(t.boundnum){
    const int2* beginendcellfluid=dvd.beginendcell+dvd.cellfluid;
    dim3 sgridb=GetSimpleGridSize(t.boundnum,t.bsbound);
    //printf("bsbound:%u\n",bsbound);
    KerInteractionForcesBound<tker,ftmode> <<<sgridb,t.bsbound,0,t.stm>>> 
      (t.boundnum,t.boundini,dvd.scelldiv,dvd.nc,dvd.cellzero,beginendcellfluid,t.dcell
      ,t.ftomassp,t.poscell,t.velrho,t.code,t.idp,t.viscdt,t.ar);
  }
}

// #define FAST_COMPILATION
//==============================================================================
template<TpKernel tker,TpFtMode ftmode,TpVisco tvisco,TpDensity tdensity,bool shift
  ,TpMdbc2Mode mdbc2,bool shiftadv,bool aleform,bool ncpress>
  void Interaction_Forces_gt5(const StInterParmsg& t)
{
  #ifdef AVAILABLE_DIVCLEAN
  if(t.divclean) Interaction_ForcesGpuT<tker,ftmode,tvisco,tdensity,shift,mdbc2,shiftadv,aleform,ncpress,true>(t);
  else           Interaction_ForcesGpuT<tker,ftmode,tvisco,tdensity,shift,mdbc2,shiftadv,aleform,ncpress,false>(t);
  #else 
  Interaction_ForcesGpuT<tker,ftmode,tvisco,tdensity,shift,mdbc2,shiftadv,aleform,ncpress,false>(t);
  #endif
}
//==============================================================================
template<TpKernel tker,TpFtMode ftmode,TpVisco tvisco,TpDensity tdensity,bool shift,TpMdbc2Mode mdbc2>
  void Interaction_Forces_gt4(const StInterParmsg& t)
{
  if(t.shiftadv){
    if(t.aleform){ const bool ale=true;
      if(t.ncpress)Interaction_Forces_gt5<tker,ftmode,tvisco,tdensity,shift,mdbc2,true,ale,true > (t);
      else         Interaction_Forces_gt5<tker,ftmode,tvisco,tdensity,shift,mdbc2,true,ale,false> (t);
    }
    else{          const bool ale=false;
      if(t.ncpress)Interaction_Forces_gt5<tker,ftmode,tvisco,tdensity,shift,mdbc2,true,ale,true > (t);
      else         Interaction_Forces_gt5<tker,ftmode,tvisco,tdensity,shift,mdbc2,true,ale,false> (t);
    }    
  } //<vs_advshift_end>
  else             Interaction_Forces_gt5<tker,ftmode,tvisco,tdensity,shift,mdbc2,false,false,false> (t);
}

//==============================================================================
template<TpKernel tker,TpFtMode ftmode,TpVisco tvisco,TpDensity tdensity,bool shift>
  void Interaction_Forces_gt3(const StInterParmsg& t)
{
  if(t.mdbc2==MDBC2_None) Interaction_Forces_gt4<tker,ftmode,tvisco,tdensity,shift,MDBC2_None>   (t); //<vs_m2dbcNp>
  if(t.mdbc2==MDBC2_Std)  Interaction_Forces_gt4<tker,ftmode,tvisco,tdensity,shift,MDBC2_Std>    (t); //<vs_m2dbcNp>
  if(t.mdbc2==MDBC2_NoPen)Interaction_Forces_gt4<tker,ftmode,tvisco,tdensity,shift,MDBC2_NoPen>  (t); //<vs_m2dbcNp>
}
//==============================================================================
template<TpKernel tker,TpFtMode ftmode,TpVisco tvisco>
  void Interaction_Forces_gt2(const StInterParmsg& t)
{
#ifdef FAST_COMPILATION
  if(t.shiftmode){              const bool shift=true;
    if(t.tdensity==DDT_DDT2Full)Interaction_Forces_gt3<tker,ftmode,tvisco,DDT_DDT2Full,shift> (t);
    else throw "Only DDT==DDT_DDT2Full is compiled for FastCompilation...";
  }
  else{                         const bool shift=false;
    if(t.tdensity==DDT_DDT2Full)Interaction_Forces_gt3<tker,ftmode,tvisco,DDT_DDT2Full,shift> (t);
    else throw "Only DDT==DDT_DDT2Full is compiled for FastCompilation...";
  }
#else
  if(t.shiftmode){              const bool shift=true;
    if(t.tdensity==DDT_None)    Interaction_Forces_gt3<tker,ftmode,tvisco,DDT_None    ,shift> (t);
    if(t.tdensity==DDT_DDT)     Interaction_Forces_gt3<tker,ftmode,tvisco,DDT_DDT     ,shift> (t);
    if(t.tdensity==DDT_DDT2)    Interaction_Forces_gt3<tker,ftmode,tvisco,DDT_DDT2    ,shift> (t);
    if(t.tdensity==DDT_DDT2Full)Interaction_Forces_gt3<tker,ftmode,tvisco,DDT_DDT2Full,shift> (t);
  }
  else{                         const bool shift=false;
    if(t.tdensity==DDT_None)    Interaction_Forces_gt3<tker,ftmode,tvisco,DDT_None    ,shift> (t);
    if(t.tdensity==DDT_DDT)     Interaction_Forces_gt3<tker,ftmode,tvisco,DDT_DDT     ,shift> (t);
    if(t.tdensity==DDT_DDT2)    Interaction_Forces_gt3<tker,ftmode,tvisco,DDT_DDT2    ,shift> (t);
    if(t.tdensity==DDT_DDT2Full)Interaction_Forces_gt3<tker,ftmode,tvisco,DDT_DDT2Full,shift> (t);
  }
#endif
}
//==============================================================================
template<TpKernel tker,TpFtMode ftmode> 
  void Interaction_Forces_gt1(const StInterParmsg& t)
{
#ifdef FAST_COMPILATION
 if(t.tvisco!=VISCO_Artificial)throw "Extra viscosity options are disabled for FastCompilation...";
 Interaction_Forces_gt2<tker,ftmode,VISCO_Artificial> (t);
#else
  if(t.tvisco==VISCO_Artificial)     Interaction_Forces_gt2<tker,ftmode,VISCO_Artificial>(t);
  else if(t.tvisco==VISCO_Laminar)   Interaction_Forces_gt2<tker,ftmode,VISCO_Laminar>   (t);
  else if(t.tvisco==VISCO_LaminarSPS)Interaction_Forces_gt2<tker,ftmode,VISCO_LaminarSPS>(t);
#endif
}
//==============================================================================
template<TpKernel tker> void Interaction_Forces_gt0(const StInterParmsg& t){
#ifdef FAST_COMPILATION
 if(t.ftmode!=FTMODE_None)throw "Extra FtMode options are disabled for FastCompilation...";
 Interaction_Forces_gt1<tker,FTMODE_None> (t);
#else
  if(t.ftmode==FTMODE_None)    Interaction_Forces_gt1<tker,FTMODE_None> (t);
  else if(t.ftmode==FTMODE_Sph)Interaction_Forces_gt1<tker,FTMODE_Sph>  (t);
  else if(t.ftmode==FTMODE_Ext)Interaction_Forces_gt1<tker,FTMODE_Ext>  (t);
#endif
}
//==============================================================================
void Interaction_Forces(const StInterParmsg& t){
#ifdef FAST_COMPILATION
  if(t.tkernel==KERNEL_Wendland)       Interaction_Forces_gt0<KERNEL_Wendland> (t);
  else throw "Only KERNEL_Wendland is compiled for FastCompilation...";
#else
  if(t.tkernel==KERNEL_Wendland)       Interaction_Forces_gt0<KERNEL_Wendland> (t);
 #ifndef DISABLE_KERNELS_EXTRA
  else if(t.tkernel==KERNEL_Cubic)     Interaction_Forces_gt0<KERNEL_Cubic   > (t);
 #endif
#endif
}

//------------------------------------------------------------------------------
/// Returns the corrected position after applying periodic conditions.
/// Devuelve la posicion corregida tras aplicar condiciones periodicas.
//------------------------------------------------------------------------------
__device__ float4 KerComputePosCell(const double3& ps,const double3& mapposmin
  ,float poscellsize)
{
  const double dx=ps.x-mapposmin.x;
  const double dy=ps.y-mapposmin.y;
  const double dz=ps.z-mapposmin.z;
  int cx=int(dx/poscellsize);
  int cy=int(dy/poscellsize);
  int cz=int(dz/poscellsize);
  cx=(cx>=0? cx: 0);
  cy=(cy>=0? cy: 0);
  cz=(cz>=0? cz: 0);
  const float px=float(dx-(double(poscellsize)*cx));
  const float py=float(dy-(double(poscellsize)*cy));
  const float pz=float(dz-(double(poscellsize)*cz));
  const float pw=__uint_as_float(PSCEL_Code(cx,cy,cz));
  return(make_float4(px,py,pz,pw));
}


//##############################################################################
//# Kernels for DEM interaction.
//# Kernels para interaccion DEM.
//##############################################################################
//------------------------------------------------------------------------------
/// DEM interaction of a particle with a set of particles. (Float-Float/Bound)
/// Realiza la interaccion DEM de una particula con un conjunto de ellas. (Float-Float/Bound)
//------------------------------------------------------------------------------
__device__ void KerInteractionForcesDemBox(
  bool boundp2,const unsigned& pini,const unsigned& pfin
  ,const float4* demdata,float dtforce
  ,const float4* poscell,const float4* velrho,const typecode* code
  ,const unsigned* idp,const float4& pscellp1,const float4& velp1
  ,typecode tavp1,float masstotp1,float ftmassp1,float taup1,float kfricp1
  ,float restitup1,float3& acep1,float& demdtp1)
{
  for(int p2=pini;p2<pfin;p2++){
    const typecode codep2=code[p2];
    if(CODE_IsNotFluid(codep2) && tavp1!=CODE_GetTypeAndValue(codep2)){
      const float4 pscellp2=poscell[p2];
      const float drx=pscellp1.x-pscellp2.x + CTE.poscellsize*(PSCEL_GetfX(pscellp1.w)-PSCEL_GetfX(pscellp2.w));
      const float dry=pscellp1.y-pscellp2.y + CTE.poscellsize*(PSCEL_GetfY(pscellp1.w)-PSCEL_GetfY(pscellp2.w));
      const float drz=pscellp1.z-pscellp2.z + CTE.poscellsize*(PSCEL_GetfZ(pscellp1.w)-PSCEL_GetfZ(pscellp2.w));
      const float rr2=drx*drx+dry*dry+drz*drz;
      if(rr2<=CTE.kernelsize2 && rr2>=ALMOSTZERO){
        const float rad=sqrt(rr2);

        //-Computes maximum value of demdt.
        float4 demdatap2=demdata[CODE_GetTypeAndValue(codep2)];
        const float nu_mass=(boundp2? masstotp1/2: masstotp1*demdatap2.x/(masstotp1+demdatap2.x)); //-With boundary takes the actual mass of floating 1. | Con boundary toma la propia masa del floating 1.
        const float kn=4.f/(3.f*(taup1+demdatap2.y))*sqrt(CTE.dp/4); //-Generalized rigidity - Lemieux 2008.
        const float dvx=velp1.x-velrho[p2].x, dvy=velp1.y-velrho[p2].y, dvz=velp1.z-velrho[p2].z; //vji
        const float nx=drx/rad, ny=dry/rad, nz=drz/rad; //-normal_ji             
        const float vn=dvx*nx+dvy*ny+dvz*nz; //-vji.nji    
        const float demvisc=0.2f/(3.21f*(pow(nu_mass/kn,0.4f)*pow(fabs(vn),-0.2f))/40.f);
        if(demdtp1<demvisc)demdtp1=demvisc;

        const float over_lap=1.0f*CTE.dp-rad; //-(ri+rj)-|dij|
        if(over_lap>0.0f){ //-Contact.
          //-Normal.
          const float eij=(restitup1+demdatap2.w)/2;
          const float gn=-(2.f*log(eij)*sqrt(nu_mass*kn))/(sqrt(float(PI)+log(eij)*log(eij))); //-Generalized damping - Cummins 2010.
          //const float gn=0.08f*sqrt(nu_mass*sqrt(CTE.dp/2)/((taup1+demdatap2.y)/2)); //-generalized damping - Lemieux 2008.
          const float rep=kn*pow(over_lap,1.5f);
          const float fn=rep-gn*pow(over_lap,0.25f)*vn;
          float acef=fn/ftmassp1; //-Divides by the mass of particle to obtain the acceleration.
          acep1.x+=(acef*nx); acep1.y+=(acef*ny); acep1.z+=(acef*nz); //-Force is applied in the normal between the particles.
          //-Tangencial.
          const float dvxt=dvx-vn*nx, dvyt=dvy-vn*ny, dvzt=dvz-vn*nz; //Vji_t
          const float vt=sqrt(dvxt*dvxt + dvyt*dvyt + dvzt*dvzt);
          const float tx=(vt!=0? dvxt/vt: 0), ty=(vt!=0? dvyt/vt: 0), tz=(vt!=0? dvzt/vt: 0); //-Tang vel unit vector.
          const float ft_elast=(kn*dtforce-gn)*vt/3.5f; //-Elastic frictional string -->  ft_elast=2*(kn*fdispl-gn*vt)/7; fdispl=dtforce*vt;
          const float kfric_ij=(kfricp1+demdatap2.z)/2;
          float ft=kfric_ij*fn*tanh(vt*8);  //-Coulomb.
          ft=(ft<ft_elast? ft: ft_elast);   //-Not above yield criteria, visco-elastic model.
          acef=ft/ftmassp1; //-Divides by the mass of particle to obtain the acceleration.
          acep1.x+=(acef*tx); acep1.y+=(acef*ty); acep1.z+=(acef*tz);
        }
      }
    }
  }
}

//------------------------------------------------------------------------------
/// Interaction between particles. Fluid/Float-Fluid/Float or Fluid/Float-Bound.
/// Includes artificial/laminar viscosity and normal/DEM floating bodies.
///
/// Realiza interaccion entre particulas. Fluid/Float-Fluid/Float or Fluid/Float-Bound
/// Incluye visco artificial/laminar y floatings normales/dem.
//------------------------------------------------------------------------------
__global__ void KerInteractionForcesDem(unsigned nfloat
  ,int scelldiv,int4 nc,int3 cellzero,const int2* begincell,unsigned cellfluid
  ,const unsigned* dcell,const unsigned* ftridp,const float4* demdata
  ,const float* ftomassp,float dtforce,const float4* poscell,const float4* velrho
  ,const typecode* code,const unsigned* idp,float* viscdt,float3* ace)
{
  const unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<nfloat){
    const unsigned p1=ftridp[p]; //-Number of particle.
    if(p1!=UINT_MAX){
      float demdtp1=0;
      float3 acep1=make_float3(0,0,0);

      //-Obtains basic data of particle p1.
      const float4 pscellp1=poscell[p1];
      const float4 velp1=velrho[p1];
      const typecode cod=code[p1];
      const typecode tavp1=CODE_GetTypeAndValue(cod);
      const float4 rdata=demdata[tavp1];
      const float masstotp1=rdata.x;
      const float taup1=rdata.y;
      const float kfricp1=rdata.z;
      const float restitup1=rdata.w;
      const float ftmassp1=ftomassp[CODE_GetTypeValue(cod)];

      //-Obtains neighborhood search limits.
      int ini1,fin1,ini2,fin2,ini3,fin3;
      cunsearch::InitCte(dcell[p1],scelldiv,nc,cellzero,ini1,fin1,ini2,fin2,ini3,fin3);

      //-Interaction with boundaries.
      for(int c3=ini3;c3<fin3;c3+=nc.w)for(int c2=ini2;c2<fin2;c2+=nc.x){
        unsigned pini,pfin=0;  cunsearch::ParticleRange(c2,c3,ini1,fin1,begincell,pini,pfin);
        if(pfin)KerInteractionForcesDemBox (true ,pini,pfin,demdata,dtforce,poscell,velrho,code,idp
          ,pscellp1,velp1,tavp1,masstotp1,ftmassp1,taup1,kfricp1,restitup1,acep1,demdtp1);
      }

      //-Interaction with fluids.
      ini3+=cellfluid; fin3+=cellfluid;
      for(int c3=ini3;c3<fin3;c3+=nc.w)for(int c2=ini2;c2<fin2;c2+=nc.x){
        unsigned pini,pfin=0;  cunsearch::ParticleRange(c2,c3,ini1,fin1,begincell,pini,pfin);
        if(pfin)KerInteractionForcesDemBox (false,pini,pfin,demdata,dtforce,poscell,velrho,code,idp
          ,pscellp1,velp1,tavp1,masstotp1,ftmassp1,taup1,kfricp1,restitup1,acep1,demdtp1);
      }

      //-Stores results.
      if(acep1.x || acep1.y || acep1.z || demdtp1){
        float3 r=ace[p1]; r.x+=acep1.x; r.y+=acep1.y; r.z+=acep1.z; ace[p1]=r;
        if(viscdt[p1]<demdtp1)viscdt[p1]=demdtp1;
      }
    }
  }
}

#ifndef DISABLE_BSMODES
//==============================================================================
/// Collects kernel information.
//==============================================================================
void Interaction_ForcesDemT_KerInfo(StKerInfo* kerinfo)
{
#if CUDART_VERSION >= 6050
  {
    typedef void (*fun_ptr)(unsigned,int,int4,int3,const int2*,unsigned,const unsigned*,const unsigned*,const float4*,const float*,float,const float4*,const float4*,const typecode*,const unsigned*,float*,float3*);
    fun_ptr ptr=&KerInteractionForcesDem;
    int qblocksize=0,mingridsize=0;
    hipOccupancyMaxPotentialBlockSize(&mingridsize,&qblocksize,(void*)ptr,0,0);
    struct hipFuncAttributes attr;
    hipFuncGetAttributes(&attr,reinterpret_cast<const void*>((void*)ptr));
    kerinfo->forcesdem_bs=qblocksize;
    kerinfo->forcesdem_rg=attr.numRegs;
    kerinfo->forcesdem_bsmax=attr.maxThreadsPerBlock;
    //printf(">> KerInteractionForcesDem  blocksize:%u (%u)\n",qblocksize,0);
  }
  fcuda::Check_CudaErroorFun("Error collecting kernel information.");
#endif
}
#endif

//==============================================================================
/// Interaction for the force computation.
/// Interaccion para el calculo de fuerzas.
//==============================================================================
void Interaction_ForcesDem(unsigned bsize,unsigned nfloat
  ,const StDivDataGpu& dvd,const unsigned* dcell
  ,const unsigned* ftridp,const float4* demdata,const float* ftomassp
  ,float dtforce,const float4* poscell,const float4* velrho
  ,const typecode* code,const unsigned* idp,float* viscdt,float3* ace
  ,StKerInfo* kerinfo,hipStream_t stm)
{
  const int2* beginendcell=dvd.beginendcell;
  //-Collects kernel information.
#ifndef DISABLE_BSMODES
  if(kerinfo){
    Interaction_ForcesDemT_KerInfo(kerinfo);
    return;
  }
#endif
  //-Interaction Fluid-Fluid & Fluid-Bound.
  if(nfloat){
    dim3 sgrid=GetSimpleGridSize(nfloat,bsize);
    KerInteractionForcesDem <<<sgrid,bsize,0,stm>>> (nfloat
      ,dvd.scelldiv,dvd.nc,dvd.cellzero,beginendcell,dvd.cellfluid,dcell
      ,ftridp,demdata,ftomassp,dtforce,poscell,velrho,code,idp,viscdt,ace);
  }
}


//##############################################################################
//# Kernels for Laminar+SPS.
//##############################################################################
//------------------------------------------------------------------------------
/// Computes sub-particle stress tensor (Tau) for SPS turbulence model.
//------------------------------------------------------------------------------
__global__ void KerComputeSpsTau(unsigned n,unsigned pini,float smag,float blin
  ,const float4* velrho,const float2* sps2strain,float2* tau_rho2)
{
  unsigned p=blockIdx.x*blockDim.x + threadIdx.x; 
  if(p<n){
    const unsigned p1=p+pini;
    float2 rr=sps2strain[p1*3];   const float two_strain_xx=rr.x,two_strain_xy=rr.y;
           rr=sps2strain[p1*3+1]; const float two_strain_xz=rr.x,two_strain_yy=rr.y;
           rr=sps2strain[p1*3+2]; const float two_strain_yz=rr.x,two_strain_zz=rr.y;
    const float pow1=two_strain_xx*two_strain_xx
                   + two_strain_yy*two_strain_yy
                   + two_strain_zz*two_strain_zz;
    const float prr= two_strain_xy*two_strain_xy
                   + two_strain_xz*two_strain_xz
                   + two_strain_yz*two_strain_yz + pow1+pow1;
    const float visc_sps=smag*sqrt(prr);
    const float div_u=two_strain_xx+two_strain_yy+two_strain_zz;
    const float sps_k=visc_sps*div_u; //-Factor 2/3 is included in smag constant.
    const float sps_blin=blin*prr;
    const float sumsps=-(sps_k+sps_blin);
    const float twovisc_sps=(visc_sps+visc_sps);
    float one_rho=1.0f/velrho[p1].w;
    //-Computes new values of tau/rho^2.
    const float tau_xx=one_rho*(twovisc_sps*two_strain_xx +sumsps);
    const float tau_xy=one_rho*(visc_sps   *two_strain_xy);
    tau_rho2[p1*3]=make_float2(tau_xx,tau_xy);
    const float tau_xz=one_rho*(visc_sps   *two_strain_xz);
    const float tau_yy=one_rho*(twovisc_sps*two_strain_yy +sumsps);
    tau_rho2[p1*3+1]=make_float2(tau_xz,tau_yy);
    const float tau_yz=one_rho*(visc_sps   *two_strain_yz);
    const float tau_zz=one_rho*(twovisc_sps*two_strain_zz +sumsps);
    tau_rho2[p1*3+2]=make_float2(tau_yz,tau_zz);
  }
}

//==============================================================================
/// Computes sub-particle stress tensor divided by rho^2 (tau/rho^2) for SPS 
/// turbulence model.   
//==============================================================================
void ComputeSpsTau(unsigned np,unsigned npb,float smag,float blin
  ,const float4* velrho,const tsymatrix3f* sps2strain,tsymatrix3f* tau_rho2
  ,hipStream_t stm)
{
  const unsigned npf=np-npb;
  if(npf){
    dim3 sgridf=GetSimpleGridSize(npf,SPHBSIZE);
    KerComputeSpsTau <<<sgridf,SPHBSIZE,0,stm>>> (npf,npb,smag,blin,velrho
      ,(const float2*)sps2strain,(float2*)tau_rho2);
  }
}


//##############################################################################
//# Kernels for Delta-SPH.
//# Kernels para Delta-SPH.
//##############################################################################
//------------------------------------------------------------------------------
/// Adds value of delta[] to ar[] provided it is not FLT_MAX.
/// Anhade valor de delta[] a ar[] siempre que no sea FLT_MAX.
//------------------------------------------------------------------------------
__global__ void KerAddDelta(unsigned n,const float* delta,float* ar)
{
  unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    float rdelta=delta[p];
    if(rdelta!=FLT_MAX)ar[p]+=rdelta;
  }
}

//==============================================================================
/// Adds value of delta[] to ar[] provided it is not FLT_MAX.
/// Anhade valor de delta[] a ar[] siempre que no sea FLT_MAX.
//==============================================================================
void AddDelta(unsigned n,const float* delta,float* ar,hipStream_t stm){
  if(n){
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    KerAddDelta <<<sgrid,SPHBSIZE,0,stm>>> (n,delta,ar);
  }
}


//##############################################################################
//# Kernels para ComputeStep (position)
//# Kernels for ComputeStep (position)
//##############################################################################
//------------------------------------------------------------------------------
/// Updates particle position according to displacement.
/// Actualizacion de posicion de particulas segun desplazamiento.
//------------------------------------------------------------------------------
template<bool periactive,bool floatings> __global__ void KerComputeStepPos(
  unsigned n,unsigned pini,const double2* movxy,const double* movz
  ,double2* posxy,double* posz,unsigned* dcell,typecode* code)
{
  unsigned pt=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(pt<n){
    unsigned p=pt+pini;
    const typecode rcode=code[p];
    const bool outrhop=CODE_IsOutRho(rcode);
    const bool fluid=(!floatings || CODE_IsFluid(rcode));
    const bool normal=(!periactive || outrhop || CODE_IsNormal(rcode));
    if(normal && fluid){ //-Does not apply to periodic or floating particles. | No se aplica a particulas periodicas o floating.
      const double2 rmovxy=movxy[p];
      KerUpdatePos<periactive>(posxy[p],posz[p],rmovxy.x,rmovxy.y,movz[p],outrhop,p,posxy,posz,dcell,code);
    }
    //-In case of floating maintains the original position.
    //-En caso de floating mantiene la posicion original.
  }
}

//==============================================================================
/// Updates particle position according to displacement.
/// Actualizacion de posicion de particulas segun desplazamiento.
//==============================================================================
void ComputeStepPos(byte periactive,bool floatings,unsigned np,unsigned npb
  ,const double2* movxy,const double* movz
  ,double2* posxy,double* posz,unsigned* dcell,typecode* code)
{
  const unsigned pini=npb;
  const unsigned npf=np-pini;
  if(npf){
    dim3 sgrid=GetSimpleGridSize(npf,SPHBSIZE);
    if(periactive){ const bool peri=true;
      if(floatings)KerComputeStepPos<peri,true>  <<<sgrid,SPHBSIZE>>> (npf,pini,movxy,movz,posxy,posz,dcell,code);
      else         KerComputeStepPos<peri,false> <<<sgrid,SPHBSIZE>>> (npf,pini,movxy,movz,posxy,posz,dcell,code);
    }
    else{ const bool peri=false;
      if(floatings)KerComputeStepPos<peri,true>  <<<sgrid,SPHBSIZE>>> (npf,pini,movxy,movz,posxy,posz,dcell,code);
      else         KerComputeStepPos<peri,false> <<<sgrid,SPHBSIZE>>> (npf,pini,movxy,movz,posxy,posz,dcell,code);
    }
  }
}

//------------------------------------------------------------------------------
/// Updates particle position according to displacement.
/// Actualizacion de posicion de particulas segun desplazamiento.
//------------------------------------------------------------------------------
template<bool periactive,bool floatings> __global__ void KerComputeStepPos2(
  unsigned n,unsigned pini,const double2* posxypre,const double* poszpre
  ,const double2* movxy,const double* movz,double2* posxy,double* posz
  ,unsigned* dcell,typecode* code)
{
  unsigned pt=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(pt<n){
    unsigned p=pt+pini;
    const typecode rcode=code[p];
    const bool outrhop=CODE_IsOutRho(rcode);
    const bool fluid=(!floatings || CODE_IsFluid(rcode));
    const bool normal=(!periactive || outrhop || CODE_IsNormal(rcode));
    if(normal){//-Does not apply to periodic particles. | No se aplica a particulas periodicas
      if(fluid){//-Only applied for fluid displacement. | Solo se aplica desplazamiento al fluido.
        const double2 rmovxy=movxy[p];
        KerUpdatePos<periactive>(posxypre[p],poszpre[p],rmovxy.x,rmovxy.y,movz[p],outrhop,p,posxy,posz,dcell,code);

      }
      else{ //-Copy position of floating particles.
        posxy[p]=posxypre[p];
        posz[p]=poszpre[p];
      }
    }
  }
}

//==============================================================================
/// Updates particle position according to displacement.
/// Actualizacion de posicion de particulas segun desplazamiento.
//==============================================================================
void ComputeStepPos2(byte periactive,bool floatings,unsigned np,unsigned npb
  ,const double2* posxypre,const double* poszpre,const double2* movxy
  ,const double* movz,double2* posxy,double* posz,unsigned* dcell,typecode* code)
{
  const unsigned pini=npb;
  const unsigned npf=np-pini;
  if(npf){
    dim3 sgrid=GetSimpleGridSize(npf,SPHBSIZE);
    if(periactive){ const bool peri=true;
      if(floatings)KerComputeStepPos2<peri,true>  <<<sgrid,SPHBSIZE>>> (npf,pini,posxypre,poszpre,movxy,movz,posxy,posz,dcell,code);
      else         KerComputeStepPos2<peri,false> <<<sgrid,SPHBSIZE>>> (npf,pini,posxypre,poszpre,movxy,movz,posxy,posz,dcell,code);
    }
    else{ const bool peri=false;
      if(floatings)KerComputeStepPos2<peri,true>  <<<sgrid,SPHBSIZE>>> (npf,pini,posxypre,poszpre,movxy,movz,posxy,posz,dcell,code);
      else         KerComputeStepPos2<peri,false> <<<sgrid,SPHBSIZE>>> (npf,pini,posxypre,poszpre,movxy,movz,posxy,posz,dcell,code);
    }
  }
}



//##############################################################################
//# Kernels for motion.
//# Kernels para Motion
//##############################################################################
//------------------------------------------------------------------------------
/// Computes for a range of particles, their position according to idp[].
/// Calcula para un rango de particulas calcula su posicion segun idp[].
//------------------------------------------------------------------------------
__global__ void KerCalcRidp(unsigned n,unsigned ini,unsigned idini,unsigned idfin
  ,const typecode* code,const unsigned* idp,unsigned* ridp)
{
  unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    p+=ini;
    unsigned id=idp[p];
    if(idini<=id && id<idfin){
      if(CODE_IsNormal(code[p]))ridp[id-idini]=p;
    }
  }
}
//------------------------------------------------------------------------------
__global__ void KerCalcRidp(unsigned n,unsigned ini,unsigned idini,unsigned idfin
  ,const unsigned* idp,unsigned* ridp)
{
  unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    p+=ini;
    const unsigned id=idp[p];
    if(idini<=id && id<idfin)ridp[id-idini]=p;
  }
}

//==============================================================================
/// Calculate particle position according to idp[]. When it does not find UINT_MAX.
/// When periactive is false it means there are no duplicate particles (periodic)
/// and all are CODE_NORMAL.
///
/// Calcula posicion de particulas segun idp[]. Cuando no la encuentra es UINT_MAX.
/// Cuando periactive es False sumpone que no hay particulas duplicadas (periodicas)
/// y todas son CODE_NORMAL.
//==============================================================================
void CalcRidp(bool periactive,unsigned np,unsigned pini,unsigned idini
  ,unsigned idfin,const typecode* code,const unsigned* idp,unsigned* ridp
  ,hipStream_t stm)
{
  //-Assigns values UINT_MAX
  const unsigned nsel=idfin-idini;
  hipMemset(ridp,255,sizeof(unsigned)*nsel); 
  //-Computes position according to id. | Calcula posicion segun id.
  if(np){
    dim3 sgrid=GetSimpleGridSize(np,SPHBSIZE);
    if(periactive)KerCalcRidp <<<sgrid,SPHBSIZE,0,stm>>> (np,pini,idini,idfin,code,idp,ridp);
    else          KerCalcRidp <<<sgrid,SPHBSIZE,0,stm>>> (np,pini,idini,idfin,idp,ridp);
  }
}


//------------------------------------------------------------------------------
/// Load current reference position data from particle data.
//------------------------------------------------------------------------------
__global__ void KerLoadPosRef(unsigned pscount,unsigned casenfixed,unsigned np
  ,const double2* posxy,const double* posz,const unsigned* ridpmot
  ,const unsigned* idpref,double3* posref)
{
  unsigned cp=blockIdx.x*blockDim.x + threadIdx.x;
  if(cp<pscount){
    const unsigned iref=idpref[cp]-casenfixed;
    const unsigned p=ridpmot[iref];
    if(p<np){
      const double2 rxy=posxy[p];
      const double rz=posz[p];
      posref[cp]=make_double3(rxy.x,rxy.y,rz);
    }
    else posref[cp]=make_double3(DBL_MAX,DBL_MAX,DBL_MAX);
  }
}

//==============================================================================
/// Load current reference position data from particle data.
//==============================================================================
void LoadPosRef(unsigned pscount,unsigned casenfixed,unsigned np
  ,const double2* posxy,const double* posz,const unsigned* ridpmot
  ,const unsigned* idpref,double3* posref)
{
  //-Computes position according to id. | Calcula posicion segun id.
  if(pscount){
    const dim3 sgrid=GetSimpleGridSize(pscount,SPHBSIZE);
    KerLoadPosRef <<<sgrid,SPHBSIZE>>> (pscount,casenfixed,np,posxy,posz
      ,ridpmot,idpref,posref);
  }
}


//------------------------------------------------------------------------------
/// Applies a linear movement to a set of particles.
/// Aplica un movimiento lineal a un conjunto de particulas.
//------------------------------------------------------------------------------
template<bool periactive> __global__ void KerMoveLinBound(unsigned n,unsigned ini
  ,double3 mvpos,float3 mvvel,const unsigned* ridpmot,double2* posxy,double* posz
  ,unsigned* dcell,float4* velrho,typecode* code)
{
  unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    int pid=ridpmot[p+ini];
    if(pid>=0){
      //-Computes displacement and updates position.
      KerUpdatePos<periactive>(posxy[pid],posz[pid],mvpos.x,mvpos.y,mvpos.z,false,pid,posxy,posz,dcell,code);
      //-Computes velocity.
      velrho[pid]=make_float4(mvvel.x,mvvel.y,mvvel.z,velrho[pid].w);
    }
  }
}

//==============================================================================
/// Applies a linear movement to a set of particles.
/// Aplica un movimiento lineal a un conjunto de particulas.
//==============================================================================
void MoveLinBound(byte periactive,unsigned np,unsigned ini,tdouble3 mvpos
  ,tfloat3 mvvel,const unsigned* ridpmot,double2* posxy,double* posz
  ,unsigned* dcell,float4* velrho,typecode* code)
{
  dim3 sgrid=GetSimpleGridSize(np,SPHBSIZE);
  if(periactive)KerMoveLinBound<true>  <<<sgrid,SPHBSIZE>>> (np,ini,Double3(mvpos),Float3(mvvel),ridpmot,posxy,posz,dcell,velrho,code);
  else          KerMoveLinBound<false> <<<sgrid,SPHBSIZE>>> (np,ini,Double3(mvpos),Float3(mvvel),ridpmot,posxy,posz,dcell,velrho,code);
}


//------------------------------------------------------------------------------
/// Applies a matrix movement to a set of particles.
/// Aplica un movimiento matricial a un conjunto de particulas.
//------------------------------------------------------------------------------
template<bool periactive,bool simulate2d> __global__ void KerMoveMatBound(
  unsigned n,unsigned ini,tmatrix4d m,double dt,const unsigned* ridmot
  ,double2* posxy,double* posz,unsigned* dcell,float4* velrho,typecode* code
  ,float3* boundnor)
{
  unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    int pid=ridmot[p+ini];
    if(pid>=0){
      double2 rxy=posxy[pid];
      double3 rpos=make_double3(rxy.x,rxy.y,posz[pid]);
      //-Computes new position.
      double3 rpos2;
      rpos2.x= rpos.x*m.a11 + rpos.y*m.a12 + rpos.z*m.a13 + m.a14;
      rpos2.y= rpos.x*m.a21 + rpos.y*m.a22 + rpos.z*m.a23 + m.a24;
      rpos2.z= rpos.x*m.a31 + rpos.y*m.a32 + rpos.z*m.a33 + m.a34;
      if(simulate2d)rpos2.y=rpos.y;
      //-Computes displacement and updates position.
      const double dx=rpos2.x-rpos.x;
      const double dy=rpos2.y-rpos.y;
      const double dz=rpos2.z-rpos.z;
      KerUpdatePos<periactive>(make_double2(rpos.x,rpos.y),rpos.z,dx,dy,dz,false,pid,posxy,posz,dcell,code);
      //-Computes velocity.
      velrho[pid]=make_float4(float(dx/dt),float(dy/dt),float(dz/dt),velrho[pid].w);
      //-Computes normal.
      if(boundnor){
        const float3 bnor=boundnor[pid];
        const double3 gs=make_double3(rpos.x+bnor.x,rpos.y+bnor.y,rpos.z+bnor.z);
        const double gs2x=gs.x*m.a11 + gs.y*m.a12 + gs.z*m.a13 + m.a14;
        const double gs2y=gs.x*m.a21 + gs.y*m.a22 + gs.z*m.a23 + m.a24;
        const double gs2z=gs.x*m.a31 + gs.y*m.a32 + gs.z*m.a33 + m.a34;
        boundnor[pid]=make_float3(gs2x-rpos2.x,gs2y-rpos2.y,gs2z-rpos2.z);
      }
    }
  }
}

//==============================================================================
/// Applies a matrix movement to a set of particles.
/// Aplica un movimiento matricial a un conjunto de particulas.
//==============================================================================
void MoveMatBound(byte periactive,bool simulate2d,unsigned np,unsigned ini
  ,tmatrix4d m,double dt,const unsigned* ridpmot,double2* posxy,double* posz
  ,unsigned* dcell,float4* velrho,typecode* code,float3* boundnor)
{
  dim3 sgrid=GetSimpleGridSize(np,SPHBSIZE);
  if(periactive){ const bool peri=true;
    if(simulate2d)KerMoveMatBound<peri,true>  <<<sgrid,SPHBSIZE>>> (np,ini,m,dt,ridpmot,posxy,posz,dcell,velrho,code,boundnor);
    else          KerMoveMatBound<peri,false> <<<sgrid,SPHBSIZE>>> (np,ini,m,dt,ridpmot,posxy,posz,dcell,velrho,code,boundnor);
  }
  else{ const bool peri=false;
    if(simulate2d)KerMoveMatBound<peri,true>  <<<sgrid,SPHBSIZE>>> (np,ini,m,dt,ridpmot,posxy,posz,dcell,velrho,code,boundnor);
    else          KerMoveMatBound<peri,false> <<<sgrid,SPHBSIZE>>> (np,ini,m,dt,ridpmot,posxy,posz,dcell,velrho,code,boundnor);
  }
}


//------------------------------------------------------------------------------
/// Applies a matrix movement to a set of particles.
/// Aplica un movimiento matricial a un conjunto de particulas.
//------------------------------------------------------------------------------
__global__ void KerFtNormalsUpdate(unsigned n,unsigned fpini
  ,double a11,double a12,double a13,double a21,double a22,double a23
  ,double a31,double a32,double a33
  ,const unsigned* ridpmot,float3* boundnor)
{
  const unsigned fp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of floating particle.
  if(fp<n){
    const unsigned p=ridpmot[fp+fpini];
    if(p!=UINT_MAX){
      float3 rnor=boundnor[p];
      const double nx=rnor.x;
      const double ny=rnor.y;
      const double nz=rnor.z;
      rnor.x=float(a11*nx + a12*ny + a13*nz);
      rnor.y=float(a21*nx + a22*ny + a23*nz);
      rnor.z=float(a31*nx + a32*ny + a33*nz);
      boundnor[p]=rnor;
    }
  }
}

//==============================================================================
/// Applies a matrix movement to a set of particles.
/// Aplica un movimiento matricial a un conjunto de particulas.
//==============================================================================
void FtNormalsUpdate(unsigned np,unsigned ini,tmatrix4d m,const unsigned* ridpmot
  ,float3* boundnor)
{
  dim3 sgrid=GetSimpleGridSize(np,SPHBSIZE);
  if(np)KerFtNormalsUpdate <<<sgrid,SPHBSIZE>>> (np,ini,m.a11,m.a12,m.a13
    ,m.a21,m.a22,m.a23,m.a31,m.a32,m.a33,ridpmot,boundnor);
}



//##############################################################################
//# Kernels for MLPistons motion.
//##############################################################################
//------------------------------------------------------------------------------
/// Applies movement and velocity of piston 1D to a group of particles.
/// Aplica movimiento y velocidad de piston 1D a conjunto de particulas.
//------------------------------------------------------------------------------
template<byte periactive> __global__ void KerMovePiston1d(unsigned n,unsigned idini
  ,double dp,double poszmin,unsigned poszcount,const byte* pistonid
  ,const double* movx,const double* velx,const unsigned* ridpmv,double2* posxy
  ,double* posz,unsigned* dcell,float4* velrho,typecode* code)
{
  unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle
  if(p<n){
    const unsigned id=p+idini;
    int pid=ridpmv[id];
    if(pid>=0){
      const unsigned pisid=pistonid[CODE_GetTypeValue(code[pid])];
      if(pisid<255){
        const double2 rpxy=posxy[pid];
        const double rpz=posz[pid];
        const unsigned cz=unsigned((rpz-poszmin)/dp);
        const double rmovx=(cz<poszcount? movx[pisid*poszcount+cz]: 0);
        const float rvelx=float(cz<poszcount? velx[pisid*poszcount+cz]: 0);
        //-Updates position.
        KerUpdatePos<periactive>(rpxy,rpz,rmovx,0,0,false,pid,posxy,posz,dcell,code);
        //-Updates velocity.
        velrho[pid].x=rvelx;
      }
    }
  }
}

//==============================================================================
/// Applies movement and velocity of piston 1D to a group of particles.
/// Aplica movimiento y velocidad de piston 1D a conjunto de particulas.
//==============================================================================
void MovePiston1d(bool periactive,unsigned np,unsigned idini
  ,double dp,double poszmin,unsigned poszcount,const byte* pistonid
  ,const double* movx,const double* velx,const unsigned* ridpmv,double2* posxy
  ,double* posz,unsigned* dcell,float4* velrho,typecode* code)
{
  if(np){
    dim3 sgrid=GetSimpleGridSize(np,SPHBSIZE);
    if(periactive)KerMovePiston1d<true>  <<<sgrid,SPHBSIZE>>> (np,idini,dp,poszmin,poszcount,pistonid,movx,velx,ridpmv,posxy,posz,dcell,velrho,code);
    else          KerMovePiston1d<false> <<<sgrid,SPHBSIZE>>> (np,idini,dp,poszmin,poszcount,pistonid,movx,velx,ridpmv,posxy,posz,dcell,velrho,code);
  }
}

//------------------------------------------------------------------------------
/// Applies movement and velocity of piston 2D to a group of particles.
/// Aplica movimiento y velocidad de piston 2D a conjunto de particulas.
//------------------------------------------------------------------------------
template<byte periactive> __global__ void KerMovePiston2d(unsigned n,unsigned idini
  ,double dp,double posymin,double poszmin,unsigned poszcount,const double* movx
  ,const double* velx,const unsigned* ridpmot,double2* posxy,double* posz
  ,unsigned* dcell,float4* velrho,typecode* code)
{
  unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle
  if(p<n){
    const unsigned id=p+idini;
    int pid=ridpmot[id];
    if(pid>=0){
      const double2 rpxy=posxy[pid];
      const double rpz=posz[pid];
      const unsigned cy=unsigned((rpxy.y-posymin)/dp);
      const unsigned cz=unsigned((rpz-poszmin)/dp);
      const double rmovx=(cz<poszcount? movx[cy*poszcount+cz]: 0);
      const float rvelx=float(cz<poszcount? velx[cy*poszcount+cz]: 0);
      //-Actualiza posicion.
      KerUpdatePos<periactive>(rpxy,rpz,rmovx,0,0,false,pid,posxy,posz,dcell,code);
      //-Actualiza velocidad.
      velrho[pid].x=rvelx;
    }
  }
}

//==============================================================================
/// Applies movement and velocity of piston 2D to a group of particles.
/// Aplica movimiento y velocidad de piston 2D a conjunto de particulas.
//==============================================================================
void MovePiston2d(bool periactive,unsigned np,unsigned idini,double dp
  ,double posymin,double poszmin,unsigned poszcount,const double* movx
  ,const double* velx,const unsigned* ridpmot,double2* posxy,double* posz
  ,unsigned* dcell,float4* velrho,typecode* code)
{
  if(np){
    dim3 sgrid=GetSimpleGridSize(np,SPHBSIZE);
    if(periactive)KerMovePiston2d<true>  <<<sgrid,SPHBSIZE>>> (np,idini,dp,posymin,poszmin,poszcount,movx,velx,ridpmot,posxy,posz,dcell,velrho,code);
    else          KerMovePiston2d<false> <<<sgrid,SPHBSIZE>>> (np,idini,dp,posymin,poszmin,poszcount,movx,velx,ridpmot,posxy,posz,dcell,velrho,code);
  }
}


//##############################################################################
//# Kernels for Floating bodies.
//##############################################################################
//==============================================================================
/// Computes distance between floating and centre particles according to periodic conditions.
/// Calcula distancia entre pariculas floating y centro segun condiciones periodicas.
//==============================================================================
template<bool periactive> __device__ void KerFtPeriodicDist(double px,double py
  ,double pz,double cenx,double ceny,double cenz,float radius,float& dx
  ,float& dy,float& dz)
{
  if(periactive){
    double ddx=px-cenx;
    double ddy=py-ceny;
    double ddz=pz-cenz;
    const unsigned peri=CTE.periactive;
    if(PERI_AxisX(peri) && fabs(ddx)>radius){
      if(ddx>0){ ddx+=CTE.xperincx; ddy+=CTE.xperincy; ddz+=CTE.xperincz; }
      else{      ddx-=CTE.xperincx; ddy-=CTE.xperincy; ddz-=CTE.xperincz; }
    }
    if(PERI_AxisY(peri) && fabs(ddy)>radius){
      if(ddy>0){ ddx+=CTE.yperincx; ddy+=CTE.yperincy; ddz+=CTE.yperincz; }
      else{      ddx-=CTE.yperincx; ddy-=CTE.yperincy; ddz-=CTE.yperincz; }
    }
    if(PERI_AxisZ(peri) && fabs(ddz)>radius){
      if(ddz>0){ ddx+=CTE.zperincx; ddy+=CTE.zperincy; ddz+=CTE.zperincz; }
      else{      ddx-=CTE.zperincx; ddy-=CTE.zperincy; ddz-=CTE.zperincz; }
    }
    dx=float(ddx);
    dy=float(ddy);
    dz=float(ddz);
  }
  else{
    dx=float(px-cenx);
    dy=float(py-ceny);
    dz=float(pz-cenz);
  }
}

//------------------------------------------------------------------------------
/// Calculate summation: face, fomegaace in ftoforcessum[].
/// Calcula suma de face y fomegaace a partir de particulas floating en ftoforcessum[].
//------------------------------------------------------------------------------
template<bool periactive> __global__ void KerFtPartsSumAce( //ftodatp={pini-CaseNfixed,np,radius,massp}
  const float4* ftodatp,const double3* ftocenter,const unsigned* ridpmot
  ,const double2* posxy,const double* posz,const float3* ace
  ,float3* ftoacelinang)
{
  extern __shared__ float racelinx[];
  float* raceliny=racelinx+blockDim.x;
  float* racelinz=raceliny+blockDim.x;
  float* raceangx=racelinz+blockDim.x;
  float* raceangy=raceangx+blockDim.x;
  float* raceangz=raceangy+blockDim.x;

  const unsigned tid=threadIdx.x;  //-Thread number.
  const unsigned cf=blockIdx.x;    //-Floating number.
  
  //-Loads floating data.
  const float4 rfdata=ftodatp[cf];
  const unsigned fpini=(unsigned)__float_as_int(rfdata.x);
  const unsigned fnp=(unsigned)__float_as_int(rfdata.y);
  const float fradius=rfdata.z;
  //const float fmassp=rfdata.w;
  const double3 rcenter=ftocenter[cf];

  //-Initialises shared memory to zero.
  const unsigned ntid=(fnp<blockDim.x? fnp: blockDim.x); //-Number of used threads. | Numero de threads utilizados.
  if(tid<ntid){
    racelinx[tid]=raceliny[tid]=racelinz[tid]=0;
    raceangx[tid]=raceangy[tid]=raceangz[tid]=0;
  }

  //-Computes data in shared memory. | Calcula datos en memoria shared.
  const unsigned nfor=unsigned((fnp+blockDim.x-1)/blockDim.x);
  for(unsigned cfor=0;cfor<nfor;cfor++){
    unsigned p=cfor*blockDim.x+tid;
    if(p<fnp){
      const unsigned rp=ridpmot[p+fpini];
      if(rp!=UINT_MAX){
        const float3 acep=ace[rp];
        racelinx[tid]+=acep.x; raceliny[tid]+=acep.y; racelinz[tid]+=acep.z;
        //-Computes distance from the centre. | Calcula distancia al centro.
        const double2 rposxy=posxy[rp];
        float dx,dy,dz;
        KerFtPeriodicDist<periactive>(rposxy.x,rposxy.y,posz[rp],rcenter.x,rcenter.y,rcenter.z,fradius,dx,dy,dz);
        //-Computes omegaace.
        raceangx[tid]+=(acep.z*dy - acep.y*dz);
        raceangy[tid]+=(acep.x*dz - acep.z*dx);
        raceangz[tid]+=(acep.y*dx - acep.x*dy);
      }
    }
  }

  //-Reduces data in shared memory and stores results.
  //-Reduce datos de memoria shared y guarda resultados.
  __syncthreads();
  if(!tid){
    float3 acelin=make_float3(0,0,0);
    float3 aceang=make_float3(0,0,0);
    for(unsigned c=0;c<ntid;c++){
      acelin.x+=racelinx[c];  acelin.y+=raceliny[c];  acelin.z+=racelinz[c];
      aceang.x+=raceangx[c];  aceang.y+=raceangy[c];  aceang.z+=raceangz[c];
    }
    //-Stores results in ftoacelinang[].
    const unsigned cf2=cf*2;
    ftoacelinang[cf2  ]=acelin;
    ftoacelinang[cf2+1]=aceang;
  }
}

//==============================================================================
/// Calculate summation: face, fomegaace in ftoforcessum[].
/// Calcula suma de face y fomegaace a partir de particulas floating en ftoforcessum[].
//==============================================================================
void FtPartsSumAce(bool periactive,unsigned ftcount
  ,const float4* ftodatp,const double3* ftocenter,const unsigned* ridpmot
  ,const double2* posxy,const double* posz,const float3* ace
  ,float3* ftoacelinang)
{
  if(ftcount){
    const unsigned bsize=256;
    const unsigned smem=sizeof(float)*(3+3)*bsize;
    dim3 sgrid=GetSimpleGridSize(ftcount*bsize,bsize);
    if(periactive)KerFtPartsSumAce<true>  <<<sgrid,bsize,smem>>> (ftodatp,ftocenter,ridpmot,posxy,posz,ace,ftoacelinang);
    else          KerFtPartsSumAce<false> <<<sgrid,bsize,smem>>> (ftodatp,ftocenter,ridpmot,posxy,posz,ace,ftoacelinang);
  }
}

//------------------------------------------------------------------------------
/// Updates information and particles of floating bodies.
//------------------------------------------------------------------------------
template<bool periactive,bool mdbc2> __global__ void KerFtPartsUpdate(double dt
  ,bool updatenormals,unsigned np,unsigned fpini,float fradius,tmatrix4d mat
  ,float3 fvel,float3 fomega,double3 fcenter
  ,const unsigned* ridpmot,double2* posxy,double* posz,float4* velrho
  ,unsigned* dcell,typecode* code,float3* boundnor,float3* motionvel
  ,float3* motionace)
{
  const unsigned fp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(fp<np){
    const int p=ridpmot[fp+fpini];
    if(p>=0){
      float4 vr=velrho[p];
      //-Computes displacement and updates position.
      const double dx=dt*double(vr.x);
      const double dy=dt*double(vr.y);
      const double dz=dt*double(vr.z);
      const double2 rxy=posxy[p];
      KerUpdatePos<periactive>(rxy,posz[p],dx,dy,dz,false,p,posxy,posz,dcell,code);
      //-Computes and updates velocity.
      const double2 rposxy=posxy[p];
      float distx,disty,distz;
      KerFtPeriodicDist<periactive>(rposxy.x,rposxy.y,posz[p],fcenter.x,fcenter.y,fcenter.z,fradius,distx,disty,distz);
      vr.x=fvel.x+(fomega.y*distz - fomega.z*disty);
      vr.y=fvel.y+(fomega.z*distx - fomega.x*distz);
      vr.z=fvel.z+(fomega.x*disty - fomega.y*distx);
      velrho[p]=vr;
      //-Updates motionvel and motionace for mDBC no-slip.
      if(mdbc2){ //<vs_m2dbc_ini>
        const float3 mvel0=motionvel[p];
        motionace[p]=make_float3(float((double(vr.x)-mvel0.x)/dt),
                                 float((double(vr.y)-mvel0.y)/dt),
                                 float((double(vr.z)-mvel0.z)/dt));
        motionvel[p]=make_float3(vr.x,vr.y,vr.z);
      } //<vs_m2dbc_end>
      //-Updates floating normals for mDBC.
      if(updatenormals){
        const float3 norf=boundnor[p];
        const double norx=norf.x;
        const double nory=norf.y;
        const double norz=norf.z;
        const float nx=float( mat.a11*norx + mat.a12*nory + mat.a13*norz );
        const float ny=float( mat.a21*norx + mat.a22*nory + mat.a23*norz );
        const float nz=float( mat.a31*norx + mat.a32*nory + mat.a33*norz );
        boundnor[p]=make_float3(nx,ny,nz);
      }
    }
  }
}

//==============================================================================
/// Updates information and particles of floating bodies.
//==============================================================================
void FtPartsUpdate(bool periactive,double dt,bool updatenormals
  ,unsigned np,unsigned fpini,float fradius,tmatrix4d mat
  ,tfloat3 fto_vellin,tfloat3 fto_velang,tdouble3 fto_center
  ,const unsigned* ridpmot,double2* posxy,double* posz,float4* velrho
  ,unsigned* dcell,typecode* code,float3* boundnor,float3* motionvel
  ,float3* motionace,hipStream_t stm)
{
  if(np){
    const unsigned bsize=128; 
    dim3 sgrid=GetSimpleGridSize(np,bsize);
    if(updatenormals && motionvel!=NULL && motionace!=NULL){
      const bool mdbc2=true;
      if(periactive)KerFtPartsUpdate<true ,mdbc2> <<<sgrid,bsize,0,stm>>> (dt,updatenormals,np,fpini,fradius,mat,Float3(fto_vellin),Float3(fto_velang),Double3(fto_center),ridpmot,posxy,posz,velrho,dcell,code,boundnor,motionvel,motionace);
      else          KerFtPartsUpdate<false,mdbc2> <<<sgrid,bsize,0,stm>>> (dt,updatenormals,np,fpini,fradius,mat,Float3(fto_vellin),Float3(fto_velang),Double3(fto_center),ridpmot,posxy,posz,velrho,dcell,code,boundnor,motionvel,motionace); 
    }
    else{
      const bool mdbc2=false;
      if(periactive)KerFtPartsUpdate<true ,mdbc2> <<<sgrid,bsize,0,stm>>> (dt,updatenormals,np,fpini,fradius,mat,Float3(fto_vellin),Float3(fto_velang),Double3(fto_center),ridpmot,posxy,posz,velrho,dcell,code,boundnor,motionvel,motionace);
      else          KerFtPartsUpdate<false,mdbc2> <<<sgrid,bsize,0,stm>>> (dt,updatenormals,np,fpini,fradius,mat,Float3(fto_vellin),Float3(fto_velang),Double3(fto_center),ridpmot,posxy,posz,velrho,dcell,code,boundnor,motionvel,motionace); 
    
    }
  }
}



//##############################################################################
//# Kernels for Periodic conditions
//# Kernels para Periodic conditions
//##############################################################################
//------------------------------------------------------------------------------
/// Marks current periodics to be ignored.
/// Marca las periodicas actuales como ignorar.
//------------------------------------------------------------------------------
__global__ void KerPeriodicIgnore(unsigned n,typecode* code)
{
  const unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    //-Checks code of particles.
    //-Comprueba codigo de particula.
    const typecode rcode=code[p];
    if(CODE_IsPeriodic(rcode))code[p]=CODE_SetOutIgnore(rcode);
  }
}

//==============================================================================
/// Marks current periodics to be ignored.
/// Marca las periodicas actuales como ignorar.
//==============================================================================
void PeriodicIgnore(unsigned n,typecode* code){
  if(n){
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    KerPeriodicIgnore <<<sgrid,SPHBSIZE>>> (n,code);
  }
}

//------------------------------------------------------------------------------
/// Create list of new periodic particles to be duplicated and 
/// marks old periodics to be ignored.
///
/// Crea lista de nuevas particulas periodicas a duplicar y con delper activado
/// marca las periodicas viejas para ignorar.
//------------------------------------------------------------------------------
__global__ void KerPeriodicMakeList(unsigned n,unsigned pini,unsigned nmax
  ,double3 mapposmin,double3 mapposmax,double3 perinc
  ,const double2* posxy,const double* posz,const typecode* code,unsigned* listp)
{
  extern __shared__ unsigned slist[];
  if(!threadIdx.x)slist[0]=0;
  __syncthreads();
  const unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    const unsigned p2=p+pini;
    //-Inteacts with normal or periodic particles.
    //-Se queda con particulas normales o periodicas.
    if(CODE_GetSpecialValue(code[p2])<=CODE_PERIODIC){
      //-Obtains particle position.
      const double2 rxy=posxy[p2];
      const double rx=rxy.x,ry=rxy.y;
      const double rz=posz[p2];
      double rx2=rx+perinc.x,ry2=ry+perinc.y,rz2=rz+perinc.z;
      if(mapposmin.x<=rx2 && mapposmin.y<=ry2 && mapposmin.z<=rz2 && rx2<mapposmax.x && ry2<mapposmax.y && rz2<mapposmax.z){
        unsigned cp=atomicAdd(slist,1);  slist[cp+1]=p2;
      }
      rx2=rx-perinc.x; ry2=ry-perinc.y; rz2=rz-perinc.z;
      if(mapposmin.x<=rx2 && mapposmin.y<=ry2 && mapposmin.z<=rz2 && rx2<mapposmax.x && ry2<mapposmax.y && rz2<mapposmax.z){
        unsigned cp=atomicAdd(slist,1);  slist[cp+1]=(p2|0x80000000);
      }
    }
  }
  __syncthreads();
  const unsigned ns=slist[0];
  __syncthreads();
  if(!threadIdx.x && ns)slist[0]=atomicAdd((listp+nmax),ns);
  __syncthreads();
  if(threadIdx.x<ns){
    unsigned cp=slist[0]+threadIdx.x;
    if(cp<nmax)listp[cp]=slist[threadIdx.x+1];
  }
  if(blockDim.x+threadIdx.x<ns){ //-There may be twice as many periodics per thread. | Puede haber el doble de periodicas que threads.
    unsigned cp=blockDim.x+slist[0]+threadIdx.x;
    if(cp<nmax)listp[cp]=slist[blockDim.x+threadIdx.x+1];
  }
}

//==============================================================================
/// Create list of new periodic particles to be duplicated.
/// With stable activated reorders perioc list.
///
/// Crea lista de nuevas particulas periodicas a duplicar.
/// Con stable activado reordena lista de periodicas.
//==============================================================================
unsigned PeriodicMakeList(unsigned n,unsigned pini,bool stable,unsigned nmax
  ,tdouble3 mapposmin,tdouble3 mapposmax,tdouble3 perinc
  ,const double2* posxy,const double* posz,const typecode* code,unsigned* listp)
{
  unsigned count=0;
  if(n){
    //-lspg size list initialized to zero.
    //-Inicializa tamanho de lista lspg a cero.
    hipMemset(listp+nmax,0,sizeof(unsigned));
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    const unsigned smem=(SPHBSIZE*2+1)*sizeof(unsigned); //-Each particle can leave two new periodic over the counter position. | De cada particula pueden salir 2 nuevas periodicas mas la posicion del contador.
    KerPeriodicMakeList <<<sgrid,SPHBSIZE,smem>>> (n,pini,nmax,Double3(mapposmin),Double3(mapposmax),Double3(perinc),posxy,posz,code,listp);
    hipMemcpy(&count,listp+nmax,sizeof(unsigned),hipMemcpyDeviceToHost);
    //-Reorders list if it is valid and stable has been activated.
    //-Reordena lista si es valida y stable esta activado.
    if(stable && count && count<=nmax){
      thrust::device_ptr<unsigned> dev_list(listp);
      thrust::sort(dev_list,dev_list+count);
    }
  }
  return(count);
}

//------------------------------------------------------------------------------
/// Doubles the position of the indicated particle using a displacement.
/// Duplicate particles are considered valid and are always within
/// the domain.
/// This kernel applies to single-GPU and multi-GPU because the calculations are made
/// from domposmin.
/// It controls the cell coordinates not exceed the maximum.
///
/// Duplica la posicion de la particula indicada aplicandole un desplazamiento.
/// Las particulas duplicadas se considera que siempre son validas y estan dentro
/// del dominio.
/// Este kernel vale para single-gpu y multi-gpu porque los calculos se hacen 
/// a partir de domposmin.
/// Se controla que las coordendas de celda no sobrepasen el maximo.
//------------------------------------------------------------------------------
__device__ void KerPeriodicDuplicatePos(unsigned pnew,unsigned pcopy
  ,bool inverse,double dx,double dy,double dz,uint3 cellmax
  ,double2* posxy,double* posz,unsigned* dcell)
{
  //-Obtains position of the particle to be duplicated.
  //-Obtiene pos de particula a duplicar.
  double2 rxy=posxy[pcopy];
  double rz=posz[pcopy];
  //-Applies displacement.
  rxy.x+=(inverse? -dx: dx);
  rxy.y+=(inverse? -dy: dy);
  rz+=(inverse? -dz: dz);
  //-Computes cell coordinates within the domain.
  //-Calcula coordendas de celda dentro de dominio.
  unsigned cx=unsigned((rxy.x-CTE.domposminx)/CTE.scell);
  unsigned cy=unsigned((rxy.y-CTE.domposminy)/CTE.scell);
  unsigned cz=unsigned((rz-CTE.domposminz)/CTE.scell);
  //-Adjust cell coordinates if they exceed the maximum.
  //-Ajusta las coordendas de celda si sobrepasan el maximo.
  cx=(cx<=cellmax.x? cx: cellmax.x);
  cy=(cy<=cellmax.y? cy: cellmax.y);
  cz=(cz<=cellmax.z? cz: cellmax.z);
  //-Stores position and cell of the new particles.
  //-Graba posicion y celda de nuevas particulas.
  posxy[pnew]=rxy;
  posz[pnew]=rz;
  dcell[pnew]=DCEL_Cell(CTE.cellcode,cx,cy,cz);
}

//------------------------------------------------------------------------------
/// Creates periodic particles from a list of particles to duplicate.
/// It is assumed that all particles are valid.
/// This kernel applies to single-GPU and multi-GPU because it uses domposmin.
///
/// Crea particulas periodicas a partir de una lista con las particulas a duplicar.
/// Se presupone que todas las particulas son validas.
/// Este kernel vale para single-gpu y multi-gpu porque usa domposmin. 
//------------------------------------------------------------------------------
__global__ void KerPeriodicDuplicateVerlet(unsigned n,unsigned pini,uint3 cellmax
  ,double3 perinc,const unsigned* listp,unsigned* idp,typecode* code,unsigned* dcell
  ,double2* posxy,double* posz,float4* velrho,tsymatrix3f* spstau,float4* velrhom1)
{
  const unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    const unsigned pnew=p+pini;
    const unsigned rp=listp[p];
    const unsigned pcopy=(rp&0x7FFFFFFF);
    //-Adjusts cell position of the new particles.
    //-Ajusta posicion y celda de nueva particula.
    KerPeriodicDuplicatePos(pnew,pcopy,(rp>=0x80000000)
      ,perinc.x,perinc.y,perinc.z,cellmax,posxy,posz,dcell);
    //-Copies the remaining data.
    //-Copia el resto de datos.
    idp     [pnew]=idp[pcopy];
    code    [pnew]=CODE_SetPeriodic(code[pcopy]);
    velrho  [pnew]=velrho[pcopy];
    velrhom1[pnew]=velrhom1[pcopy];
    if(spstau)spstau[pnew]=spstau[pcopy];
  }
}

//==============================================================================
/// Creates periodic particles from a list of particles to duplicate.
/// Crea particulas periodicas a partir de una lista con las particulas a duplicar.
//==============================================================================
void PeriodicDuplicateVerlet(unsigned n,unsigned pini,tuint3 domcells
  ,tdouble3 perinc,const unsigned* listp,unsigned* idp,typecode* code
  ,unsigned* dcell,double2* posxy,double* posz,float4* velrho
  ,tsymatrix3f* spstau,float4* velrhom1)
{
  if(n){
    uint3 cellmax=make_uint3(domcells.x-1,domcells.y-1,domcells.z-1);
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    KerPeriodicDuplicateVerlet <<<sgrid,SPHBSIZE>>> (n,pini,cellmax,Double3(perinc)
      ,listp,idp,code,dcell,posxy,posz,velrho,spstau,velrhom1);
  }
}

//------------------------------------------------------------------------------
/// Creates periodic particles from a list of particles to duplicate.
/// It is assumed that all particles are valid.
/// This kernel applies to single-GPU and multi-GPU because it uses domposmin.
///
/// Crea particulas periodicas a partir de una lista con las particulas a duplicar.
/// Se presupone que todas las particulas son validas.
/// Este kernel vale para single-gpu y multi-gpu porque usa domposmin. 
//------------------------------------------------------------------------------
template<bool varspre> __global__ void KerPeriodicDuplicateSymplectic(unsigned n
  ,unsigned pini,uint3 cellmax,double3 perinc,const unsigned* listp,unsigned* idp
  ,typecode* code,unsigned* dcell,double2* posxy,double* posz,float4* velrho
  ,tsymatrix3f* spstau,double2* posxypre,double* poszpre,float4* velrhopre)
{
  const unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    const unsigned pnew=p+pini;
    const unsigned rp=listp[p];
    const unsigned pcopy=(rp&0x7FFFFFFF);
    //-Adjusts cell position of the new particles.
    //-Ajusta posicion y celda de nueva particula.
    KerPeriodicDuplicatePos(pnew,pcopy,(rp>=0x80000000),perinc.x,perinc.y,perinc.z,cellmax,posxy,posz,dcell);
    //-Copies the remaining data.
    //-Copia el resto de datos.
    idp   [pnew]=idp[pcopy];
    code  [pnew]=CODE_SetPeriodic(code[pcopy]);
    velrho[pnew]=velrho[pcopy];
    if(varspre){
      posxypre [pnew]=posxypre[pcopy];
      poszpre  [pnew]=poszpre[pcopy];
      velrhopre[pnew]=velrhopre[pcopy];
    }
    if(spstau)spstau[pnew]=spstau[pcopy];
  }
}

//==============================================================================
/// Creates periodic particles from a list of particles to duplicate.
/// Crea particulas periodicas a partir de una lista con las particulas a duplicar.
//==============================================================================
void PeriodicDuplicateSymplectic(unsigned n,unsigned pini
  ,tuint3 domcells,tdouble3 perinc,const unsigned* listp,unsigned* idp
  ,typecode* code,unsigned* dcell,double2* posxy,double* posz,float4* velrho
  ,tsymatrix3f* spstau,double2* posxypre,double* poszpre,float4* velrhopre)
{
  if(n){
    uint3 cellmax=make_uint3(domcells.x-1,domcells.y-1,domcells.z-1);
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    if(posxypre!=NULL)KerPeriodicDuplicateSymplectic<true>  <<<sgrid,SPHBSIZE>>> 
      (n,pini,cellmax,Double3(perinc),listp,idp,code,dcell,posxy,posz,velrho,spstau
        ,posxypre,poszpre,velrhopre);
    else              KerPeriodicDuplicateSymplectic<false> <<<sgrid,SPHBSIZE>>> 
      (n,pini,cellmax,Double3(perinc),listp,idp,code,dcell,posxy,posz,velrho,spstau
        ,posxypre,poszpre,velrhopre);
  }
}

//------------------------------------------------------------------------------
/// Creates periodic particles from a list of particles to duplicate.
/// It is assumed that all particles are valid.
/// This kernel applies to single-GPU and multi-GPU because it uses domposmin.
///
/// Crea particulas periodicas a partir de una lista con las particulas a duplicar.
/// Se presupone que todas las particulas son validas.
/// Este kernel vale para single-gpu y multi-gpu porque usa domposmin. 
//------------------------------------------------------------------------------
__global__ void KerPeriodicDuplicateNormals(unsigned n,unsigned pini
  ,const unsigned* listp,float3* normals,float3* motionvel,float3* motionace)
{
  const unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    const unsigned pnew=p+pini;
    const unsigned rp=listp[p];
    const unsigned pcopy=(rp&0x7FFFFFFF);
    normals[pnew]=normals[pcopy];
    if(motionvel)motionvel[pnew]=motionvel[pcopy]; //<vs_m2dbc>
    if(motionace)motionace[pnew]=motionace[pcopy]; //<vs_m2dbc>
  }
}

//==============================================================================
/// Creates periodic particles from a list of particles to duplicate.
/// Crea particulas periodicas a partir de una lista con las particulas a duplicar.
//==============================================================================
void PeriodicDuplicateNormals(unsigned n,unsigned pini,const unsigned* listp
  ,float3* normals,float3* motionvel,float3* motionace)
{
  if(n){
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    KerPeriodicDuplicateNormals <<<sgrid,SPHBSIZE>>> (n,pini,listp,normals
      ,motionvel,motionace);
  }
}


//##############################################################################
//# Kernels for Damping.
//##############################################################################
//------------------------------------------------------------------------------
/// Returns TRUE when code==NULL or particle is normal and fluid.
//------------------------------------------------------------------------------
__device__ bool KerIsNormalFluid(const typecode* code,unsigned p){
  if(code){//-Descarta particulas floating o periodicas.
    const typecode cod=code[p];
    return(CODE_IsNormal(cod) && CODE_IsFluid(cod));
  }
  return(true);
}
//------------------------------------------------------------------------------
/// Checks position is inside box limits.
/// Comprueba si la posicion esta dentro de los limites.
//------------------------------------------------------------------------------
__device__ bool KerPointInBox(double px,double py,double pz,const double3& p1
  ,const double3& p2)
{
  return(p1.x<=px && p1.y<=py && p1.z<=pz && px<=p2.x && py<=p2.y && pz<=p2.z);
}
//------------------------------------------------------------------------------
/// Solves point on the plane.
/// Resuelve punto en el plano.
//------------------------------------------------------------------------------
__device__ double KerPointPlane(const double4& pla,double px,double py,double pz)
{
  return(pla.x*px+pla.y*py+pla.z*pz+pla.w);
}
//------------------------------------------------------------------------------
/// Solves point on the plane.
/// Resuelve punto en el plano.
//------------------------------------------------------------------------------
__device__ double KerPointPlane(const double4& pla,const double3& pt)
{
  return(pla.x*pt.x+pla.y*pt.y+pla.z*pt.z+pla.w);
}

//------------------------------------------------------------------------------
/// Applies Damping.
/// Aplica Damping.
//------------------------------------------------------------------------------
__global__ void KerComputeDampingPlane(unsigned n,unsigned pini
  ,double dt,double4 plane,float dist,float over,float3 factorxyz,float redumax
  ,const double2* posxy,const double* posz,const typecode* code
  ,float4* velrho)
{
  unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    const unsigned p1=p+pini;
    const bool ok=KerIsNormalFluid(code,p1);//-Ignore floating and periodic particles. | Descarta particulas floating o periodicas.
    if(ok){
      const double2 rposxy=posxy[p1];
      const double rposz=posz[p1];
      double vdis=KerPointPlane(plane,rposxy.x,rposxy.y,rposz);  //fgeo::PlanePoint(plane,ps);
      if(0<vdis && vdis<=dist+over){
        const double fdis=(vdis>=dist? 1.: vdis/dist);
        const double redudt=dt*(fdis*fdis)*redumax;
        double redudtx=(1.-redudt*factorxyz.x);
        double redudty=(1.-redudt*factorxyz.y);
        double redudtz=(1.-redudt*factorxyz.z);
        redudtx=(redudtx<0? 0.: redudtx);
        redudty=(redudty<0? 0.: redudty);
        redudtz=(redudtz<0? 0.: redudtz);
        float4 rvel=velrho[p1];
        rvel.x=float(redudtx*rvel.x); 
        rvel.y=float(redudty*rvel.y); 
        rvel.z=float(redudtz*rvel.z);
        velrho[p1]=rvel;
      }
    }
  }
}
//==============================================================================
/// Applies Damping.
/// Aplica Damping.
//==============================================================================
void ComputeDampingPlane(double dt,double4 plane,float dist,float over
  ,float3 factorxyz,float redumax,unsigned n,unsigned pini
  ,const double2* posxy,const double* posz,const typecode* code,float4* velrho)
{
  if(n){
    dim3 sgridf=GetSimpleGridSize(n,SPHBSIZE);
    KerComputeDampingPlane <<<sgridf,SPHBSIZE>>> (n,pini,dt,plane,dist,over
      ,factorxyz,redumax,posxy,posz,code,velrho);
  }
}

//------------------------------------------------------------------------------
/// Applies Damping to limited domain.
/// Aplica Damping limitado a un dominio.
//------------------------------------------------------------------------------
__global__ void KerComputeDampingPlaneDom(unsigned n,unsigned pini
  ,double dt,double4 plane,float dist,float over,float3 factorxyz,float redumax
  ,double zmin,double zmax,double4 pla0,double4 pla1,double4 pla2,double4 pla3
  ,const double2* posxy,const double* posz,const typecode* code
  ,float4* velrho)
{
  unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    const unsigned p1=p+pini;
    const bool ok=KerIsNormalFluid(code,p1);//-Ignore floating and periodic particles. | Descarta particulas floating o periodicas.
    if(ok){
      const double2 rposxy=posxy[p1];
      const double rposz=posz[p1];
      const double3 ps=make_double3(rposxy.x,rposxy.y,rposz);
      double vdis=KerPointPlane(plane,ps);  //fgeo::PlanePoint(plane,ps);
      if(0<vdis && vdis<=dist+over){
        if(ps.z>=zmin && ps.z<=zmax && KerPointPlane(pla0,ps)<=0 && KerPointPlane(pla1,ps)<=0 && KerPointPlane(pla2,ps)<=0 && KerPointPlane(pla3,ps)<=0){
          const double fdis=(vdis>=dist? 1.: vdis/dist);
          const double redudt=dt*(fdis*fdis)*redumax;
          double redudtx=(1.-redudt*factorxyz.x);
          double redudty=(1.-redudt*factorxyz.y);
          double redudtz=(1.-redudt*factorxyz.z);
          redudtx=(redudtx<0? 0.: redudtx);
          redudty=(redudty<0? 0.: redudty);
          redudtz=(redudtz<0? 0.: redudtz);
          float4 rvel=velrho[p1];
          rvel.x=float(redudtx*rvel.x); 
          rvel.y=float(redudty*rvel.y); 
          rvel.z=float(redudtz*rvel.z); 
          velrho[p1]=rvel;
        }
      }
    }
  }
}
//==============================================================================
/// Applies Damping to limited domain.
/// Aplica Damping limitado a un dominio.
//==============================================================================
void ComputeDampingPlaneDom(double dt,double4 plane,float dist,float over
  ,float3 factorxyz,float redumax
  ,double zmin,double zmax,double4 pla0,double4 pla1,double4 pla2,double4 pla3
  ,unsigned n,unsigned pini,const double2* posxy,const double* posz
  ,const typecode* code,float4* velrho)
{
  if(n){
    dim3 sgridf=GetSimpleGridSize(n,SPHBSIZE);
    KerComputeDampingPlaneDom <<<sgridf,SPHBSIZE>>> (n,pini,dt,plane,dist,over,factorxyz
      ,redumax,zmin,zmax,pla0,pla1,pla2,pla3,posxy,posz,code,velrho);
  }
}


//------------------------------------------------------------------------------
/// Applies Damping according box configuration.
/// Aplica Damping segun cofiguracion de caja.
//------------------------------------------------------------------------------
__global__ void KerComputeDampingBox(unsigned n,unsigned pini
  ,double dt,float3 factorxyz,float redumax
  ,double3 limitmin1,double3 limitmin2,double3 limitmax1,double3 limitmax2
  ,double3 limitover1,double3 limitover2,double3 boxsize1,double3 boxsize2
  ,const double2* posxy,const double* posz,const typecode* code
  ,float4* velrho)
{
  unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    const unsigned p1=p+pini;
    const bool ok=KerIsNormalFluid(code,p1);//-Ignore floating and periodic particles. | Descarta particulas floating o periodicas.
    if(ok){
      const double2 rposxy=posxy[p1];
      const double rposz=posz[p1];
      //-Check if it is within the domain. | Comprueba si esta dentro del dominio.
      if(KerPointInBox(rposxy.x,rposxy.y,rposz,limitover1,limitover2)){//-Inside overlimit domain.
        if(!KerPointInBox(rposxy.x,rposxy.y,rposz,limitmin1,limitmin2)){//-Outside free domain.
          double fdis=1.;
          if(KerPointInBox(rposxy.x,rposxy.y,rposz,limitmax1,limitmax2)){//-Compute damping coefficient.
            fdis=0;
            if(boxsize2.z){ const double fdiss=(rposz   -limitmin2.z)/boxsize2.z; fdis=(fdis>=fdiss? fdis: fdiss); }
            if(boxsize2.y){ const double fdiss=(rposxy.y-limitmin2.y)/boxsize2.y; fdis=(fdis>=fdiss? fdis: fdiss); }
            if(boxsize2.x){ const double fdiss=(rposxy.x-limitmin2.x)/boxsize2.x; fdis=(fdis>=fdiss? fdis: fdiss); }
            if(boxsize1.z){ const double fdiss=(limitmin1.z-rposz   )/boxsize1.z; fdis=(fdis>=fdiss? fdis: fdiss); }
            if(boxsize1.y){ const double fdiss=(limitmin1.y-rposxy.y)/boxsize1.y; fdis=(fdis>=fdiss? fdis: fdiss); }
            if(boxsize1.x){ const double fdiss=(limitmin1.x-rposxy.x)/boxsize1.x; fdis=(fdis>=fdiss? fdis: fdiss); }
          }
          const double redudt=dt*(fdis*fdis)*redumax;
          double redudtx=(1.-redudt*factorxyz.x);
          double redudty=(1.-redudt*factorxyz.y);
          double redudtz=(1.-redudt*factorxyz.z);
          redudtx=(redudtx<0? 0.: redudtx);
          redudty=(redudty<0? 0.: redudty);
          redudtz=(redudtz<0? 0.: redudtz);
          float4 rvel=velrho[p1];
          rvel.x=float(redudtx*rvel.x); 
          rvel.y=float(redudty*rvel.y); 
          rvel.z=float(redudtz*rvel.z);
          //rvel.x=rvel.y=rvel.z=0;
          velrho[p1]=rvel;
        }
      }
    }
  }
}
//==============================================================================
/// Applies Damping according box configuration.
/// Aplica Damping segun cofiguracion de caja.
//==============================================================================
void ComputeDampingBox(unsigned n,unsigned pini,double dt,float3 factorxyz,float redumax
  ,double3 limitmin1,double3 limitmin2,double3 limitmax1,double3 limitmax2
  ,double3 limitover1,double3 limitover2,double3 boxsize1,double3 boxsize2
  ,const double2* posxy,const double* posz,const typecode* code,float4* velrho)
{
  if(n){
    dim3 sgridf=GetSimpleGridSize(n,SPHBSIZE);
    KerComputeDampingBox <<<sgridf,SPHBSIZE>>> (n,pini,dt,factorxyz,redumax
      ,limitmin1,limitmin2,limitmax1,limitmax2,limitover1,limitover2,boxsize1,boxsize2
      ,posxy,posz,code,velrho);
  }
}


//------------------------------------------------------------------------------
/// Applies Damping to limited cylinder domain.
/// Aplica Damping limitado a un dominio de cilindro.
//------------------------------------------------------------------------------
__global__ void KerComputeDampingCylinder(unsigned n,unsigned pini
  ,double dt,bool isvertical,double3 point1,double3 point2,double limitmin
  ,float dist,float over,float3 factorxyz,float redumax
  ,const double2* posxy,const double* posz,const typecode* code
  ,float4* velrho)
{
  unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    const unsigned p1=p+pini;
    const bool ok=KerIsNormalFluid(code,p1);//-Ignore floating and periodic particles. | Descarta particulas floating o periodicas.
    if(ok){
      //-Check if it is within the domain. | Comprueba si esta dentro del dominio.
      const double2 rposxy=posxy[p1];
      const double rposz=posz[p1];
      const double3 ps=make_double3(rposxy.x,rposxy.y,rposz);
      const double vdis=(isvertical? 
        sqrt((ps.x-point1.x)*(ps.x-point1.x)+(ps.y-point1.y)*(ps.y-point1.y)): 
        cugeo::LinePointDist(ps,point1,point2)
        ) - limitmin;
      if(0<vdis && vdis<=dist+over){
        const double fdis=(vdis>=dist? 1.: vdis/dist);
        const double redudt=dt*(fdis*fdis)*redumax;
        double redudtx=(1.-redudt*factorxyz.x);
        double redudty=(1.-redudt*factorxyz.y);
        double redudtz=(1.-redudt*factorxyz.z);
        redudtx=(redudtx<0? 0.: redudtx);
        redudty=(redudty<0? 0.: redudty);
        redudtz=(redudtz<0? 0.: redudtz);
        float4 rvel=velrho[p1];
        rvel.x=float(redudtx*rvel.x); 
        rvel.y=float(redudty*rvel.y); 
        rvel.z=float(redudtz*rvel.z); 
        velrho[p1]=rvel;
      }
    }
  }
}
//==============================================================================
/// Applies Damping to limited cylinder domain.
/// Aplica Damping limitado a un dominio de cilindro.
//==============================================================================
void ComputeDampingCylinder(unsigned n,unsigned pini
  ,double dt,double3 point1,double3 point2,double limitmin
  ,float dist,float over,float3 factorxyz,float redumax
  ,const double2* posxy,const double* posz,const typecode* code
  ,float4* velrho)
{
  if(n){
    const bool isvertical=(point1.x==point2.x && point1.y==point2.y);
    dim3 sgridf=GetSimpleGridSize(n,SPHBSIZE);
    KerComputeDampingCylinder <<<sgridf,SPHBSIZE>>> (n,pini,dt
      ,isvertical,point1,point2,limitmin,dist,over,factorxyz,redumax
      ,posxy,posz,code,velrho);
  }
}

 //<vs_outpaarts_ini>
//##############################################################################
//# Kernels for OutputParts.
//##############################################################################
//------------------------------------------------------------------------------
/// Compute filter for particles.
//------------------------------------------------------------------------------
__global__ void KerComputeOutputPartsInit(unsigned n,unsigned pini
  ,byte resmask,bool selall,byte* sel)
{
  unsigned pp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(pp<n){
    const unsigned p=pp+pini;
    byte psel=sel[p];
    sel[p]=(selall? psel|resmask: psel&(~resmask));
  }
}
//==============================================================================
/// Compute filter for particles.
//==============================================================================
void ComputeOutputPartsInit(byte resmask,bool selall,unsigned n,unsigned pini
  ,byte* sel)
{
  if(n){
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    KerComputeOutputPartsInit <<<sgrid,SPHBSIZE>>> (n,pini,resmask,selall,sel);
  }
}

//------------------------------------------------------------------------------
/// Compute filter for particles.
//------------------------------------------------------------------------------
__global__ void KerComputeOutputPartsGroup(unsigned n,unsigned pini
  ,byte resmask,byte resprev,bool cmband,bool inverse,byte* sel)
{
  unsigned pp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(pp<n){
    const unsigned p=pp+pini;
    byte psel=sel[p];
    bool r=((psel&resprev)!=0);  //-lv=0
    bool ok=((psel&resmask)!=0); //-lv+1
    if(inverse)ok=!ok;
    r=(cmband? r&&ok: r||ok);
    psel=psel&(~resprev);
    if(r)psel=psel|resprev;
    sel[p]=psel;
  }
}
//==============================================================================
/// Compute filter for particles.
//==============================================================================
void ComputeOutputPartsGroup(byte resmask,byte resprev,bool cmband,bool inverse
  ,unsigned n,unsigned pini,byte* sel)
{
  if(n){
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    KerComputeOutputPartsGroup <<<sgrid,SPHBSIZE>>> (n,pini,resmask,resprev,cmband,inverse,sel);
  }
}

//------------------------------------------------------------------------------
/// Compute filter for particles.
//------------------------------------------------------------------------------
__global__ void KerComputeOutputPartsPos(unsigned n,unsigned pini
  ,byte resmask,bool cmband,bool inverse,double3 pmin,double3 pmax
  ,const double2* posxy,const double* posz,byte* sel)
{
  unsigned pp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(pp<n){
    const unsigned p=pp+pini;
    byte psel=sel[p];
    bool r=((psel&resmask)!=0);
    if(r==cmband){ //if((r && cmband) || (!r && !cmband)){
      const double2 rposxy=posxy[p];
      const double rposz=posz[p];
      bool ok=(pmin.x<=rposxy.x && rposxy.x <=pmax.x &&
               pmin.z<=rposz    && rposz    <=pmax.z &&
               pmin.y<=rposxy.y && rposxy.y <=pmax.y);
      if(inverse)ok=!ok;
      r=(cmband? r&&ok: r||ok);
      psel=psel&(~resmask);
      if(r)psel=psel|resmask;
      sel[p]=psel;
    }
  }
}
//==============================================================================
/// Compute filter for particles.
//==============================================================================
void ComputeOutputPartsPos(byte resmask,bool cmband,bool inverse
  ,double3 pmin,double3 pmax,unsigned n,unsigned pini
  ,const double2* posxy,const double* posz,byte* sel)
{
  if(n){
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    KerComputeOutputPartsPos <<<sgrid,SPHBSIZE>>> (n,pini,resmask,cmband,inverse
      ,pmin,pmax,posxy,posz,sel);
  }
}

//------------------------------------------------------------------------------
/// Compute filter for particles.
//------------------------------------------------------------------------------
__global__ void KerComputeOutputPartsPlane(unsigned n,unsigned pini
  ,byte resmask,bool cmband,bool inverse,double4 plane,float maxdist
  ,const double2* posxy,const double* posz,byte* sel)
{
  unsigned pp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(pp<n){
    const unsigned p=pp+pini;
    byte psel=sel[p];
    bool r=((psel&resmask)!=0);
    if(r==cmband){ //if((r && cmband) || (!r && !cmband)){
      const double2 rposxy=posxy[p];
      const double rposz=posz[p];
      const double dist=KerPointPlane(plane,rposxy.x,rposxy.y,rposz);  //fgeo::PlanePoint(plane,ps);
      bool ok=(dist>=0 && dist<=maxdist);
      if(inverse)ok=!ok;
      r=(cmband? r&&ok: r||ok);
      psel=psel&(~resmask);
      if(r)psel=psel|resmask;
      sel[p]=psel;
    }
  }
}
//==============================================================================
/// Compute filter for particles.
//==============================================================================
void ComputeOutputPartsPlane(byte resmask,bool cmband,bool inverse
  ,double4 plane,float maxdist,unsigned n,unsigned pini
  ,const double2* posxy,const double* posz,byte* sel)
{
  if(n){
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    KerComputeOutputPartsPlane <<<sgrid,SPHBSIZE>>> (n,pini,resmask,cmband,inverse
      ,plane,maxdist,posxy,posz,sel);
  }
}

//------------------------------------------------------------------------------
/// Compute filter for particles.
//------------------------------------------------------------------------------
__global__ void KerComputeOutputPartsSphere(unsigned n,unsigned pini
  ,byte resmask,bool cmband,bool inverse,double3 pcen,float radius2
  ,const double2* posxy,const double* posz,byte* sel)
{
  unsigned pp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(pp<n){
    const unsigned p=pp+pini;
    byte psel=sel[p];
    bool r=((psel&resmask)!=0);
    if(r==cmband){ //if((r && cmband) || (!r && !cmband)){
      const double2 rposxy=posxy[p];
      const double rposz=posz[p];
      const float dx=float(pcen.x-rposxy.x);
      const float dy=float(pcen.y-rposxy.y);
      const float dz=float(pcen.z-rposz);
      bool ok=(dx*dx+dy*dy+dz*dz <= radius2);
      if(inverse)ok=!ok;
      r=(cmband? r&&ok: r||ok);
      psel=psel&(~resmask);
      if(r)psel=psel|resmask;
      sel[p]=psel;
    }
  }
}
//==============================================================================
/// Compute filter for particles.
//==============================================================================
void ComputeOutputPartsSphere(byte resmask,bool cmband,bool inverse
  ,double3 pcen,float radius2,unsigned n,unsigned pini
  ,const double2* posxy,const double* posz,byte* sel)
{
  if(n){
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    KerComputeOutputPartsSphere <<<sgrid,SPHBSIZE>>> (n,pini,resmask,cmband,inverse
      ,pcen,radius2,posxy,posz,sel);
  }
}


//------------------------------------------------------------------------------
/// Compute filter for particles.
//------------------------------------------------------------------------------
template<bool isvertical> __global__ void KerComputeOutputPartsCylinder(unsigned n
  ,unsigned pini,byte resmask,bool cmband,bool inverse
  ,double4 plane,float maxdist,double3 pcen1,double3 pcen2,float radius
  ,const double2* posxy,const double* posz,byte* sel)
{
  unsigned pp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(pp<n){
    const unsigned p=pp+pini;
    byte psel=sel[p];
    bool r=((psel&resmask)!=0);
    if(r==cmband){ //if((r && cmband) || (!r && !cmband)){
      const double2 rposxy=posxy[p];
      const double rposz=posz[p];
      const double dist=KerPointPlane(plane,rposxy.x,rposxy.y,rposz);  //fgeo::PlanePoint(plane,ps);
      bool ok=(dist>=0 && dist<=maxdist);
      if(ok && isvertical){
        const float dx=float(rposxy.x-pcen1.x);
        const float dy=float(rposxy.y-pcen1.y);
        ok=(dx*dx+dy*dy <= radius*radius);
      }
      if(ok && !isvertical){
        //cugeo::LinePointDist(ps,Point1,Point2)
        const double ar=cugeo::TriangleArea(make_double3(rposxy.x,rposxy.y,rposz),pcen1,pcen2);
        const double dis=(ar*2)/maxdist;
        ok=(dis<=radius);
      }
      if(inverse)ok=!ok;
      r=(cmband? r&&ok: r||ok);
      psel=psel&(~resmask);
      if(r)psel=psel|resmask;
      sel[p]=psel;
    }
  }
}
//==============================================================================
/// Compute filter for particles.
//==============================================================================
void ComputeOutputPartsCylinder(byte resmask,bool cmband,bool inverse
  ,double4 plane,float maxdist,double3 pcen1,double3 pcen2,float radius
  ,unsigned n,unsigned pini,const double2* posxy,const double* posz,byte* sel)
{
  if(n){
    const bool isvertical=(pcen1.x==pcen2.x && pcen1.y==pcen2.y);
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    if(isvertical) KerComputeOutputPartsCylinder<true > <<<sgrid,SPHBSIZE>>> (n,pini
      ,resmask,cmband,inverse,plane,maxdist,pcen1,pcen2,radius,posxy,posz,sel);
    if(!isvertical)KerComputeOutputPartsCylinder<false> <<<sgrid,SPHBSIZE>>> (n,pini
      ,resmask,cmband,inverse,plane,maxdist,pcen1,pcen2,radius,posxy,posz,sel);
  }
}

//------------------------------------------------------------------------------
/// Compute filter for particles.
//------------------------------------------------------------------------------
__global__ void KerComputeOutputPartsType(unsigned n,unsigned pini
  ,byte resmask,bool cmband,bool inverse,byte types
  ,const typecode* code,byte* sel)
{
  unsigned pp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(pp<n){
    const unsigned p=pp+pini;
    byte psel=sel[p];
    bool r=((psel&resmask)!=0);
    if(r==cmband){ //if((r && cmband) || (!r && !cmband)){
      const typecode codetp=CODE_GetType(code[p]);
      bool ok=((types&1 && codetp==CODE_TYPE_FIXED) ||
               (types&2 && codetp==CODE_TYPE_MOVING) ||
               (types&4 && codetp==CODE_TYPE_FLOATING) ||
               (types&8 && codetp==CODE_TYPE_FLUID) );
      if(inverse)ok=!ok;
      r=(cmband? r&&ok: r||ok);
      psel=psel&(~resmask);
      if(r)psel=psel|resmask;
      sel[p]=psel;
    }
  }
}
//==============================================================================
/// Compute filter for particles.
//==============================================================================
void ComputeOutputPartsType(byte resmask,bool cmband,bool inverse
  ,byte types,unsigned n,unsigned pini,const typecode* code,byte* sel)
{
  if(n){
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    KerComputeOutputPartsType <<<sgrid,SPHBSIZE>>> (n,pini,resmask,cmband,inverse
      ,types,code,sel);
  }
}

//------------------------------------------------------------------------------
/// Compute filter for particles.
//------------------------------------------------------------------------------
__global__ void KerComputeOutputPartsMk(unsigned n,unsigned pini
  ,byte resmask,bool cmband,bool inverse,typecode mkcode1,typecode mkcode2
  ,const typecode* code,byte* sel)
{
  unsigned pp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(pp<n){
    const unsigned p=pp+pini;
    byte psel=sel[p];
    bool r=((psel&resmask)!=0);
    if(r==cmband){ //if((r && cmband) || (!r && !cmband)){
      const typecode codetp=CODE_GetTypeAndValue(code[p]);
      bool ok=(mkcode1<=codetp && codetp<=mkcode2);
      if(inverse)ok=!ok;
      r=(cmband? r&&ok: r||ok);
      psel=psel&(~resmask);
      if(r)psel=psel|resmask;
      sel[p]=psel;
    }
  }
}
//==============================================================================
/// Compute filter for particles.
//==============================================================================
void ComputeOutputPartsMk(byte resmask,bool cmband,bool inverse
  ,typecode mkcode1,typecode mkcode2,unsigned n,unsigned pini
  ,const typecode* code,byte* sel)
{
  if(n){
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    KerComputeOutputPartsMk <<<sgrid,SPHBSIZE>>> (n,pini,resmask,cmband,inverse
      ,mkcode1,mkcode2,code,sel);
  }
}

//<vs_outpaarts_end>

//<vs_flexstruc_ini>
//==============================================================================
/// Functor for checking if particle is a flexible structure particle.
/// Funtor para verificar si la partícula es una partícula de estructura flexible.
//==============================================================================
struct IsFlexStrucAny{ __host__ __device__ bool operator()(const typecode& code) { return CODE_IsFlexStrucAny(code); } };

//==============================================================================
/// Functor for checking if a flexible structure is out.
/// Functor para comprobar si una estructura flexible está fuera.
//==============================================================================
struct FlexStrucAnyIsOut{ __host__ __device__ bool operator()(const typecode& code) { return CODE_IsFlexStrucAny(code) && !CODE_IsNotOut(code); } };

//==============================================================================
/// Finds the clamp particles and updates the code.
/// Encuentra las partículas de abrazadera y actualiza el código.
//==============================================================================
__global__ void KerSetFlexStrucClampCodes(unsigned n,const float4* poscell,const StFlexStrucData* flexstrucdata,typecode* code){
  const unsigned p=blockIdx.x*blockDim.x+threadIdx.x; //-Number of thread.
  if(p<n){
    const unsigned p1=p;      //-Number of particle.
    const typecode codep1=code[p1];

    //-If potentially a clamp particle.
    if((CODE_IsFixed(codep1)||CODE_IsMoving(codep1))&&!CODE_IsFlexStrucAny(codep1)){

      //-Loads particle p1 data.
      const float4 pscellp1=poscell[p1];

      //-Loop through other boundary particles.
      for(unsigned p2=0;p2<n;p2++){
        const typecode codep2=code[p2];
        if(CODE_IsFlexStrucFlex(codep2)){
          const unsigned nc=flexstrucdata[CODE_GetIbodyFlexStruc(codep2)].nc;
          const typecode* clampcode=flexstrucdata[CODE_GetIbodyFlexStruc(codep2)].clampcode;
          if(thrust::find(thrust::seq,clampcode,clampcode+nc,codep1)!=clampcode+nc){
            const float4 pscellp2=poscell[p2];
            float drx=pscellp1.x-pscellp2.x+CTE.poscellsize*(PSCEL_GetfX(pscellp1.w)-PSCEL_GetfX(pscellp2.w));
            float dry=pscellp1.y-pscellp2.y+CTE.poscellsize*(PSCEL_GetfY(pscellp1.w)-PSCEL_GetfY(pscellp2.w));
            float drz=pscellp1.z-pscellp2.z+CTE.poscellsize*(PSCEL_GetfZ(pscellp1.w)-PSCEL_GetfZ(pscellp2.w));
            const float rr2=drx*drx+dry*dry+drz*drz;
            if(rr2<=CTE.kernelsize2&&rr2>=ALMOSTZERO){
              code[p1]=CODE_ToFlexStrucClamp(codep1,CODE_GetIbodyFlexStruc(codep2));
              break;
            }
          }
        }
      }
    }
  }
}

//==============================================================================
/// Finds the clamp particles and updates the code.
/// Encuentra las partículas de abrazadera y actualiza el código.
//==============================================================================
void SetFlexStrucClampCodes(unsigned npb,const float4* poscell,const StFlexStrucData* flexstrucdata,typecode* code){
  if(npb){
    dim3 sgridb=GetSimpleGridSize(npb,SPHBSIZE);
    KerSetFlexStrucClampCodes <<<sgridb,SPHBSIZE>>> (npb,poscell,flexstrucdata,code);
  }
}

//==============================================================================
/// Counts the number of flexible structure particles (includes clamps).
/// Cuenta el número de partículas de estructura flexible (incluye abrazaderas).
//==============================================================================
unsigned CountFlexStrucParts(unsigned npb,const typecode* code){
  if(npb){
    thrust::device_ptr<const typecode> dev_code(code);
    return thrust::count_if(dev_code,dev_code+npb,IsFlexStrucAny());
  }
  return 0;
}

//==============================================================================
/// Calculates indices to the main arrays for the flexible structure particles.
/// Calcula los índices de las matrices principales para las partículas de estructura flexible.
//==============================================================================
void CalcFlexStrucRidp(unsigned npb,const typecode* code,unsigned* flexstrucridp){
  if(npb){
    thrust::counting_iterator<unsigned> idx(0);
    thrust::device_ptr<const typecode> dev_code(code);
    thrust::device_ptr<unsigned> dev_flexstrucridp(flexstrucridp);
    thrust::copy_if(idx,idx+npb,dev_code,dev_flexstrucridp,IsFlexStrucAny());
  }
}

//==============================================================================
/// Gathers values from a main array into the smaller flexible structure array.
/// Reúne valores de una matriz principal en la matriz de estructura flexible más pequeña.
//==============================================================================
void GatherToFlexStrucArray(unsigned npfs,const unsigned* flexstrucridp,const float4* fullarray,float4* flexstrucarray){
  if(npfs){
    thrust::device_ptr<const unsigned> dev_flexstrucridp(flexstrucridp);
    thrust::device_ptr<const float4> dev_fullarray(fullarray);
    thrust::device_ptr<float4> dev_flexstrucarray(flexstrucarray);
    thrust::gather(dev_flexstrucridp,dev_flexstrucridp+npfs,dev_fullarray,dev_flexstrucarray);
  }
}

//==============================================================================
/// Gathers values from a main array into the smaller flexible structure array.
/// Reúne valores de una matriz principal en la matriz de estructura flexible más pequeña.
//==============================================================================
void GatherToFlexStrucArray(unsigned npfs,const unsigned* flexstrucridp,const float3* fullarray,float3* flexstrucarray){
  if(npfs){
    thrust::device_ptr<const unsigned> dev_flexstrucridp(flexstrucridp);
    thrust::device_ptr<const float3> dev_fullarray(fullarray);
    thrust::device_ptr<float3> dev_flexstrucarray(flexstrucarray);
    thrust::gather(dev_flexstrucridp,dev_flexstrucridp+npfs,dev_fullarray,dev_flexstrucarray);
  }
}

//==============================================================================
/// Counts the total number of flexible structure pairs (neighbours).
/// Cuenta el número total de pares de estructuras flexibles (vecinos).
//==============================================================================
__global__ void KerCountFlexStrucPairs(unsigned n,const float4* poscell0,unsigned* numpairs){
  const unsigned p=blockIdx.x*blockDim.x+threadIdx.x; //-Number of thread.
  if(p<n){
    const unsigned pfs1=p;      //-Number of particle.
    unsigned numpairsp1=0;

    //-Loads particle p1 data.
    const float4 pscell0p1=poscell0[pfs1];

    //-Loop through other flexible structure particles.
    for(unsigned pfs2=0;pfs2<n;pfs2++){
      const float4 pscell0p2=poscell0[pfs2];
      float drx0=pscell0p1.x-pscell0p2.x+CTE.poscellsize*(PSCEL_GetfX(pscell0p1.w)-PSCEL_GetfX(pscell0p2.w));
      float dry0=pscell0p1.y-pscell0p2.y+CTE.poscellsize*(PSCEL_GetfY(pscell0p1.w)-PSCEL_GetfY(pscell0p2.w));
      float drz0=pscell0p1.z-pscell0p2.z+CTE.poscellsize*(PSCEL_GetfZ(pscell0p1.w)-PSCEL_GetfZ(pscell0p2.w));
      const float rr20=drx0*drx0+dry0*dry0+drz0*drz0;
      if(rr20<=CTE.kernelsize2&&rr20>=ALMOSTZERO)numpairsp1++;
    }
    numpairs[pfs1]=numpairsp1;
  }
}

//==============================================================================
/// Counts the total number of flexible structure pairs (neighbours).
/// Cuenta el número total de pares de estructuras flexibles (vecinos).
//==============================================================================
unsigned CountFlexStrucPairs(unsigned npfs,const float4* poscell0,unsigned* numpairs){
  if(npfs){
    dim3 sgridb=GetSimpleGridSize(npfs,SPHBSIZE);
    KerCountFlexStrucPairs <<<sgridb,SPHBSIZE>>> (npfs,poscell0,numpairs);
    thrust::device_ptr<unsigned> dev_numpairs(numpairs);
    return thrust::reduce(dev_numpairs,dev_numpairs+npfs);
  }
  return 0;
}

//==============================================================================
/// Sets the indices for each flexible structure pair.
/// Establece los índices para cada par de estructuras flexibles.
//==============================================================================
__global__ void KerSetFlexStrucPairs(unsigned n,const float4* poscell0,unsigned** pairidx)
{
  const unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of thread.
  if(p<n){
    const unsigned pfs1=p;      //-Number of particle.
    unsigned idx=0;

    //-Loads particle p1 data.
    const float4 pscell0p1=poscell0[pfs1];

    //-Loop through other flexible structure particles.
    for(unsigned pfs2=0;pfs2<n;pfs2++){
      const float4 pscell0p2=poscell0[pfs2];
      float drx0=pscell0p1.x-pscell0p2.x+CTE.poscellsize*(PSCEL_GetfX(pscell0p1.w)-PSCEL_GetfX(pscell0p2.w));
      float dry0=pscell0p1.y-pscell0p2.y+CTE.poscellsize*(PSCEL_GetfY(pscell0p1.w)-PSCEL_GetfY(pscell0p2.w));
      float drz0=pscell0p1.z-pscell0p2.z+CTE.poscellsize*(PSCEL_GetfZ(pscell0p1.w)-PSCEL_GetfZ(pscell0p2.w));
      const float rr20=drx0*drx0+dry0*dry0+drz0*drz0;
      if(rr20<=CTE.kernelsize2&&rr20>=ALMOSTZERO)pairidx[pfs1][idx++]=pfs2;
    }
  }
}

//==============================================================================
/// Sets the indices for each flexible structure pair.
/// Establece los índices para cada par de estructuras flexibles.
//==============================================================================
void SetFlexStrucPairs(unsigned npfs,const float4* poscell0,unsigned** pairidx){
  if(npfs){
    dim3 sgridb=GetSimpleGridSize(npfs,SPHBSIZE);
    KerSetFlexStrucPairs <<<sgridb,SPHBSIZE>>> (npfs,poscell0,pairidx);
  }
}

//==============================================================================
/// Calculates the kernel correction matrix for each flexible structure particle.
/// Calcula la matriz de corrección del kernel para cada partícula de estructura flexible.
//==============================================================================
template<TpKernel tker,bool simulate2d> __global__ void KerCalcFlexStrucKerCorr(unsigned n,const typecode* code,const StFlexStrucData* flexstrucdata
    ,const unsigned* flexstrucridp,const float4* poscell0,const unsigned* numpairs,const unsigned* const* pairidx
    ,tmatrix3f* kercorr)
{
  const unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of thread.
  if(p<n){
    const unsigned pfs1=p;      //-Number of particle.

    //-Get number of pairs for this particle.
    const unsigned numpairsp1=numpairs[pfs1];

    //-If this particle has pairs.
    if(numpairsp1){
      tmatrix3f kercorrp1={0};

      //-Obtains basic data of particle p1.
      const float vol0p1=flexstrucdata[CODE_GetIbodyFlexStruc(code[flexstrucridp[pfs1]])].vol0;
      const float4 pscell0p1=poscell0[pfs1];

      //-Calculate kernel correction matrix.
      for(unsigned pair=0;pair<numpairsp1;pair++){
        const unsigned pfs2=pairidx[pfs1][pair];
        const float4 pscell0p2=poscell0[pfs2];
        float drx0=pscell0p1.x-pscell0p2.x + CTE.poscellsize*(PSCEL_GetfX(pscell0p1.w)-PSCEL_GetfX(pscell0p2.w));
        float dry0=pscell0p1.y-pscell0p2.y + CTE.poscellsize*(PSCEL_GetfY(pscell0p1.w)-PSCEL_GetfY(pscell0p2.w));
        float drz0=pscell0p1.z-pscell0p2.z + CTE.poscellsize*(PSCEL_GetfZ(pscell0p1.w)-PSCEL_GetfZ(pscell0p2.w));
        const float rr20=drx0*drx0+dry0*dry0+drz0*drz0;
        //-Computes kernel.
        const float fac0=cufsph::GetKernel_Fac<tker>(rr20);
        const float frx0=fac0*drx0,fry0=fac0*dry0,frz0=fac0*drz0; //-Gradients.
        kercorrp1.a11-=vol0p1*drx0*frx0; kercorrp1.a12-=vol0p1*drx0*fry0; kercorrp1.a13-=vol0p1*drx0*frz0;
        kercorrp1.a21-=vol0p1*dry0*frx0; kercorrp1.a22-=vol0p1*dry0*fry0; kercorrp1.a23-=vol0p1*dry0*frz0;
        kercorrp1.a31-=vol0p1*drz0*frx0; kercorrp1.a32-=vol0p1*drz0*fry0; kercorrp1.a33-=vol0p1*drz0*frz0;
      }
      if(simulate2d){
        kercorrp1.a12=kercorrp1.a21=kercorrp1.a23=kercorrp1.a32=0.0;
        kercorrp1.a22=1.0;
      }
      kercorr[pfs1]=cumath::InverseMatrix3x3(kercorrp1);
    }
  }
}

//==============================================================================
/// Calculates the kernel correction matrix for each flexible structure particle.
/// Calcula la matriz de corrección del kernel para cada partícula de estructura flexible.
//==============================================================================
template<TpKernel tker,bool simulate2d> void CalcFlexStrucKerCorrT(const StInterParmsFlexStrucg& tfs){
  if(tfs.vnpfs){
    dim3 sgridb=GetSimpleGridSize(tfs.vnpfs,SPHBSIZE);
    KerCalcFlexStrucKerCorr<tker,simulate2d> <<<sgridb,SPHBSIZE>>>
        (tfs.vnpfs,tfs.code,tfs.flexstrucdata,tfs.flexstrucridp,tfs.poscell0,tfs.numpairs,tfs.pairidx,const_cast<tmatrix3f*>(tfs.kercorr));
  }
}

//==============================================================================
/// Calculates the kernel correction matrix for each flexible structure particle.
/// Calcula la matriz de corrección del kernel para cada partícula de estructura flexible.
//==============================================================================
template<TpKernel tker> void CalcFlexStrucKerCorr_gt0(const StInterParmsFlexStrucg& tfs){
  if(tfs.simulate2d)CalcFlexStrucKerCorrT<tker,true>  (tfs);
  else              CalcFlexStrucKerCorrT<tker,false> (tfs);
}

//==============================================================================
/// Calculates the kernel correction matrix for each flexible structure particle.
/// Calcula la matriz de corrección del kernel para cada partícula de estructura flexible.
//==============================================================================
void CalcFlexStrucKerCorr(const StInterParmsFlexStrucg& tfs){
#ifdef FAST_COMPILATION
  if(tfs.tkernel!=KERNEL_Wendland)throw "Extra kernels are disabled for FastCompilation...";
  CalcFlexStrucKerCorr_gt0<KERNEL_Wendland> (tfs);
#else
  if(tfs.tkernel==KERNEL_Wendland)     CalcFlexStrucKerCorr_gt0<KERNEL_Wendland> (tfs);
#ifndef DISABLE_KERNELS_EXTRA
  else if(tfs.tkernel==KERNEL_Cubic)   CalcFlexStrucKerCorr_gt0<KERNEL_Cubic   > (tfs);
#endif
#endif
}

//==============================================================================
/// Calculates the deformation gradient matrix for each flexible structure particle.
/// Calcula la matriz de gradiente de deformación para cada partícula de estructura flexible.
//==============================================================================
template<TpKernel tker,bool simulate2d> __global__ void KerCalcFlexStrucDefGrad(unsigned n,const float4* poscell,const typecode* code
    ,const StFlexStrucData* flexstrucdata,const unsigned* flexstrucridp
    ,const float4* poscell0,const unsigned* numpairs,const unsigned* const* pairidx,const tmatrix3f* kercorr
    ,tmatrix3f* defgrad)
{
  const unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of thread.
  if(p<n){
    const unsigned pfs1=p;      //-Number of particle.

    //-Get number of pairs for this particle.
    const unsigned numpairsp1=numpairs[pfs1];

    //-If this particle has pairs.
    if(numpairsp1){
      tmatrix3f defgradp1={0};

      //-Obtains basic data of particle p1.
      const unsigned p1=flexstrucridp[pfs1];
      const float vol0p1=flexstrucdata[CODE_GetIbodyFlexStruc(code[p1])].vol0;
      const float4 pscellp1=poscell[p1];
      const float4 pscell0p1=poscell0[pfs1];
      const tmatrix3f kercorrp1=kercorr[pfs1];

      //-Calculate deformation gradient.
      for(unsigned pair=0;pair<numpairsp1;pair++){
        const unsigned pfs2=pairidx[pfs1][pair];
        const unsigned p2=flexstrucridp[pfs2];
        const float4 pscellp2=poscell[p2];
        const float4 pscell0p2=poscell0[pfs2];
        const float drx=pscellp1.x-pscellp2.x + CTE.poscellsize*(PSCEL_GetfX(pscellp1.w)-PSCEL_GetfX(pscellp2.w));
        const float dry=pscellp1.y-pscellp2.y + CTE.poscellsize*(PSCEL_GetfY(pscellp1.w)-PSCEL_GetfY(pscellp2.w));
        const float drz=pscellp1.z-pscellp2.z + CTE.poscellsize*(PSCEL_GetfZ(pscellp1.w)-PSCEL_GetfZ(pscellp2.w));
        const float drx0=pscell0p1.x-pscell0p2.x + CTE.poscellsize*(PSCEL_GetfX(pscell0p1.w)-PSCEL_GetfX(pscell0p2.w));
        const float dry0=pscell0p1.y-pscell0p2.y + CTE.poscellsize*(PSCEL_GetfY(pscell0p1.w)-PSCEL_GetfY(pscell0p2.w));
        const float drz0=pscell0p1.z-pscell0p2.z + CTE.poscellsize*(PSCEL_GetfZ(pscell0p1.w)-PSCEL_GetfZ(pscell0p2.w));
        const float rr20=drx0*drx0+dry0*dry0+drz0*drz0;
        const float fac0=cufsph::GetKernel_Fac<tker>(rr20);
        const float frx0=fac0*drx0,fry0=fac0*dry0,frz0=fac0*drz0; //-Gradients.
        defgradp1.a11-=vol0p1*drx*frx0; defgradp1.a12-=vol0p1*drx*fry0; defgradp1.a13-=vol0p1*drx*frz0;
        defgradp1.a21-=vol0p1*dry*frx0; defgradp1.a22-=vol0p1*dry*fry0; defgradp1.a23-=vol0p1*dry*frz0;
        defgradp1.a31-=vol0p1*drz*frx0; defgradp1.a32-=vol0p1*drz*fry0; defgradp1.a33-=vol0p1*drz*frz0;
      }
      if(simulate2d){
        defgradp1.a12=defgradp1.a21=defgradp1.a23=defgradp1.a32=0.0;
        defgradp1.a22=1.0;
      }
      defgrad[pfs1]=cumath::MulMatrix3x3(defgradp1,kercorrp1);
    }
  }
}

//==============================================================================
/// Calculates the deformation gradient matrix for each flexible structure particle.
/// Calcula la matriz de gradiente de deformación para cada partícula de estructura flexible.
//==============================================================================
template<TpKernel tker,bool simulate2d> void CalcFlexStrucDefGradT(const StInterParmsFlexStrucg& tfs){
  if(tfs.vnpfs){
    dim3 sgridb=GetSimpleGridSize(tfs.vnpfs,SPHBSIZE);
    KerCalcFlexStrucDefGrad<tker,simulate2d> <<<sgridb,SPHBSIZE,0,tfs.stm>>>
        (tfs.vnpfs,tfs.poscell,tfs.code,tfs.flexstrucdata,tfs.flexstrucridp,tfs.poscell0,tfs.numpairs,tfs.pairidx,tfs.kercorr,tfs.defgrad);
  }
}

//==============================================================================
/// Calculates the deformation gradient matrix for each flexible structure particle.
/// Calcula la matriz de gradiente de deformación para cada partícula de estructura flexible.
//==============================================================================
template<TpKernel tker> void CalcFlexStrucDefGrad_ct0(const StInterParmsFlexStrucg& tfs){
  if(tfs.simulate2d)CalcFlexStrucDefGradT<tker,true>  (tfs);
  else              CalcFlexStrucDefGradT<tker,false> (tfs);
}

//==============================================================================
/// Calculates the surface normal for each flexible structure particle.
/// Calcula la superficie normal para cada partícula de estructura flexible.
//==============================================================================
__global__ void KerCalcFlexStrucNormals(unsigned n,const unsigned* flexstrucridp
    ,const tmatrix3f* defgrad,const float3* boundnor0
    ,float3* boundnor)
{
  const unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of thread.
  if(p<n){
    const float3 boundnor0p=boundnor0[p];
    if(boundnor0p.x!=0 || boundnor0p.y!=0 || boundnor0p.z!=0){
      const tmatrix3f rot=cumath::InverseMatrix3x3(cumath::TrasMatrix3x3(defgrad[p]));
      float3 boundnornewp;
      boundnornewp.x=rot.a11*boundnor0p.x+rot.a12*boundnor0p.y+rot.a13*boundnor0p.z;
      boundnornewp.y=rot.a21*boundnor0p.x+rot.a22*boundnor0p.y+rot.a23*boundnor0p.z;
      boundnornewp.z=rot.a31*boundnor0p.x+rot.a32*boundnor0p.y+rot.a33*boundnor0p.z;
      const float mag0=sqrt(boundnor0p.x*boundnor0p.x+boundnor0p.y*boundnor0p.y+boundnor0p.z*boundnor0p.z);
      const float magnew=sqrt(boundnornewp.x*boundnornewp.x+boundnornewp.y*boundnornewp.y+boundnornewp.z*boundnornewp.z);
      boundnornewp.x*=mag0/magnew; boundnornewp.y*=mag0/magnew; boundnornewp.z*=mag0/magnew;
      boundnor[flexstrucridp[p]]=boundnornewp;
    }
  }
}

//==============================================================================
/// Updates the geometric information for each flexible structure particle.
/// Actualiza la información geométrica de cada partícula de estructura flexible.
//==============================================================================
void UpdateFlexStrucGeometry(const StInterParmsFlexStrucg& tfs){
#ifdef FAST_COMPILATION
  if(tfs.tkernel!=KERNEL_Wendland)throw "Extra kernels are disabled for FastCompilation...";
  CalcFlexStrucDefGrad_ct0<KERNEL_Wendland> (tfs);
#else
  if(tfs.tkernel==KERNEL_Wendland)  CalcFlexStrucDefGrad_ct0<KERNEL_Wendland>(tfs);
#ifndef DISABLE_KERNELS_EXTRA
  else if(tfs.tkernel==KERNEL_Cubic)CalcFlexStrucDefGrad_ct0<KERNEL_Cubic>   (tfs);
#endif
#endif
  if(tfs.vnpfs&&tfs.mdbc2>=MDBC2_Std){
    dim3 sgridb=GetSimpleGridSize(tfs.vnpfs,SPHBSIZE);
    KerCalcFlexStrucNormals <<<sgridb,SPHBSIZE,0,tfs.stm>>>(tfs.vnpfs,tfs.flexstrucridp,tfs.defgrad,tfs.boundnor0,tfs.boundnor);
  }
}

//==============================================================================
/// Calculates the PK1 stress matrix for each flexible structure particle.
/// Calcula la matriz de tensión PK1 para cada partícula de estructura flexible.
//==============================================================================
__device__ tmatrix3f KerCalcFlexStrucPK1Stress(const tmatrix3f& defgrad,const tmatrix6f& cmat)
{
  //-Calculate Green-Lagrange strain from deformation gradient.
  tmatrix3f gl=cumath::MulMatrix3x3(cumath::TrasMatrix3x3(defgrad),defgrad);
  gl.a11-=1.0f; gl.a22-=1.0f; gl.a33-=1.0f;
  gl.a11*=0.5f; gl.a12*=0.5f; gl.a13*=0.5f;
  gl.a21*=0.5f; gl.a22*=0.5f; gl.a23*=0.5f;
  gl.a31*=0.5f; gl.a32*=0.5f; gl.a33*=0.5f;
  //-Convert to Voigt notation.
  const float gl_1=gl.a11;
  const float gl_2=gl.a22;
  const float gl_3=gl.a33;
  const float gl_4=gl.a23+gl.a32;
  const float gl_5=gl.a31+gl.a13;
  const float gl_6=gl.a12+gl.a21;
  //-Multiply with stiffness tensor to get PK2 stress in Voigt form.
  const float pk2_1=cmat.a11*gl_1+cmat.a12*gl_2+cmat.a13*gl_3+cmat.a14*gl_4+cmat.a15*gl_5+cmat.a16*gl_6;
  const float pk2_2=cmat.a21*gl_1+cmat.a22*gl_2+cmat.a23*gl_3+cmat.a24*gl_4+cmat.a25*gl_5+cmat.a26*gl_6;
  const float pk2_3=cmat.a31*gl_1+cmat.a32*gl_2+cmat.a33*gl_3+cmat.a34*gl_4+cmat.a35*gl_5+cmat.a36*gl_6;
  const float pk2_4=cmat.a41*gl_1+cmat.a42*gl_2+cmat.a43*gl_3+cmat.a44*gl_4+cmat.a45*gl_5+cmat.a46*gl_6;
  const float pk2_5=cmat.a51*gl_1+cmat.a52*gl_2+cmat.a53*gl_3+cmat.a54*gl_4+cmat.a55*gl_5+cmat.a56*gl_6;
  const float pk2_6=cmat.a61*gl_1+cmat.a62*gl_2+cmat.a63*gl_3+cmat.a64*gl_4+cmat.a65*gl_5+cmat.a66*gl_6;
  //-Convert PK2 stress back to normal notation.
  const tmatrix3f pk2={pk2_1,pk2_6,pk2_5,pk2_6,pk2_2,pk2_4,pk2_5,pk2_4,pk2_3};
  //-Convert PK2 to PK1 and return.
  return cumath::MulMatrix3x3(defgrad,pk2);
}

//==============================================================================
/// Interaction forces for the flexible structure particles.
/// Fuerzas de interacción para las partículas de estructura flexible.
//==============================================================================
template<TpKernel tker,TpVisco tvisco,TpMdbc2Mode mdbc2> __global__ void KerInteractionForcesFlexStruc(unsigned n,float visco
    ,int scelldiv,int4 nc,int3 cellzero,const int2* beginendcellfluid,const unsigned* dcell
    ,const float4* poscell,const float4* velrhop,const typecode* code
    ,const byte* boundmode,const float3* tangenvel
    ,const StFlexStrucData* flexstrucdata,const unsigned* flexstrucridp
    ,const float4* poscell0,const unsigned* numpairs,const unsigned* const* pairidx,const tmatrix3f* kercorr,const tmatrix3f* defgrad
    ,float* flexstrucdt,float3* ace)
{
  const unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of thread.
  if(p<n){
    const unsigned pfs1=p;      //-Number of particle.
    const unsigned p1=flexstrucridp[pfs1];

    //-Get codep1.
    const typecode codep1=code[p1];
    if(CODE_IsFlexStrucFlex(codep1)){
      float3 acep1=make_float3(0,0,0);

      //-Obtains basic data of particle p1.
      const float4 pscellp1=poscell[p1];
      float4 velrhop1=velrhop[p1];
      const float pressp1=cufsph::ComputePressCte(velrhop1.w);
      const float4 pscell0p1=poscell0[pfs1];
      const tmatrix3f kercorrp1=kercorr[pfs1];
      const tmatrix3f defgradp1=defgrad[pfs1];

      //-Obtains flexible structure data.
      const float vol0p1=flexstrucdata[CODE_GetIbodyFlexStruc(codep1)].vol0;
      const float rho0p1=flexstrucdata[CODE_GetIbodyFlexStruc(codep1)].rho0;
      const float youngmod=flexstrucdata[CODE_GetIbodyFlexStruc(codep1)].youngmod;
      const float poissratio=flexstrucdata[CODE_GetIbodyFlexStruc(codep1)].poissratio;
      const float hgfactor=flexstrucdata[CODE_GetIbodyFlexStruc(codep1)].hgfactor;
      const tmatrix6f cmat=flexstrucdata[CODE_GetIbodyFlexStruc(codep1)].cmat;
      const tmatrix3f pk1p1=KerCalcFlexStrucPK1Stress(defgradp1,cmat);
      const tmatrix3f pk1kercorrp1=cumath::MulMatrix3x3(pk1p1,kercorrp1);

      //-Get current mass of flexible structure particle.
      const float mass0p1=vol0p1*rho0p1;
      const float rhop1=rho0p1/cumath::Determinant3x3(defgradp1);

      //-Calculate structural speed of sound.
      const float csp1=sqrt(youngmod*(1.0-poissratio)/(rhop1*(1.0+poissratio)*(1.0-2.0*poissratio)));

      //-Modify values if using mDBC.
      float massp2=CTE.massf;
      if(mdbc2>=MDBC2_Std){
        if(boundmode[p1]==BMODE_MDBC2OFF)massp2=0;
        const float3 tangentvelp1=tangenvel[p1];
        velrhop1.x=tangentvelp1.x;
        velrhop1.y=tangentvelp1.y;
        velrhop1.z=tangentvelp1.z;
      }

      //-Obtains neighborhood search limits.
      int ini1,fin1,ini2,fin2,ini3,fin3;
      cunsearch::InitCte(dcell[p1],scelldiv,nc,cellzero,ini1,fin1,ini2,fin2,ini3,fin3);

      //-Flexible structure-Fluid interaction.
      for(int c3=ini3;c3<fin3;c3+=nc.w){
        for(int c2=ini2;c2<fin2;c2+=nc.x){
          unsigned pini,pfin=0;  cunsearch::ParticleRange(c2,c3,ini1,fin1,beginendcellfluid,pini,pfin);
          if(pfin){
            for(int p2=pini;p2<pfin;p2++){
              if(CODE_IsFluid(code[p2])){
                const float4 pscellp2=poscell[p2];
                const float drx=pscellp1.x-pscellp2.x+CTE.poscellsize*(PSCEL_GetfX(pscellp1.w)-PSCEL_GetfX(pscellp2.w));
                const float dry=pscellp1.y-pscellp2.y+CTE.poscellsize*(PSCEL_GetfY(pscellp1.w)-PSCEL_GetfY(pscellp2.w));
                const float drz=pscellp1.z-pscellp2.z+CTE.poscellsize*(PSCEL_GetfZ(pscellp1.w)-PSCEL_GetfZ(pscellp2.w));
                const float rr2=drx*drx+dry*dry+drz*drz;
                if(rr2<=CTE.kernelsize2&&rr2>=ALMOSTZERO){
                  //-Computes kernel.
                  const float fac=cufsph::GetKernel_Fac<tker>(rr2);
                  const float frx=fac*drx,fry=fac*dry,frz=fac*drz; //-Gradients.
                  //-Obtains velocity of particle p2 and compute difference.
                  const float4 velrhop2=velrhop[p2];
                  const float dvx=velrhop1.x-velrhop2.x,dvy=velrhop1.y-velrhop2.y,dvz=velrhop1.z-velrhop2.z;
                  //-Pressure derivative (Momentum equation).
                  const float pressp2=cufsph::ComputePressCte(velrhop2.w);
                  const float prs=(pressp1+pressp2)/(velrhop1.w*velrhop2.w)+(tker==KERNEL_Cubic?cufsph::GetKernelCubic_Tensil(rr2,velrhop1.w,pressp1,velrhop2.w,pressp2):0);
                  const float p_vpm=-prs*massp2*(CTE.massf/mass0p1);
                  acep1.x+=p_vpm*frx; acep1.y+=p_vpm*fry; acep1.z+=p_vpm*frz;                  
                  //-Artificial viscosity.
                  if(tvisco==VISCO_Artificial){
                    const float dot=drx*dvx+dry*dvy+drz*dvz;
                    if(dot<0){
                      const float dot_rr2=dot/(rr2+CTE.eta2);
                      const float amubar=CTE.kernelh*dot_rr2;
                      const float robar=(velrhop1.w+velrhop2.w)*0.5f;
                      const float pi_visc=(-visco*CTE.cs0*amubar/robar)*massp2*(CTE.massf/mass0p1);
                      acep1.x-=pi_visc*frx; acep1.y-=pi_visc*fry; acep1.z-=pi_visc*frz;
                    }
                  }
                  //-Laminar viscosity.
                  else{
                    const float robar2=(velrhop1.w+velrhop2.w);
                    const float temp=4.f*visco/((rr2+CTE.eta2)*robar2);
                    const float vtemp=massp2*temp*(drx*frx+dry*fry+drz*frz)*(CTE.massf/mass0p1);
                    acep1.x+=vtemp*dvx; acep1.y+=vtemp*dvy; acep1.z+=vtemp*dvz;
                  }
                }
              }
            }
          }
        }
      }

      //-Loop through pairs and calculate forces.
      for(unsigned pair=0;pair<numpairs[pfs1];pair++){
        const unsigned pfs2=pairidx[pfs1][pair];
        const float4 pscell0p2=poscell0[pfs2];
        float drx0=pscell0p1.x-pscell0p2.x+CTE.poscellsize*(PSCEL_GetfX(pscell0p1.w)-PSCEL_GetfX(pscell0p2.w));
        float dry0=pscell0p1.y-pscell0p2.y+CTE.poscellsize*(PSCEL_GetfY(pscell0p1.w)-PSCEL_GetfY(pscell0p2.w));
        float drz0=pscell0p1.z-pscell0p2.z+CTE.poscellsize*(PSCEL_GetfZ(pscell0p1.w)-PSCEL_GetfZ(pscell0p2.w));
        const float rr20=drx0*drx0+dry0*dry0+drz0*drz0;
        const float fac0=cufsph::GetKernel_Fac<tker>(rr20);
        const float frx0=fac0*drx0,fry0=fac0*dry0,frz0=fac0*drz0; //-Gradients.
        //-Acceleration due to structure.
        const tmatrix3f defgradp2=defgrad[pfs2];
        const tmatrix3f pk1p2=KerCalcFlexStrucPK1Stress(defgradp2,cmat);
        const tmatrix3f kercorrp2=kercorr[pfs2];
        const tmatrix3f pk1kercorrp2=cumath::MulMatrix3x3(pk1p2,kercorrp2);
        tmatrix3f pk1kercorrp1p2;
        pk1kercorrp1p2.a11=pk1kercorrp1.a11+pk1kercorrp2.a11; pk1kercorrp1p2.a12=pk1kercorrp1.a12+pk1kercorrp2.a12; pk1kercorrp1p2.a13=pk1kercorrp1.a13+pk1kercorrp2.a13;
        pk1kercorrp1p2.a21=pk1kercorrp1.a21+pk1kercorrp2.a21; pk1kercorrp1p2.a22=pk1kercorrp1.a22+pk1kercorrp2.a22; pk1kercorrp1p2.a23=pk1kercorrp1.a23+pk1kercorrp2.a23;
        pk1kercorrp1p2.a31=pk1kercorrp1.a31+pk1kercorrp2.a31; pk1kercorrp1p2.a32=pk1kercorrp1.a32+pk1kercorrp2.a32; pk1kercorrp1p2.a33=pk1kercorrp1.a33+pk1kercorrp2.a33;
        float3 pk1kercorrdw;
        pk1kercorrdw.x=pk1kercorrp1p2.a11*frx0+pk1kercorrp1p2.a12*fry0+pk1kercorrp1p2.a13*frz0;
        pk1kercorrdw.y=pk1kercorrp1p2.a21*frx0+pk1kercorrp1p2.a22*fry0+pk1kercorrp1p2.a23*frz0;
        pk1kercorrdw.z=pk1kercorrp1p2.a31*frx0+pk1kercorrp1p2.a32*fry0+pk1kercorrp1p2.a33*frz0;
        acep1.x+=pk1kercorrdw.x*vol0p1/rho0p1; acep1.y+=pk1kercorrdw.y*vol0p1/rho0p1; acep1.z+=pk1kercorrdw.z*vol0p1/rho0p1;
        //-Hourglass correction.
        if(hgfactor){
          const unsigned p2=flexstrucridp[pfs2];
          const float4 pscellp2=poscell[p2];
          float drx=pscellp1.x-pscellp2.x + CTE.poscellsize*(PSCEL_GetfX(pscellp1.w)-PSCEL_GetfX(pscellp2.w));
          float dry=pscellp1.y-pscellp2.y + CTE.poscellsize*(PSCEL_GetfY(pscellp1.w)-PSCEL_GetfY(pscellp2.w));
          float drz=pscellp1.z-pscellp2.z + CTE.poscellsize*(PSCEL_GetfZ(pscellp1.w)-PSCEL_GetfZ(pscellp2.w));
          const float rr2=drx*drx+dry*dry+drz*drz;
          const float wab=cufsph::GetKernel_Wab<tker>(rr2);
          const float xijbarx=defgradp1.a11*drx0+defgradp1.a12*dry0+defgradp1.a13*drz0;
          const float xijbary=defgradp1.a21*drx0+defgradp1.a22*dry0+defgradp1.a23*drz0;
          const float xijbarz=defgradp1.a31*drx0+defgradp1.a32*dry0+defgradp1.a33*drz0;
          const float xjibarx=-(defgradp2.a11*drx0+defgradp2.a12*dry0+defgradp2.a13*drz0);
          const float xjibary=-(defgradp2.a21*drx0+defgradp2.a22*dry0+defgradp2.a23*drz0);
          const float xjibarz=-(defgradp2.a31*drx0+defgradp2.a32*dry0+defgradp2.a33*drz0);
          const float3 epsij=make_float3(drx-xijbarx,dry-xijbary,drz-xijbarz);
          const float3 epsji=make_float3(-drx-xjibarx,-dry-xjibary,-drz-xjibarz);
          const float rr=sqrt(rr2);
          const float deltaij=(epsij.x*drx+epsij.y*dry+epsij.z*drz)/rr;
          const float deltaji=-(epsji.x*drx+epsji.y*dry+epsji.z*drz)/rr;
          const float mulfac=(hgfactor*vol0p1*vol0p1*wab*youngmod/(rr20*rr*mass0p1)*0.5f*(deltaij+deltaji));
          acep1.x-=mulfac*drx; acep1.y-=mulfac*dry; acep1.z-=mulfac*drz;
        }
      }

      //-Store results.
      if(acep1.x||acep1.y||acep1.z||csp1){
        float3 r=ace[p1]; r.x+=acep1.x; r.y+=acep1.y; r.z+=acep1.z; ace[p1]=r;
        if(csp1>flexstrucdt[pfs1])flexstrucdt[pfs1]=csp1;
      }
    }
  }
}

//==============================================================================
/// Interaction forces for the flexible structure particles.
/// Fuerzas de interacción para las partículas de estructura flexible.
//==============================================================================
template<TpKernel tker,TpVisco tvisco,TpMdbc2Mode mdbc2> void Interaction_ForcesFlexStrucT(const StInterParmsFlexStrucg& tfs){
  if(tfs.vnpfs){
    const StDivDataGpu& dvd=tfs.divdatag;
    dim3 sgridb=GetSimpleGridSize(tfs.vnpfs,SPHBSIZE);
    KerInteractionForcesFlexStruc<tker,tvisco,mdbc2> <<<sgridb,SPHBSIZE,0,tfs.stm>>>
        (tfs.vnpfs,tfs.viscob,dvd.scelldiv,dvd.nc,dvd.cellzero,dvd.beginendcell+dvd.cellfluid,tfs.dcell,tfs.poscell,tfs.velrhop,tfs.code,tfs.boundmode,tfs.tangenvel,tfs.flexstrucdata,tfs.flexstrucridp,tfs.poscell0,tfs.numpairs,tfs.pairidx,tfs.kercorr,tfs.defgrad,tfs.flexstrucdt,tfs.ace);
  }
}

//==============================================================================
/// Interaction forces for the flexible structure particles.
/// Fuerzas de interacción para las partículas de estructura flexible.
//==============================================================================
template<TpKernel tker,TpVisco tvisco> void Interaction_ForcesFlexStruc_gt1(const StInterParmsFlexStrucg& tfs){
#ifdef FAST_COMPILATION
  if(tfs.mdbc2!=MDBC2_None)throw "Extra mDBC options are disabled for FastCompilation...";
  Interaction_ForcesFlexStrucT<tker,tvisco,MDBC2_None>(tfs);
#else
  if(tfs.mdbc2==MDBC2_None)      Interaction_ForcesFlexStrucT<tker,tvisco,MDBC2_None> (tfs);
  else if(tfs.mdbc2==MDBC2_Std)  Interaction_ForcesFlexStrucT<tker,tvisco,MDBC2_Std>  (tfs);
  else if(tfs.mdbc2==MDBC2_NoPen)Interaction_ForcesFlexStrucT<tker,tvisco,MDBC2_NoPen>(tfs);
#endif
}

//==============================================================================
/// Interaction forces for the flexible structure particles.
/// Fuerzas de interacción para las partículas de estructura flexible.
//==============================================================================
template<TpKernel tker> void Interaction_ForcesFlexStruc_gt0(const StInterParmsFlexStrucg& tfs){
#ifdef FAST_COMPILATION
  if(tfs.tvisco!=VISCO_Artificial)throw "Extra viscosity options are disabled for FastCompilation...";
  Interaction_ForcesFlexStruc_gt1<tker,VISCO_Artificial> (tfs);
#else
  if(tfs.tvisco==VISCO_Artificial)     Interaction_ForcesFlexStruc_gt1<tker,VISCO_Artificial>(tfs);
  else if(tfs.tvisco==VISCO_Laminar)   Interaction_ForcesFlexStruc_gt1<tker,VISCO_Laminar>   (tfs);
  else if(tfs.tvisco==VISCO_LaminarSPS)Interaction_ForcesFlexStruc_gt1<tker,VISCO_LaminarSPS>(tfs);
#endif
}

//==============================================================================
/// Interaction forces for the flexible structure particles.
/// Fuerzas de interacción para las partículas de estructura flexible.
//==============================================================================
void Interaction_ForcesFlexStruc(const StInterParmsFlexStrucg& tfs){
#ifdef FAST_COMPILATION
  if(tfs.tkernel!=KERNEL_Wendland)throw "Extra kernels are disabled for FastCompilation...";
  Interaction_ForcesFlexStruc_gt0<KERNEL_Wendland> (tfs);
#else
  if(tfs.tkernel==KERNEL_Wendland)  Interaction_ForcesFlexStruc_gt0<KERNEL_Wendland>(tfs);
#ifndef DISABLE_KERNELS_EXTRA
  else if(tfs.tkernel==KERNEL_Cubic)Interaction_ForcesFlexStruc_gt0<KERNEL_Cubic>   (tfs);
#endif
#endif
}

//==============================================================================
/// Updates particle position according to displacement.
/// Actualizacion de posicion de particulas segun desplazamiento.
//==============================================================================
__global__ void KerComputeStepPosFlexStruc(unsigned n,const unsigned* flexstrucridp
    ,const double2* posxypre,const double* poszpre,const double2* movxy,const double* movz
    ,double2* posxy,double* posz,unsigned* dcell,typecode* code)
{
  unsigned pt=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(pt<n){
    unsigned p=flexstrucridp[pt];
    const typecode rcode=code[p];
    const bool outrhop=CODE_IsOutRho(rcode);
    const bool flexstruc=CODE_IsFlexStrucFlex(rcode);
    const bool normal=(outrhop || CODE_IsNormal(rcode));
    if(normal && flexstruc){
      const double2 rmovxy=movxy[p];
      KerUpdatePos<false>(posxypre[p],poszpre[p],rmovxy.x,rmovxy.y,movz[p],outrhop,p,posxy,posz,dcell,code);  // Periodic not implemented yet.
    }
  }
}

//==============================================================================
/// Updates particle position according to displacement.
/// Actualizacion de posicion de particulas segun desplazamiento.
//==============================================================================
void ComputeStepPosFlexStruc(unsigned npfs,const unsigned* flexstrucridp
    ,const double2* posxypre,const double* poszpre,const double2* movxy,const double* movz
    ,double2* posxy,double* posz,unsigned* dcell,typecode* code)
{
  if(npfs){
    dim3 sgrid=GetSimpleGridSize(npfs,SPHBSIZE);
    KerComputeStepPosFlexStruc <<<sgrid,SPHBSIZE>>> (npfs,flexstrucridp,posxypre,poszpre,movxy,movz,posxy,posz,dcell,code);
  }
}

//==============================================================================
/// Checks if any issues with FlexStruc particle update.
/// Comprueba si hay algún problema con la actualización de partículas FlexStruc.
//==============================================================================
bool FlexStrucStepIsValid(unsigned npb,const typecode* code){
  if(npb){
    thrust::device_ptr<const typecode> dev_code(code);
    return !thrust::any_of(dev_code,dev_code+npb,FlexStrucAnyIsOut());
  }
  return true;
}
//<vs_flexstruc_end>

}


//##############################################################################
//# Kernels for InOut (JSphInOut).
//# Kernels para InOut (JSphInOut).
//##############################################################################
// #include "JSphGpu_InOut_iker.cu"

namespace cusphinout{
#include "FunctionsBasic_iker.h"
#include "FunctionsMath_iker.h"
#include "FunSphKernel_iker.h"
#include "FunctionsGeo3d_iker.h"

#undef _JCellSearch_iker_
#include "JCellSearch_iker.h"

//##############################################################################
//# Kernels for inlet/outlet (JSphInOut).
//# Kernels para inlet/outlet (JSphInOut).
//##############################################################################

//------------------------------------------------------------------------------
/// Mark special fluid particles to ignore.
/// Marca las particulas fluidas especiales para ignorar.
//------------------------------------------------------------------------------
__global__ void KerInOutIgnoreFluidDef(unsigned n,typecode cod,typecode codnew
  ,typecode* code)
{
  const unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    if(code[p]==cod)code[p]=codnew;
  }
}

//==============================================================================
/// Mark special fluid particles to ignore.
/// Marca las particulas fluidas especiales para ignorar.
//==============================================================================
void InOutIgnoreFluidDef(unsigned n,typecode cod,typecode codnew,typecode* code){
  if(n){
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    KerInOutIgnoreFluidDef <<<sgrid,SPHBSIZE>>> (n,cod,codnew,code);
  }
}


//------------------------------------------------------------------------------
/// Returns original position of periodic particle.
//------------------------------------------------------------------------------
__device__ double3 KerInteraction_PosNoPeriodic(double3 posp1)
{
  if(CTE.periactive&1){//-xperi
    if(posp1.x<CTE.maprealposminx)                 { posp1.x-=CTE.xperincx; posp1.y-=CTE.xperincy; posp1.z-=CTE.xperincz; }
    if(posp1.x>CTE.maprealposminx+CTE.maprealsizex){ posp1.x+=CTE.xperincx; posp1.y+=CTE.xperincy; posp1.z+=CTE.xperincz; }
  }
  if(CTE.periactive&2){//-yperi
    if(posp1.y<CTE.maprealposminy)                 { posp1.x-=CTE.yperincx; posp1.y-=CTE.yperincy; posp1.z-=CTE.yperincz; }
    if(posp1.y>CTE.maprealposminy+CTE.maprealsizey){ posp1.x+=CTE.yperincx; posp1.y+=CTE.yperincy; posp1.z+=CTE.yperincz; }
  }
  if(CTE.periactive&4){//-zperi
    if(posp1.z<CTE.maprealposminz)                 { posp1.x-=CTE.zperincx; posp1.y-=CTE.zperincy; posp1.z-=CTE.zperincz; }
    if(posp1.z>CTE.maprealposminz+CTE.maprealsizez){ posp1.x+=CTE.zperincx; posp1.y+=CTE.zperincy; posp1.z+=CTE.zperincz; }
  }
  return(posp1);
}

//------------------------------------------------------------------------------
/// Updates fluid particle position according to current position.
/// Actualizacion de posicion de particulas fluidas segun posicion actual.
//------------------------------------------------------------------------------
template<bool periactive> __global__ void KerUpdatePosFluid(unsigned n,unsigned pini
  ,double2* posxy,double* posz,unsigned* dcell,typecode* code)
{
  const unsigned pp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(pp<n){
    unsigned p=pp+pini;
    const typecode rcode=code[p];
    const bool outrhop=(CODE_GetSpecialValue(rcode)==CODE_OUTRHO);
    cusph::KerUpdatePos<periactive>(posxy[p],posz[p],0,0,0,outrhop,p,posxy,posz,dcell,code);
  }
}

//==============================================================================
/// Updates fluid particle position according to current position.
/// Actualizacion de posicion de particulas fluidas segun posicion actual.
//==============================================================================
void UpdatePosFluid(byte periactive,unsigned n,unsigned pini
  ,double2* posxy,double* posz,unsigned* dcell,typecode* code)
{
  if(n){
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    if(periactive)KerUpdatePosFluid<true>  <<<sgrid,SPHBSIZE>>> (n,pini,posxy,posz,dcell,code);
    else          KerUpdatePosFluid<false> <<<sgrid,SPHBSIZE>>> (n,pini,posxy,posz,dcell,code);
  }
}



//------------------------------------------------------------------------------
/// Creates list with current inout particles (normal and periodic).
//------------------------------------------------------------------------------
__global__ void KerInOutCreateListSimple(unsigned n,unsigned pini
  ,const typecode* code,unsigned* listp)
{
  extern __shared__ unsigned slist[];
  if(!threadIdx.x)slist[0]=0;
  __syncthreads();
  const unsigned pp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(pp<n){
    const unsigned p=pp+pini;
    const typecode rcode=code[p];
    if(CODE_IsNotOut(rcode) && CODE_IsFluidInout(rcode)){//-It includes normal and periodic particles.
      slist[atomicAdd(slist,1)+1]=p; //-Add particle in the list.
    }
  }
  __syncthreads();
  const unsigned ns=slist[0];
  __syncthreads();
  if(!threadIdx.x && ns)slist[0]=atomicAdd((listp+n),ns);
  __syncthreads();
  if(threadIdx.x<ns){
    const unsigned cp=slist[0]+threadIdx.x;
    listp[cp]=slist[threadIdx.x+1];
  }
}
//==============================================================================
/// Creates list with current inout particles (normal and periodic).
/// With stable activated reorders perioc list.
//==============================================================================
unsigned InOutCreateListSimple(bool stable,unsigned n,unsigned pini
  ,const typecode* code,unsigned* listp)
{
  unsigned count=0;
  if(n){
    //-listp size list initialized to zero.
    //-Inicializa tamanho de lista listp a cero.
    hipMemset(listp+n,0,sizeof(unsigned));
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    const unsigned smem=(SPHBSIZE+1)*sizeof(unsigned); //-All fluid particles can be in in/out area and one position for counter.
    KerInOutCreateListSimple <<<sgrid,SPHBSIZE,smem>>> (n,pini,code,listp);
    hipMemcpy(&count,listp+n,sizeof(unsigned),hipMemcpyDeviceToHost);
    //-Reorders list when stable has been activated.
    //-Reordena lista cuando stable esta activado.
    if(stable && count){ //-Does not affect results.
      thrust::device_ptr<unsigned> dev_list(listp);
      thrust::sort(dev_list,dev_list+count);
    }
  }
  return(count);
}

//------------------------------------------------------------------------------
/// Creates list with current inout particles and normal (no periodic) fluid in 
/// inlet/outlet zones (update its code).
//------------------------------------------------------------------------------
__global__ void KerInOutCreateList(unsigned n,unsigned pini
  ,byte chkinputmask,byte nzone,const byte* cfgzone,const float4* planes
  ,float3 freemin,float3 freemax
  ,const float2* boxlimit,const double2* posxy,const double* posz
  ,typecode* code,unsigned* listp)
{
  extern __shared__ unsigned slist[];
  //float* splanes=(float*)(slist+(n+1));
  if(!threadIdx.x)slist[0]=0;
  __syncthreads();
  const unsigned pp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(pp<n){
    const unsigned p=pp+pini;
    const typecode rcode=code[p];
    if(CODE_IsNormal(rcode) && CODE_IsFluid(rcode)){//-It includes only normal fluid particles (no periodic).
      bool select=CODE_IsFluidInout(rcode);//-Particles already selected as InOut.
      if(!select){//-Particulas no periodicas y no marcadas como in/out.
        const double2 rxy=posxy[p];
        const double rz=posz[p];
        if(rxy.x<=freemin.x || rxy.y<=freemin.y || rz<=freemin.z || rxy.x>=freemax.x || rxy.y>=freemax.y || rz>=freemax.z){
          byte zone=255;
          if(boxlimit!=NULL){
            for(byte cz=0;cz<nzone && zone==255;cz++)if((cfgzone[cz]&chkinputmask)!=0){
              const float2 xlim=boxlimit[cz];
              const float2 ylim=boxlimit[nzone+cz];
              const float2 zlim=boxlimit[nzone*2+cz];
              if(xlim.x<=rxy.x && rxy.x<=xlim.y && ylim.x<=rxy.y && rxy.y<=ylim.y && zlim.x<=rz && rz<=zlim.y){
                const float4 rpla=planes[cz];
                if((rpla.x*rxy.x+rpla.y*rxy.y+rpla.z*rz+rpla.w)<0)zone=byte(cz);
              }
            }
          }
          else{
            for(byte cz=0;cz<nzone && zone==255;cz++)if((cfgzone[cz]&chkinputmask)!=0){
              const float4 rpla=planes[cz];
              if((rpla.x*rxy.x+rpla.y*rxy.y+rpla.z*rz+rpla.w)<0)zone=byte(cz);
            }        
          }
          if(zone!=255){
            code[p]=CODE_ToFluidInout(rcode,zone)|CODE_TYPE_FLUID_INOUTNUM; //-Adds 16 to indicate new particle in zone.
            select=true;
          }
        }
      }
      if(select)slist[atomicAdd(slist,1)+1]=p; //-Add particle in the list.
    }
  }
  __syncthreads();
  const unsigned ns=slist[0];
  __syncthreads();
  if(!threadIdx.x && ns)slist[0]=atomicAdd((listp+n),ns);
  __syncthreads();
  if(threadIdx.x<ns){
    const unsigned cp=slist[0]+threadIdx.x;
    listp[cp]=slist[threadIdx.x+1];
  }
}

//==============================================================================
/// Creates list with current inout particles and normal (no periodic) fluid in 
/// inlet/outlet zones (update its code).
//==============================================================================
unsigned InOutCreateList(bool stable,unsigned n,unsigned pini
  ,byte chkinputmask,byte nzone,const byte* cfgzone,const float4* planes
  ,tfloat3 freemin,tfloat3 freemax
  ,const float2* boxlimit,const double2* posxy,const double* posz
  ,typecode* code,unsigned* listp)
{
  unsigned count=0;
  if(n){
    //-listp size list initialized to zero.
    //-Inicializa tamanho de lista listp a cero.
    hipMemset(listp+n,0,sizeof(unsigned));
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    const unsigned smem=(SPHBSIZE+1)*sizeof(unsigned); //-All fluid particles can be in in/out area and one position for counter.
    KerInOutCreateList <<<sgrid,SPHBSIZE,smem>>> (n,pini,chkinputmask,nzone,cfgzone
      ,planes,Float3(freemin),Float3(freemax),boxlimit,posxy,posz,code,listp);
    hipMemcpy(&count,listp+n,sizeof(unsigned),hipMemcpyDeviceToHost);
    //-Reorders list when stable has been activated.
    //-Reordena lista cuando stable esta activado.
    if(stable && count){ //-Does not affect results.
      thrust::device_ptr<unsigned> dev_list(listp);
      thrust::sort(dev_list,dev_list+count);
    }
  }
  return(count);
}


//------------------------------------------------------------------------------
/// Returns velocity according profile configuration (JSphInOutZone::TpVelProfile).
//------------------------------------------------------------------------------
__device__ float KerInOutCalcVel(byte vprof,const float4& vdata,float posz){
  float vel=0;
  if(vprof==0)vel=vdata.x;  //-InVelP_Uniform
  else if(vprof==1){        //-InVelP_Linear
    const float m=vdata.x;
    const float b=vdata.y;
    vel=m*posz+b;
  }
  else if(vprof==2){        //-InVelP_Parabolic
    const float a=vdata.x;
    const float b=vdata.y;
    const float c=vdata.z;
    vel=a*posz*posz+b*posz+c;
  }
  return(vel);
}

//------------------------------------------------------------------------------
/// Updates velocity and rho of inlet/outlet particles when it uses an 
/// analytical solution.
//------------------------------------------------------------------------------
__global__ void KerInOutSetAnalyticalData(unsigned n,const unsigned* inoutpart
  ,byte izone,byte rmode,byte vmode,byte vprof,byte refillspfull
  ,float timestep,float zsurfv,float4 veldata,float4 veldata2,float3 dirdata
  ,float coefhydro,float rhopzero,float gamma
  ,const typecode* code,const double* posz,const float* zsurfpart,float4* velrhop)
{
  const unsigned cp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(cp<n){
    const unsigned p=inoutpart[cp];
    if(izone==byte(CODE_GetIzoneFluidInout(code[p]))){
      const float zsurf=(zsurfpart? zsurfpart[cp]: zsurfv);
      const double rposz=posz[p];
      float4 rvelrhop=velrhop[p];
      //-Compute rho value.
      if(rmode==0)rvelrhop.w=rhopzero; //-InRhop_Constant
      if(rmode==1){                    //-InRhop_Hydrostatic
        const float depth=float(double(zsurf)-rposz);
        const float rh=1.f+coefhydro*depth;     //rh=1.+rhop0*(-gravity.z)*(Dp*ptdata.GetDepth(p))/vCteB;
        const float frhop=pow(rh,1.f/gamma);    //rho[id]=rhop0*pow(rh,(1./gamma));
        rvelrhop.w=rhopzero*(frhop<1.f? 1.f: frhop);//-Avoid rho lower thand rhopzero to prevent suction.
        //rvelrhop.w=rhopzero*pow(rh,1.f/gamma);  //rho[id]=rhop0*pow(rh,(1./gamma));
      }
      //-Compute velocity value.
      if(vmode<2){//-VelMode InVelM_Fixed or InVelM_Variable.
        float vel=0;
        if(!refillspfull || rposz<=zsurf){
          if(vmode==0){ //-InVelM_Fixed
            vel=KerInOutCalcVel(vprof,veldata,float(rposz));
          }
          else{ //-InVelM_Variable
            const float vel1=KerInOutCalcVel(vprof,veldata,float(rposz));
            const float vel2=KerInOutCalcVel(vprof,veldata2,float(rposz));
            const float time1=veldata.w;
            const float time2=veldata2.w;
            if(timestep<=time1 || time1==time2)vel=vel1;
            else if(timestep>=time2)vel=vel2;
            else vel=(timestep-time1)/(time2-time1)*(vel2-vel1)+vel1;
          }
        }
        rvelrhop.x=vel*dirdata.x;
        rvelrhop.y=vel*dirdata.y;
        rvelrhop.z=vel*dirdata.z;
      }
      velrhop[p]=rvelrhop;
    }
  }
}

//==============================================================================
/// Updates velocity and rho of inlet/outlet particles when it uses an 
/// analytical solution.
//==============================================================================
void InOutSetAnalyticalData(unsigned n,const unsigned* inoutpart
  ,byte izone,byte rmode,byte vmode,byte vprof,byte refillspfull
  ,float timestep,float zsurfv,tfloat4 veldata,tfloat4 veldata2,tfloat3 dirdata
  ,float coefhydro,float rhopzero,float gamma
  ,const typecode* code,const double* posz,const float* zsurfpart,float4* velrhop)
{
  if(n){
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    KerInOutSetAnalyticalData <<<sgrid,SPHBSIZE>>> (n,inoutpart,izone,rmode,vmode
      ,vprof,refillspfull,timestep,zsurfv,Float4(veldata),Float4(veldata2)
      ,Float3(dirdata),coefhydro,rhopzero,gamma,code,posz,zsurfpart,velrhop);
  }
}


//<vs_meeshdat_ini>
//------------------------------------------------------------------------------
/// Computes Zsurf of current inout particles when the zsurf is non-uniform.
//------------------------------------------------------------------------------
__global__ void KerInOutComputeZsurfPart(unsigned n,const unsigned* inoutpart,byte izone
  ,double pla_a,double pla_b,double pla_d,unsigned nptx,const float* zsurfdata
  ,const typecode* code,const double2* posxy,const double* posz
  ,float* zsurfpart,byte* zsurfok)
{
  const unsigned cp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(cp<n){
    const unsigned p=inoutpart[cp];
    if(izone==byte(code[p]&CODE_TYPE_FLUID_INOUT015MASK)){//-Substract 16 to obtain the actual zone (0-15).
      const double2 pxy=posxy[p];
      const float dx=float(pla_a*pxy.x+pla_b*pxy.y+pla_d);//-It is simpliefied since z is always zero.
      const unsigned cx=(dx<=0? 0: unsigned(dx));
      const float zsurf=zsurfdata[(cx<nptx? cx: nptx-1)];
      if(zsurfpart)zsurfpart[cp]=zsurf;
      if(zsurfok)zsurfok[cp]=(posz[p]<=zsurf? 1: 0);
    }
  }
}

//==============================================================================
/// Computes Zsurf of current inout particles when the zsurf is non-uniform.
//==============================================================================
void InOutComputeZsurfPart(unsigned n,const unsigned* inoutpart,byte izone
  ,tplane3d pladisx,unsigned nptx,const float* zsurfdata
  ,const typecode* code,const double2* posxy,const double* posz
  ,float* zsurfpart,byte* zsurfok)
{
  if(n){
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    KerInOutComputeZsurfPart <<<sgrid,SPHBSIZE>>> (n,inoutpart,izone
      ,pladisx.a,pladisx.b,pladisx.d,nptx,zsurfdata
      ,code,posxy,posz,zsurfpart,zsurfok);
  }
}

//------------------------------------------------------------------------------
/// Computes Zsurf of current inout particles when the zsurf is uniform.
//------------------------------------------------------------------------------
__global__ void KerInOutComputeZsurfPartSp(unsigned n,const unsigned* inoutpart
  ,byte izone,float zsurf,const typecode* code,const double* posz
  ,float* zsurfpart,byte* zsurfok)
{
  const unsigned cp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(cp<n){
    const unsigned p=inoutpart[cp];
    if(izone==byte(code[p]&CODE_TYPE_FLUID_INOUT015MASK)){//-Substract 16 to obtain the actual zone (0-15).
      if(zsurfpart)zsurfpart[cp]=zsurf;
      if(zsurfok)zsurfok[cp]=(posz[p]<=zsurf? 1: 0);
    }
  }
}

//==============================================================================
/// Computes Zsurf of current inout particles when the zsurf is uniform.
//==============================================================================
void InOutComputeZsurfPartSp(unsigned n,const unsigned* inoutpart,byte izone
  ,float zsurf,const typecode* code,const double* posz
  ,float* zsurfpart,byte* zsurfok)
{
  if(n){
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    KerInOutComputeZsurfPartSp <<<sgrid,SPHBSIZE>>> (n,inoutpart,izone
      ,zsurf,code,posz,zsurfpart,zsurfok);
  }
}

//------------------------------------------------------------------------------
/// Computes Zsurf of current inout points when the zsurf is uniform.
//------------------------------------------------------------------------------
__global__ void KerInOutComputeZsurfokPtosSp(unsigned n,byte izone,float zsurf
  ,const byte* ptzone,const double* ptposz,byte* zsurfok)
{
  const unsigned cp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(cp<n){
    if(izone==ptzone[cp]){
      zsurfok[cp]=(ptposz[cp]<=zsurf? 1: 0);
    }
  }
}

//==============================================================================
/// Computes Zsurf of current inout points when the zsurf is uniform.
//==============================================================================
void InOutComputeZsurfokPtosSp(unsigned n,byte izone,float zsurf
 ,const byte* ptzone,const double* ptposz,byte* zsurfok)
{
  if(n){
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    KerInOutComputeZsurfokPtosSp <<<sgrid,SPHBSIZE>>> (n,izone,zsurf
      ,ptzone,ptposz,zsurfok);
  }
}

//------------------------------------------------------------------------------
/// Computes Zsurf of current inout points when the zsurf is non-uniform.
//------------------------------------------------------------------------------
__global__ void KerInOutComputeZsurfokPtos(unsigned n,byte izone
  ,double pla_a,double pla_b,double pla_d,unsigned nptx,const float* zsurfdata
  ,const byte* ptzone,const double2* ptposxy,const double* ptposz,byte* zsurfok)
{
  const unsigned cp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(cp<n){
    if(izone==ptzone[cp]){
      const double2 pxy=ptposxy[cp];
      const float dx=float(pla_a*pxy.x+pla_b*pxy.y+pla_d);//-It is simpliefied since z is always zero.
      const unsigned cx=(dx<=0? 0: unsigned(dx));
      const float zsurf=zsurfdata[(cx<nptx? cx: nptx-1)];
      zsurfok[cp]=(ptposz[cp]<=zsurf? 1: 0);
    }
  }
}

//==============================================================================
/// Computes Zsurf of current inout points when the zsurf is non-uniform.
//==============================================================================
void InOutComputeZsurfokPtos(unsigned n,byte izone,tplane3d pladisx,unsigned nptx
  ,const float* zsurfdata,const byte* ptzone,const double2* ptposxy
  ,const double* ptposz,byte* zsurfok)
{
  if(n){
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    KerInOutComputeZsurfokPtos <<<sgrid,SPHBSIZE>>> (n,izone
      ,pladisx.a,pladisx.b,pladisx.d,nptx,zsurfdata
      ,ptzone,ptposxy,ptposz,zsurfok);
  }
}
//<vs_meeshdat_end>

//------------------------------------------------------------------------------
/// Updates velocity and rho of inlet/outlet particles when it is not extrapolated. 
/// Actualiza velocidad y densidad de particulas inlet/outlet cuando no es extrapolada.
//------------------------------------------------------------------------------
__global__ void KerInoutClearInteractionVars(unsigned n,unsigned pini
  ,const typecode* code,float3* ace,float* ar,float* viscdt,float4* shiftposfs)
{
  const unsigned pp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(pp<n){
    const unsigned p=pp+pini;
    if(CODE_IsFluidInout(code[p])){
      ace[p]=make_float3(0,0,0);
      ar[p]=0;
      viscdt[p]=0;
      if(shiftposfs)shiftposfs[p]=make_float4(0,0,0,0);
    }
  }
}

//==============================================================================
/// Updates velocity and rhop of inlet/outlet particles when it is not extrapolated. 
/// Actualiza velocidad y densidad de particulas inlet/outlet cuando no es extrapolada.
//==============================================================================
void InoutClearInteractionVars(unsigned npf,unsigned pini,const typecode* code
  ,float3* ace,float* ar,float* viscdt,float4* shiftposfs)
{
  if(npf){
    dim3 sgrid=GetSimpleGridSize(npf,SPHBSIZE);
    KerInoutClearInteractionVars <<<sgrid,SPHBSIZE>>> (npf,pini,code,ace,ar,viscdt,shiftposfs);
  }
}


//------------------------------------------------------------------------------
/// Updates velocity and rhop for M1 variable when Verlet is used. 
/// Actualiza velocidad y densidad de variable M1 cuando se usa Verlet.
//------------------------------------------------------------------------------
__global__ void KerInOutUpdateVelrhopM1(unsigned n,const int* inoutpart
  ,const float4* velrhop,float4* velrhopm1)
{
  const unsigned cp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(cp<n){
    const unsigned p=inoutpart[cp];
    velrhopm1[p]=velrhop[p];
  }
}

//==============================================================================
/// Updates velocity and rhop for M1 variable when Verlet is used. 
/// Actualiza velocidad y densidad de variable M1 cuando se usa Verlet.
//==============================================================================
void InOutUpdateVelrhopM1(unsigned n,const int* inoutpart
  ,const float4* velrhop,float4* velrhopm1)
{
  if(n){
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    KerInOutUpdateVelrhopM1 <<<sgrid,SPHBSIZE>>> (n,inoutpart,velrhop,velrhopm1);
  }
}


//------------------------------------------------------------------------------
/// Checks particle position.
/// If particle is moved to fluid zone then it changes to fluid particle and 
/// it creates a new in/out particle.
/// If particle is moved out the domain then it changes to ignore particle.
//------------------------------------------------------------------------------
__global__ void KerInOutComputeStep(unsigned n,int* inoutpart,const float4* planes
  ,const float* width,const byte* cfgupdate,const float* zsurfv,typecode codenewpart
  ,const double2* posxy,const double* posz,const byte* zsurfok
  ,typecode* code,byte* newizone)
{
  const unsigned cp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(cp<n){
    typecode cod=0;
    byte newiz=255;
    const int p=inoutpart[cp];
    const typecode rcode=code[p];
    const byte izone0=byte(CODE_GetIzoneFluidInout(rcode));
    const byte izone=(izone0&CODE_TYPE_FLUID_INOUT015MASK); //-Substract 16 to obtain the actual zone (0-15).
    const byte cfupdate=cfgupdate[izone];
    const bool refilladvan=(cfupdate&INOUT_RefillAdvanced_MASK)!=0;
    const bool refillsfull=(cfupdate&INOUT_RefillSpFull_MASK)!=0;
    const bool removeinput=(cfupdate&INOUT_RemoveInput_MASK )!=0;
    const bool removezsurf=(cfupdate&INOUT_RemoveZsurf_MASK )!=0;
    const bool converinput=(cfupdate&INOUT_ConvertInput_MASK)!=0;
    const double2 rposxy=posxy[p];
    const float rposz=float(posz[p]);
    const bool zok=(zsurfok? zsurfok[cp]: rposz<=zsurfv[izone]);
    if(izone0>=16){//-Normal fluid particle in zone inlet/outlet.
      if(removeinput || (removezsurf && !zok))cod=CODE_SetOutPos(rcode); //-Normal fluid particle in zone inlet/outlet is removed.
      else cod=(converinput? rcode^0x10: codenewpart); //-Converts to inout particle or not.
    }
    else{//-Previous inout fluid particle.
      const float displane=-cugeo::PlaneDistSign(planes[izone],float(rposxy.x),float(rposxy.y),rposz);
      if(displane>width[izone] || (removezsurf && !zok)){
        cod=CODE_SetOutIgnore(rcode); //-Particle is moved out domain.
      }
      else if(displane<0){
        cod=codenewpart;//-Inout particle changes to fluid particle.
        if(!refilladvan && (refillsfull || zok))newiz=byte(izone); //-A new particle is created.
      }
    }
    newizone[cp]=newiz;
    if(cod!=0)code[p]=cod;
  }
}

//==============================================================================
/// Checks particle position.
/// If particle is moved to fluid zone then it changes to fluid particle and 
/// it creates a new in/out particle.
/// If particle is moved out the domain then it changes to ignore particle.
//==============================================================================
void InOutComputeStep(unsigned n,int* inoutpart,const float4* planes
  ,const float* width,const byte* cfgupdate,const float* zsurfv,typecode codenewpart
  ,const double2* posxy,const double* posz,const byte* zsurfok
  ,typecode* code,byte* newizone)
{
  if(n){
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    KerInOutComputeStep <<<sgrid,SPHBSIZE>>> (n,inoutpart,planes,width,cfgupdate,zsurfv
      ,codenewpart,posxy,posz,zsurfok,code,newizone);
  }
}


//------------------------------------------------------------------------------
/// Create list for new inlet particles to create.
/// Crea lista de nuevas particulas inlet a crear.
//------------------------------------------------------------------------------
__global__ void KerInOutListCreate(unsigned n,unsigned nmax,const byte* newizone
  ,int* inoutpart)
{
  extern __shared__ unsigned slist[];
  if(!threadIdx.x)slist[0]=0;
  __syncthreads();
  const unsigned cp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(cp<n && newizone[cp]<16){
    slist[atomicAdd(slist,1)+1]=cp; 
  }
  __syncthreads();
  const unsigned ns=slist[0];
  __syncthreads();
  if(!threadIdx.x && ns)slist[0]=n + atomicAdd((inoutpart+nmax),ns);
  __syncthreads();
  if(threadIdx.x<ns){
    const unsigned cp2=slist[0]+threadIdx.x;
    if(cp2<nmax)inoutpart[cp2]=slist[threadIdx.x+1];
  }
}

//==============================================================================
/// Create list for new inlet particles to create at end of inoutpart[]. 
/// Returns number of new particles to create.
/// 
/// Crea lista de nuevas particulas inlet a crear al final de inoutpart[].
/// Devuelve el numero de las nuevas particulas para crear.
//==============================================================================
unsigned InOutListCreate(bool stable,unsigned n,unsigned nmax
  ,const byte* newizone,int* inoutpart)
{
  unsigned count=0;
  if(n){
    //-inoutpart size list initialized to zero.
    //-Inicializa tamanho de lista inoutpart a cero.
    hipMemset(inoutpart+nmax,0,sizeof(unsigned));
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    const unsigned smem=(SPHBSIZE+1)*sizeof(unsigned); //-All fluid particles can be in in/out area and one position for counter.
    KerInOutListCreate <<<sgrid,SPHBSIZE,smem>>> (n,nmax,newizone,inoutpart);
    hipMemcpy(&count,inoutpart+nmax,sizeof(unsigned),hipMemcpyDeviceToHost);
    //-Reorders list if it is valid and stable has been activated.
    //-Reordena lista si es valida y stable esta activado.
    if(stable && count && count<=nmax){
      thrust::device_ptr<unsigned> dev_list((unsigned*)inoutpart);
      thrust::sort(dev_list+n,dev_list+(n+count));
    }
  }
  return(count);
}


//------------------------------------------------------------------------------
/// Creates new inlet particles to replace the particles moved to fluid domain.
//------------------------------------------------------------------------------
template<bool periactive> __global__ void KerInOutCreateNewInlet(unsigned newn
  ,const unsigned* inoutpart,unsigned inoutcount,const byte* newizone
  ,unsigned np,unsigned idnext,typecode codenewpart,const float3* dirdata
  ,const float* width,double2* posxy,double* posz,unsigned* dcell,typecode* code
  ,unsigned* idp,float4* velrhop)
{
  const unsigned cp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(cp<newn){
    const int cp0=inoutpart[inoutcount+cp];
    const int p=inoutpart[cp0];
    const byte izone=newizone[cp0];
    const double dis=width[izone];
    const float3 rdirdata=dirdata[izone];
    double2 rposxy=posxy[p];
    double rposz=posz[p];
    rposxy.x-=dis*rdirdata.x;
    rposxy.y-=dis*rdirdata.y;
    rposz   -=dis*rdirdata.z;
    const unsigned p2=np+cp;
    code[p2]=CODE_ToFluidInout(codenewpart,izone);
    cusph::KerUpdatePos<periactive>(rposxy,rposz,0,0,0,false,p2,posxy,posz,dcell,code);
    idp[p2]=idnext+cp;
    velrhop[p2]=make_float4(0,0,0,1000);
  }
}

//==============================================================================
/// Creates new inlet particles to replace the particles moved to fluid domain.
//==============================================================================
void InOutCreateNewInlet(byte periactive,unsigned newn
  ,const unsigned* inoutpart,unsigned inoutcount,const byte* newizone
  ,unsigned np,unsigned idnext,typecode codenewpart,const float3* dirdata
  ,const float* width,double2* posxy,double* posz,unsigned* dcell,typecode* code
  ,unsigned* idp,float4* velrhop)
{
  if(newn){
    dim3 sgrid=GetSimpleGridSize(newn,SPHBSIZE);
    if(periactive)KerInOutCreateNewInlet<true>  <<<sgrid,SPHBSIZE>>> (newn,inoutpart,inoutcount,newizone,np,idnext,codenewpart,dirdata,width,posxy,posz,dcell,code,idp,velrhop);
    else          KerInOutCreateNewInlet<false> <<<sgrid,SPHBSIZE>>> (newn,inoutpart,inoutcount,newizone,np,idnext,codenewpart,dirdata,width,posxy,posz,dcell,code,idp,velrhop);
  }
}


//------------------------------------------------------------------------------
/// Move in/out particles according its velocity.
//------------------------------------------------------------------------------
template<bool periactive> __global__ void KerInOutFillMove(unsigned n
  ,const unsigned* inoutpart,double dt,const float4* velrhop
  ,double2* posxy,double* posz,unsigned* dcell,typecode* code)
{
  const unsigned cp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(cp<n){
    const unsigned p=inoutpart[cp];
    //-Updates position of particles.
    const float4 rvelrhop=velrhop[p];
    const double dx=double(rvelrhop.x)*dt;
    const double dy=double(rvelrhop.y)*dt;
    const double dz=double(rvelrhop.z)*dt;
    cusph::KerUpdatePos<periactive>(posxy[p],posz[p],dx,dy,dz,false,p,posxy,posz,dcell,code);
  }
}

//==============================================================================
/// Move particles in/out according its velocity.
//==============================================================================
void InOutFillMove(byte periactive,unsigned n,const unsigned* inoutpart
  ,double dt,const float4* velrhop
  ,double2* posxy,double* posz,unsigned* dcell,typecode* code)
{
  if(n){
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    if(periactive)KerInOutFillMove<true>  <<<sgrid,SPHBSIZE>>> (n,inoutpart,dt,velrhop,posxy,posz,dcell,code);
    else          KerInOutFillMove<false> <<<sgrid,SPHBSIZE>>> (n,inoutpart,dt,velrhop,posxy,posz,dcell,code);
  }
}


//------------------------------------------------------------------------------
/// Computes projection data to filling mode.
//------------------------------------------------------------------------------
__global__ void KerInOutFillProjection(unsigned n,const unsigned* inoutpart
  ,const byte* cfgupdate,const float4* planes,const double2* posxy
  ,const double* posz,const typecode* code,float* prodist,double2* proposxy
  ,double* proposz)
{
  const unsigned cp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(cp<n){
    const unsigned p=inoutpart[cp];
    float rprodis=0;
    double rpropx=0,rpropy=0,rpropz=0;
    //-Checks if particle was moved to fluid domain.
    const typecode rcode=code[p];
    if(CODE_IsNotOut(rcode) && CODE_IsFluidInout(rcode)){
      const byte izone=byte(CODE_GetIzoneFluidInout(rcode));
      if((cfgupdate[izone]&INOUT_RefillAdvanced_MASK)!=0){
        const double2 rposxy=posxy[p];
        const double rposz=posz[p];
        const float4 rplanes=planes[izone];
        //-Compute distance to plane.
        const double v1=rposxy.x*rplanes.x + rposxy.y*rplanes.y + rposz*rplanes.z + rplanes.w;
        const double v2=rplanes.x*rplanes.x+rplanes.y*rplanes.y+rplanes.z*rplanes.z;
        rprodis=-float(v1/sqrt(v2));//-Equivalent to cugeo::PlaneDistSign().
        //-Calculates point on plane.
        const double t=-v1/v2;
        rpropx=rposxy.x+t*rplanes.x;
        rpropy=rposxy.y+t*rplanes.y;
        rpropz=rposz+t*rplanes.z;
      }
    }
    //-Saves results on GPU memory.
    prodist[cp]=rprodis;
    proposxy[cp]=make_double2(rpropx,rpropy);
    proposz[cp] =rpropz;
  }
}

//==============================================================================
/// Computes projection data to filling mode.
//==============================================================================
void InOutFillProjection(unsigned n,const unsigned* inoutpart
  ,const byte* cfgupdate,const float4* planes,const double2* posxy
  ,const double* posz,const typecode* code,float* prodist,double2* proposxy
  ,double* proposz)
{
  if(n){
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    KerInOutFillProjection <<<sgrid,SPHBSIZE>>> (n,inoutpart,cfgupdate,planes,posxy,posz
      ,code,prodist,proposxy,proposz);
  }
}


//------------------------------------------------------------------------------
/// Compute maximum distance to create points in each PtPos.
/// Create list of selected ptpoints and its distance for new inlet/outlet particles.
//------------------------------------------------------------------------------
__global__ void KerInOutFillListCreate(unsigned npt
  ,const double2* ptposxy,const double* ptposz,const byte* zsurfok
  ,const byte* ptzone,const byte* cfgupdate,const float* zsurf,const float* width
  ,unsigned npropt,const float* prodist,const double2* proposxy,const double* proposz
  ,float dpmin,float dpmin2,float dp,float* ptdist,unsigned nmax,unsigned* inoutpart)
{
  extern __shared__ unsigned slist[];
  //float* sdist=(float*)(slist+(blockDim.x+1));
  if(!threadIdx.x)slist[0]=0;
  __syncthreads();
  const unsigned cpt=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(cpt<npt){
    float distmax=FLT_MAX;
    const byte izone=ptzone[cpt];
    if((cfgupdate[izone]&INOUT_RefillAdvanced_MASK)!=0){
      const double2 rptxy=ptposxy[cpt];
      const double rptz=ptposz[cpt];
      const bool zok=(zsurfok? zsurfok[cpt]: float(rptz)<=zsurf[izone]);
      if(zok){
        distmax=0;
        for(int cpro=0;cpro<npropt;cpro++){
          const double2 propsxy=proposxy[cpro];
          const float disx=rptxy.x-propsxy.x;
          const float disy=rptxy.y-propsxy.y;
          const float disz=rptz   -proposz [cpro];
          if(disx<=dpmin && disy<=dpmin && disz<=dpmin){//-particle near to ptpoint (approx.)
            const float dist2=(disx*disx+disy*disy+disz*disz);
            if(dist2<dpmin2){//-particle near to ptpoint.
              const float dmax=prodist[cpro]+sqrt(dpmin2-dist2);
              distmax=max(distmax,dmax);
            }
          }
        }
      }
    }
    distmax=(distmax==0? dp: distmax);
    //-Creates list of new inlet/outlet particles.
    if(distmax<width[ptzone[cpt]]){
      slist[atomicAdd(slist,1)+1]=cpt; //-Add ptpoint in the list.
      ptdist[cpt]=distmax;             //-Saves distance of ptpoint.
    }
  }
  __syncthreads();
  const unsigned ns=slist[0];
  __syncthreads();
  if(!threadIdx.x && ns)slist[0]=atomicAdd((inoutpart+nmax),ns);
  __syncthreads();
  if(threadIdx.x<ns){
    const unsigned cp2=slist[0]+threadIdx.x;
    if(cp2<nmax)inoutpart[cp2]=slist[threadIdx.x+1];
  }
}

//==============================================================================
/// Compute maximum distance to create points in each PtPos.
/// Create list of selected ptpoints and its distance for new inlet/outlet particles.
/// Returns number of new particles to create.
//==============================================================================
unsigned InOutFillListCreate(bool stable,unsigned npt
  ,const double2* ptposxy,const double* ptposz,const byte* zsurfok
  ,const byte* ptzone,const byte* cfgupdate,const float* zsurf,const float* width
  ,unsigned npropt,const float* prodist,const double2* proposxy,const double* proposz
  ,float dpmin,float dpmin2,float dp,float* ptdist,unsigned nmax,unsigned* inoutpart)
{
  unsigned count=0;
  if(npt){
    //-inoutpart size list initialized to zero.
    //-Inicializa tamanho de lista inoutpart a cero.
    hipMemset(inoutpart+nmax,0,sizeof(unsigned));
    dim3 sgrid=GetSimpleGridSize(npt,SPHBSIZE);
    const unsigned smem=(SPHBSIZE+1)*sizeof(unsigned); //-All fluid particles can be in in/out area and one position for counter.
    KerInOutFillListCreate <<<sgrid,SPHBSIZE,smem>>> (npt,ptposxy,ptposz,zsurfok
      ,ptzone,cfgupdate,zsurf,width,npropt,prodist,proposxy,proposz,dpmin,dpmin2
      ,dp,ptdist,nmax,inoutpart);
    hipMemcpy(&count,inoutpart+nmax,sizeof(unsigned),hipMemcpyDeviceToHost);
    //-Reorders list if it is valid and stable has been activated.
    //-Reordena lista si es valida y stable esta activado.
    if(stable && count && count<=nmax){
      thrust::device_ptr<unsigned> dev_list((unsigned*)inoutpart);
      thrust::sort(dev_list,dev_list+count);
    }
  }
  return(count);
}


//------------------------------------------------------------------------------
/// Creates new inlet/outlet particles to fill inlet/outlet domain.
//------------------------------------------------------------------------------
template<bool periactive> __global__ void KerInOutFillCreate(unsigned newn
  ,const unsigned* newinoutpart,const double2* ptposxy,const double* ptposz
  ,const byte* ptzone,const float* ptauxdist
  ,unsigned np,unsigned idnext,typecode codenewpart,const float3* dirdata
  ,double2* posxy,double* posz,unsigned* dcell,typecode* code,unsigned* idp
  ,float4* velrhop)
{
  const unsigned cp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(cp<newn){
    const unsigned cpt=newinoutpart[cp];
    const byte izone=ptzone[cpt];
    const double dis=ptauxdist[cpt];
    const float3 rdirdata=dirdata[izone];
    double2 rposxy=ptposxy[cpt];
    double rposz=ptposz[cpt];
    rposxy.x-=dis*rdirdata.x;
    rposxy.y-=dis*rdirdata.y;
    rposz   -=dis*rdirdata.z;
    const unsigned p=np+cp;
    code[p]=CODE_ToFluidInout(codenewpart,izone);
    cusph::KerUpdatePos<periactive>(rposxy,rposz,0,0,0,false,p,posxy,posz,dcell,code);
    idp[p]=idnext+cp;
    velrhop[p]=make_float4(0,0,0,1000);
  }
}

//==============================================================================
/// Creates new inlet/outlet particles to fill inlet/outlet domain.
//==============================================================================
void InOutFillCreate(byte periactive,unsigned newn,const unsigned* newinoutpart
  ,const double2* ptposxy,const double* ptposz,const byte* ptzone
  ,const float* ptauxdist,unsigned np,unsigned idnext,typecode codenewpart
  ,const float3* dirdata,double2* posxy,double* posz,unsigned* dcell
  ,typecode* code,unsigned* idp,float4* velrhop)
{
  if(newn){
    dim3 sgrid=GetSimpleGridSize(newn,SPHBSIZE);
    if(periactive)KerInOutFillCreate<true>  <<<sgrid,SPHBSIZE>>> (newn,newinoutpart,ptposxy,ptposz,ptzone,ptauxdist,np,idnext,codenewpart,dirdata,posxy,posz,dcell,code,idp,velrhop);
    else          KerInOutFillCreate<false> <<<sgrid,SPHBSIZE>>> (newn,newinoutpart,ptposxy,ptposz,ptzone,ptauxdist,np,idnext,codenewpart,dirdata,posxy,posz,dcell,code,idp,velrhop);
  }
}


//------------------------------------------------------------------------------
/// Perform interaction between ghost inlet/outlet nodes and fluid particles. GhostNodes-Fluid
/// Realiza interaccion entre ghost inlet/outlet nodes y particulas de fluido. GhostNodes-Fluid
//------------------------------------------------------------------------------
template<bool sim2d,TpKernel tker> __global__ void KerInteractionInOutExtrap_Double(
  unsigned inoutcount,const int* inoutpart,const byte* cfgzone
  ,byte computerhopmask,byte computevelmask,const float4* planes
  ,const float* width,const float3* dirdata,float determlimit
  ,int scelldiv,int4 nc,int3 cellzero,const int2* beginendcellfluid
  ,const double2* posxy,const double* posz,const typecode* code
  ,const unsigned* idp,float4* velrhop)
{
  const unsigned cp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(cp<inoutcount){
    const unsigned p1=inoutpart[cp];
    const byte izone=byte(CODE_GetIzoneFluidInout(code[p1]));
    const byte cfg=cfgzone[izone];
    const bool computerhop=((cfg&computerhopmask)!=0);
    const bool computevel= ((cfg&computevelmask )!=0);
    if(computerhop || computevel){
      //-Calculates ghost node position.
      double3 pos_p1=make_double3(posxy[p1].x,posxy[p1].y,posz[p1]);
      if(CODE_IsPeriodic(code[p1]))pos_p1=KerInteraction_PosNoPeriodic(pos_p1);
      const double displane=cumath::DistPlane(planes[izone],pos_p1)*2;
      const float3 rdirdata=dirdata[izone];
      const double3 posp1=make_double3(pos_p1.x+displane*rdirdata.x, pos_p1.y+displane*rdirdata.y, pos_p1.z+displane*rdirdata.z); //-Ghost node position.

      //-Initializes variables for calculation.
      double rhopp1=0;
      double3 gradrhopp1=make_double3(0,0,0);
      double3 velp1=make_double3(0,0,0);
      tmatrix3d gradvelp1; cumath::Tmatrix3dReset(gradvelp1); //-Only for velocity.
      tmatrix3d a_corr2; if(sim2d) cumath::Tmatrix3dReset(a_corr2); //-Only for 2D.
      tmatrix4d a_corr3; if(!sim2d)cumath::Tmatrix4dReset(a_corr3); //-Only for 3D.
    
      //-Obtains neighborhood search limits.
      int ini1,fin1,ini2,fin2,ini3,fin3;
      cunsearch::InitCte(posp1.x,posp1.y,posp1.z,scelldiv,nc,cellzero,ini1,fin1,ini2,fin2,ini3,fin3);

      //-Interaction with fluids.
      for(int c3=ini3;c3<fin3;c3+=nc.w)for(int c2=ini2;c2<fin2;c2+=nc.x){
        unsigned pini,pfin=0;  cunsearch::ParticleRange(c2,c3,ini1,fin1,beginendcellfluid,pini,pfin);
        if(pfin)for(unsigned p2=pini;p2<pfin;p2++){
          const double2 p2xy=posxy[p2];
          const double drx=double(posp1.x-p2xy.x);
          const double dry=double(posp1.y-p2xy.y);
          const double drz=double(posp1.z-posz[p2]);
          const double rr2=drx*drx+dry*dry+drz*drz;
          if(rr2<=CTE.kernelsize2 && rr2>=ALMOSTZERO && CODE_IsFluidNotInout(code[p2])){//-Only with fluid particles but not inout particles.
            //-Only Wendland or Cubic Spline kernel.
            //-Computes kernel.
            float fac;
            const double wab=cufsph::GetKernel_WabFac<tker>(rr2,fac);
            const double frx=drx*fac,fry=dry*fac,frz=drz*fac; //-Gradients.

            const float4 velrhopp2=velrhop[p2];
            //===== Get mass and volume of particle p2 =====
            double massp2=CTE.massf;
            double volp2=massp2/velrhopp2.w;

            //===== Density and its gradient =====
            rhopp1+=massp2*wab;
            gradrhopp1.x+=massp2*frx;
            gradrhopp1.y+=massp2*fry;
            gradrhopp1.z+=massp2*frz;

            //===== Kernel values multiplied by volume =====
            const double vwab=wab*volp2;
            const double vfrx=frx*volp2;
            const double vfry=fry*volp2;
            const double vfrz=frz*volp2;

            //===== Velocity and its gradient =====
            if(computevel){
              velp1.x+=vwab*velrhopp2.x;
              velp1.y+=vwab*velrhopp2.y;
              velp1.z+=vwab*velrhopp2.z;
              gradvelp1.a11+=vfrx*velrhopp2.x;    // du/dx
              gradvelp1.a12+=vfry*velrhopp2.x;    // du/dy
              gradvelp1.a13+=vfrz*velrhopp2.x;    // du/dz
              gradvelp1.a21+=vfrx*velrhopp2.y;    // dv/dx
              gradvelp1.a22+=vfry*velrhopp2.y;    // dv/dx
              gradvelp1.a23+=vfrz*velrhopp2.y;    // dv/dx
              gradvelp1.a31+=vfrx*velrhopp2.z;    // dw/dx
              gradvelp1.a32+=vfry*velrhopp2.z;    // dw/dx
              gradvelp1.a33+=vfrz*velrhopp2.z;    // dw/dx
            }

            //===== Matrix A for correction =====
            if(sim2d){
              a_corr2.a11+=vwab;  a_corr2.a12+=drx*vwab;  a_corr2.a13+=drz*vwab;
              a_corr2.a21+=vfrx;  a_corr2.a22+=drx*vfrx;  a_corr2.a23+=drz*vfrx;
              a_corr2.a31+=vfrz;  a_corr2.a32+=drx*vfrz;  a_corr2.a33+=drz*vfrz;
            }
            else{
              a_corr3.a11+=vwab;  a_corr3.a12+=drx*vwab;  a_corr3.a13+=dry*vwab;  a_corr3.a14+=drz*vwab;
              a_corr3.a21+=vfrx;  a_corr3.a22+=drx*vfrx;  a_corr3.a23+=dry*vfrx;  a_corr3.a24+=drz*vfrx;
              a_corr3.a31+=vfry;  a_corr3.a32+=drx*vfry;  a_corr3.a33+=dry*vfry;  a_corr3.a34+=drz*vfry;
              a_corr3.a41+=vfrz;  a_corr3.a42+=drx*vfrz;  a_corr3.a43+=dry*vfrz;  a_corr3.a44+=drz*vfrz;
            }
          }
        }
      }

      //-Store the results.
      //--------------------
      float4 velrhopfinal=velrhop[p1];
      const double3 dpos=make_double3(pos_p1.x-posp1.x, pos_p1.y-posp1.y, pos_p1.z-posp1.z); //-Inlet/outlet particle position - ghost node position.
      if(sim2d){
        const double determ=cumath::Determinant3x3(a_corr2);
        if(fabs(determ)>=determlimit){//-Use 1e-3f (first_order) or 1e+3f (zeroth_order).
          const tmatrix3d invacorr2=cumath::InverseMatrix3x3(a_corr2,determ);
          //-GHOST NODE DENSITY IS MIRRORED BACK TO THE INFLOW OR OUTFLOW PARTICLES.
          if(computerhop){
            const double rhoghost=rhopp1*invacorr2.a11 + gradrhopp1.x*invacorr2.a12 + gradrhopp1.z*invacorr2.a13;
            const double grx=-(rhopp1*invacorr2.a21 + gradrhopp1.x*invacorr2.a22 + gradrhopp1.z*invacorr2.a23);
            const double grz=-(rhopp1*invacorr2.a31 + gradrhopp1.x*invacorr2.a32 + gradrhopp1.z*invacorr2.a33);
            velrhopfinal.w=float(rhoghost + grx*dpos.x + grz*dpos.z);
          }
          //-GHOST NODE VELOCITY ARE MIRRORED BACK TO THE OUTFLOW PARTICLES.
          if(computevel){
            const double velghost_x=velp1.x*invacorr2.a11 + gradvelp1.a11*invacorr2.a12 + gradvelp1.a13*invacorr2.a13;
            const double velghost_z=velp1.z*invacorr2.a11 + gradvelp1.a31*invacorr2.a12 + gradvelp1.a33*invacorr2.a13;
            const double a11=-(velp1.x*invacorr2.a21 + gradvelp1.a11*invacorr2.a22 + gradvelp1.a13*invacorr2.a23);
            const double a13=-(velp1.z*invacorr2.a21 + gradvelp1.a31*invacorr2.a22 + gradvelp1.a33*invacorr2.a23);
            const double a31=-(velp1.x*invacorr2.a31 + gradvelp1.a11*invacorr2.a32 + gradvelp1.a13*invacorr2.a33);
            const double a33=-(velp1.z*invacorr2.a31 + gradvelp1.a31*invacorr2.a32 + gradvelp1.a33*invacorr2.a33);
            velrhopfinal.x=float(velghost_x + a11*dpos.x + a31*dpos.z);
            velrhopfinal.z=float(velghost_z + a13*dpos.x + a33*dpos.z);
            velrhopfinal.y=0;
          }
        }
        else if(a_corr2.a11>0){//-Determinant is small but a11 is nonzero, 0th order ANGELO.
          if(computerhop)velrhopfinal.w=float(rhopp1/a_corr2.a11);
          if(computevel){
            velrhopfinal.x=float(velp1.x/a_corr2.a11);
            velrhopfinal.z=float(velp1.z/a_corr2.a11);
            velrhopfinal.y=0;
          }
        }
      }
      else{
        const double determ=cumath::Determinant4x4(a_corr3);
        if(fabs(determ)>=determlimit){
          const tmatrix4d invacorr3=cumath::InverseMatrix4x4(a_corr3,determ);
          //-GHOST NODE DENSITY IS MIRRORED BACK TO THE INFLOW OR OUTFLOW PARTICLES.
          if(computerhop){
            const double rhoghost=rhopp1*invacorr3.a11 + gradrhopp1.x*invacorr3.a12 + gradrhopp1.y*invacorr3.a13 + gradrhopp1.z*invacorr3.a14;
            const double grx=   -(rhopp1*invacorr3.a21 + gradrhopp1.x*invacorr3.a22 + gradrhopp1.y*invacorr3.a23 + gradrhopp1.z*invacorr3.a24);
            const double gry=   -(rhopp1*invacorr3.a31 + gradrhopp1.x*invacorr3.a32 + gradrhopp1.y*invacorr3.a33 + gradrhopp1.z*invacorr3.a34);
            const double grz=   -(rhopp1*invacorr3.a41 + gradrhopp1.x*invacorr3.a42 + gradrhopp1.y*invacorr3.a43 + gradrhopp1.z*invacorr3.a44);
            velrhopfinal.w=float(rhoghost + grx*dpos.x + gry*dpos.y + grz*dpos.z);
          }
          //-GHOST NODE VELOCITY ARE MIRRORED BACK TO THE OUTFLOW PARTICLES.
          if(computevel){
            const double velghost_x=velp1.x*invacorr3.a11 + gradvelp1.a11*invacorr3.a12 + gradvelp1.a12*invacorr3.a13 + gradvelp1.a13*invacorr3.a14;
            const double velghost_y=velp1.y*invacorr3.a11 + gradvelp1.a21*invacorr3.a12 + gradvelp1.a22*invacorr3.a13 + gradvelp1.a23*invacorr3.a14;
            const double velghost_z=velp1.z*invacorr3.a11 + gradvelp1.a31*invacorr3.a12 + gradvelp1.a32*invacorr3.a13 + gradvelp1.a33*invacorr3.a14;
            const double a11=-(velp1.x*invacorr3.a21 + gradvelp1.a11*invacorr3.a22 + gradvelp1.a12*invacorr3.a23 + gradvelp1.a13*invacorr3.a24);
            const double a12=-(velp1.y*invacorr3.a21 + gradvelp1.a21*invacorr3.a22 + gradvelp1.a22*invacorr3.a23 + gradvelp1.a23*invacorr3.a24);
            const double a13=-(velp1.z*invacorr3.a21 + gradvelp1.a31*invacorr3.a22 + gradvelp1.a32*invacorr3.a23 + gradvelp1.a33*invacorr3.a24);
            const double a21=-(velp1.x*invacorr3.a31 + gradvelp1.a11*invacorr3.a32 + gradvelp1.a12*invacorr3.a33 + gradvelp1.a13*invacorr3.a34);
            const double a22=-(velp1.y*invacorr3.a31 + gradvelp1.a21*invacorr3.a32 + gradvelp1.a22*invacorr3.a33 + gradvelp1.a23*invacorr3.a34);
            const double a23=-(velp1.z*invacorr3.a31 + gradvelp1.a31*invacorr3.a32 + gradvelp1.a32*invacorr3.a33 + gradvelp1.a33*invacorr3.a34);
            const double a31=-(velp1.x*invacorr3.a41 + gradvelp1.a11*invacorr3.a42 + gradvelp1.a12*invacorr3.a43 + gradvelp1.a13*invacorr3.a44);
            const double a32=-(velp1.y*invacorr3.a41 + gradvelp1.a21*invacorr3.a42 + gradvelp1.a22*invacorr3.a43 + gradvelp1.a23*invacorr3.a44);
            const double a33=-(velp1.z*invacorr3.a41 + gradvelp1.a31*invacorr3.a42 + gradvelp1.a32*invacorr3.a43 + gradvelp1.a33*invacorr3.a44);
            velrhopfinal.x=float(velghost_x + a11*dpos.x + a21*dpos.y + a31*dpos.z);
            velrhopfinal.y=float(velghost_y + a12*dpos.x + a22*dpos.y + a32*dpos.z);
            velrhopfinal.z=float(velghost_z + a13*dpos.x + a23*dpos.y + a33*dpos.z);
          }
        }
        else if(a_corr3.a11>0){ // Determinant is small but a11 is nonzero, 0th order ANGELO
          if(computerhop)velrhopfinal.w=float(rhopp1/a_corr3.a11);
          if(computevel){
            velrhopfinal.x=float(velp1.x/a_corr3.a11);
            velrhopfinal.y=float(velp1.y/a_corr3.a11);
            velrhopfinal.z=float(velp1.z/a_corr3.a11);
          }
        }
      }
      velrhop[p1]=velrhopfinal;
    }
  }
}

//------------------------------------------------------------------------------
/// Perform interaction between ghost inlet/outlet nodes and fluid particles. GhostNodes-Fluid
/// Realiza interaccion entre ghost inlet/outlet nodes y particulas de fluido. GhostNodes-Fluid
//------------------------------------------------------------------------------
template<bool sim2d,TpKernel tker> __global__ void KerInteractionInOutExtrap_Single(
  unsigned inoutcount,const int* inoutpart,const byte* cfgzone
  ,byte computerhopmask,byte computevelmask
  ,const float4* planes,const float* width,const float3* dirdata,float determlimit
  ,int scelldiv,int4 nc,int3 cellzero,const int2* beginendcellfluid
  ,const double2* posxy,const double* posz,const typecode* code
  ,const unsigned* idp,float4* velrhop)
{
  const unsigned cp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(cp<inoutcount){
    const unsigned p1=inoutpart[cp];
    const byte izone=byte(CODE_GetIzoneFluidInout(code[p1]));
    const byte cfg=cfgzone[izone];
    const bool computerhop=((cfg&computerhopmask)!=0);
    const bool computevel= ((cfg&computevelmask )!=0);
    if(computerhop || computevel){
      //-Calculates ghost node position.
      double3 pos_p1=make_double3(posxy[p1].x,posxy[p1].y,posz[p1]);
      if(CODE_IsPeriodic(code[p1]))pos_p1=KerInteraction_PosNoPeriodic(pos_p1);
      const double displane=cumath::DistPlane(planes[izone],pos_p1)*2;
      const float3 rdirdata=dirdata[izone];
      const double3 posp1=make_double3(pos_p1.x+displane*rdirdata.x, pos_p1.y+displane*rdirdata.y, pos_p1.z+displane*rdirdata.z); //-Ghost node position.

      //-Initializes variables for calculation.
      float rhopp1=0;
      float3 gradrhopp1=make_float3(0,0,0);
      float3 velp1=make_float3(0,0,0);
      tmatrix3f gradvelp1; cumath::Tmatrix3fReset(gradvelp1); //-Only for velocity.
      tmatrix3d a_corr2; if(sim2d) cumath::Tmatrix3dReset(a_corr2); //-Only for 2D.
      tmatrix4d a_corr3; if(!sim2d)cumath::Tmatrix4dReset(a_corr3); //-Only for 3D.
    
      //-Obtains neighborhood search limits.
      int ini1,fin1,ini2,fin2,ini3,fin3;
      cunsearch::InitCte(posp1.x,posp1.y,posp1.z,scelldiv,nc,cellzero,ini1,fin1,ini2,fin2,ini3,fin3);

      //-Interaction with fluids.
      for(int c3=ini3;c3<fin3;c3+=nc.w)for(int c2=ini2;c2<fin2;c2+=nc.x){
        unsigned pini,pfin=0;  cunsearch::ParticleRange(c2,c3,ini1,fin1,beginendcellfluid,pini,pfin);
        if(pfin)for(unsigned p2=pini;p2<pfin;p2++){
          const double2 p2xy=posxy[p2];
          const float drx=float(posp1.x-p2xy.x);
          const float dry=float(posp1.y-p2xy.y);
          const float drz=float(posp1.z-posz[p2]);
          const float rr2=drx*drx+dry*dry+drz*drz;
          if(rr2<=CTE.kernelsize2 && rr2>=ALMOSTZERO && CODE_IsFluidNotInout(code[p2])){//-Only with fluid particles but not inout particles.
            //-Computes kernel.
            float fac;
            const float wab=cufsph::GetKernel_WabFac<tker>(rr2,fac);
            const float frx=fac*drx,fry=fac*dry,frz=fac*drz; //-Gradients.

            const float4 velrhopp2=velrhop[p2];
            //===== Get mass and volume of particle p2 =====
            float massp2=CTE.massf;
            float volp2=massp2/velrhopp2.w;

            //===== Density and its gradient =====
            rhopp1+=massp2*wab;
            gradrhopp1.x+=massp2*frx;
            gradrhopp1.y+=massp2*fry;
            gradrhopp1.z+=massp2*frz;

            //===== Kernel values multiplied by volume =====
            const float vwab=wab*volp2;
            const float vfrx=frx*volp2;
            const float vfry=fry*volp2;
            const float vfrz=frz*volp2;

            //===== Velocity and its gradient =====
            if(computevel){
              velp1.x+=vwab*velrhopp2.x;
              velp1.y+=vwab*velrhopp2.y;
              velp1.z+=vwab*velrhopp2.z;
              gradvelp1.a11+=vfrx*velrhopp2.x;    // du/dx
              gradvelp1.a12+=vfry*velrhopp2.x;    // du/dy
              gradvelp1.a13+=vfrz*velrhopp2.x;    // du/dz
              gradvelp1.a21+=vfrx*velrhopp2.y;    // dv/dx
              gradvelp1.a22+=vfry*velrhopp2.y;    // dv/dx
              gradvelp1.a23+=vfrz*velrhopp2.y;    // dv/dx
              gradvelp1.a31+=vfrx*velrhopp2.z;    // dw/dx
              gradvelp1.a32+=vfry*velrhopp2.z;    // dw/dx
              gradvelp1.a33+=vfrz*velrhopp2.z;    // dw/dx
            }

            //===== Matrix A for correction =====
            if(sim2d){
              a_corr2.a11+=vwab;  a_corr2.a12+=drx*vwab;  a_corr2.a13+=drz*vwab;
              a_corr2.a21+=vfrx;  a_corr2.a22+=drx*vfrx;  a_corr2.a23+=drz*vfrx;
              a_corr2.a31+=vfrz;  a_corr2.a32+=drx*vfrz;  a_corr2.a33+=drz*vfrz;
            }
            else{
              a_corr3.a11+=vwab;  a_corr3.a12+=drx*vwab;  a_corr3.a13+=dry*vwab;  a_corr3.a14+=drz*vwab;
              a_corr3.a21+=vfrx;  a_corr3.a22+=drx*vfrx;  a_corr3.a23+=dry*vfrx;  a_corr3.a24+=drz*vfrx;
              a_corr3.a31+=vfry;  a_corr3.a32+=drx*vfry;  a_corr3.a33+=dry*vfry;  a_corr3.a34+=drz*vfry;
              a_corr3.a41+=vfrz;  a_corr3.a42+=drx*vfrz;  a_corr3.a43+=dry*vfrz;  a_corr3.a44+=drz*vfrz;
            }
          }
        }
      }

      //-Store the results.
      //--------------------
      float4 velrhopfinal=velrhop[p1];
      const float3 dpos=make_float3(float(pos_p1.x-posp1.x),float(pos_p1.y-posp1.y),float(pos_p1.z-posp1.z)); //-Inlet/outlet particle position - ghost node position.
      if(sim2d){
        const double determ=cumath::Determinant3x3(a_corr2);
        if(fabs(determ)>=determlimit){//-Use 1e-3f (first_order) or 1e+3f (zeroth_order).
          const tmatrix3d invacorr2=cumath::InverseMatrix3x3(a_corr2,determ);
          //-GHOST NODE DENSITY IS MIRRORED BACK TO THE INFLOW OR OUTFLOW PARTICLES.
          if(computerhop){
            const float rhoghost=float(invacorr2.a11*rhopp1 + invacorr2.a12*gradrhopp1.x + invacorr2.a13*gradrhopp1.z);
            const float grx=    -float(invacorr2.a21*rhopp1 + invacorr2.a22*gradrhopp1.x + invacorr2.a23*gradrhopp1.z);
            const float grz=    -float(invacorr2.a31*rhopp1 + invacorr2.a32*gradrhopp1.x + invacorr2.a33*gradrhopp1.z);
            velrhopfinal.w=(rhoghost + grx*dpos.x + grz*dpos.z);
          }
          //-GHOST NODE VELOCITY ARE MIRRORED BACK TO THE OUTFLOW PARTICLES.
          if(computevel){
            const float velghost_x=float(invacorr2.a11*velp1.x + invacorr2.a12*gradvelp1.a11 + invacorr2.a13*gradvelp1.a13);
            const float velghost_z=float(invacorr2.a11*velp1.z + invacorr2.a12*gradvelp1.a31 + invacorr2.a13*gradvelp1.a33);
            const float a11=-float(invacorr2.a21*velp1.x + invacorr2.a22*gradvelp1.a11 + invacorr2.a23*gradvelp1.a13);
            const float a13=-float(invacorr2.a21*velp1.z + invacorr2.a22*gradvelp1.a31 + invacorr2.a23*gradvelp1.a33);
            const float a31=-float(invacorr2.a31*velp1.x + invacorr2.a32*gradvelp1.a11 + invacorr2.a33*gradvelp1.a13);
            const float a33=-float(invacorr2.a31*velp1.z + invacorr2.a32*gradvelp1.a31 + invacorr2.a33*gradvelp1.a33);
            velrhopfinal.x=(velghost_x + a11*dpos.x + a31*dpos.z);
            velrhopfinal.z=(velghost_z + a13*dpos.x + a33*dpos.z);
            velrhopfinal.y=0;
          }
        }
        else if(a_corr2.a11>0){//-Determinant is small but a11 is nonzero, 0th order ANGELO.
          if(computerhop)velrhopfinal.w=float(rhopp1/a_corr2.a11);
          if(computevel){
            velrhopfinal.x=float(velp1.x/a_corr2.a11);
            velrhopfinal.z=float(velp1.z/a_corr2.a11);
            velrhopfinal.y=0;
          }
        }
      }
      else{
        const double determ=cumath::Determinant4x4(a_corr3);
        if(fabs(determ)>=determlimit){
          const tmatrix4d invacorr3=cumath::InverseMatrix4x4(a_corr3,determ);
          //-GHOST NODE DENSITY IS MIRRORED BACK TO THE INFLOW OR OUTFLOW PARTICLES.
          if(computerhop){
            const float rhoghost=float(invacorr3.a11*rhopp1 + invacorr3.a12*gradrhopp1.x + invacorr3.a13*gradrhopp1.y + invacorr3.a14*gradrhopp1.z);
            const float grx=    -float(invacorr3.a21*rhopp1 + invacorr3.a22*gradrhopp1.x + invacorr3.a23*gradrhopp1.y + invacorr3.a24*gradrhopp1.z);
            const float gry=    -float(invacorr3.a31*rhopp1 + invacorr3.a32*gradrhopp1.x + invacorr3.a33*gradrhopp1.y + invacorr3.a34*gradrhopp1.z);
            const float grz=    -float(invacorr3.a41*rhopp1 + invacorr3.a42*gradrhopp1.x + invacorr3.a43*gradrhopp1.y + invacorr3.a44*gradrhopp1.z);
            velrhopfinal.w=(rhoghost + grx*dpos.x + gry*dpos.y + grz*dpos.z);
          }
          //-GHOST NODE VELOCITY ARE MIRRORED BACK TO THE OUTFLOW PARTICLES.
          if(computevel){
            const float velghost_x=float(invacorr3.a11*velp1.x + invacorr3.a12*gradvelp1.a11 + invacorr3.a13*gradvelp1.a12 + invacorr3.a14*gradvelp1.a13);
            const float velghost_y=float(invacorr3.a11*velp1.y + invacorr3.a12*gradvelp1.a21 + invacorr3.a13*gradvelp1.a22 + invacorr3.a14*gradvelp1.a23);
            const float velghost_z=float(invacorr3.a11*velp1.z + invacorr3.a12*gradvelp1.a31 + invacorr3.a13*gradvelp1.a32 + invacorr3.a14*gradvelp1.a33);
            const float a11=      -float(invacorr3.a21*velp1.x + invacorr3.a22*gradvelp1.a11 + invacorr3.a23*gradvelp1.a12 + invacorr3.a24*gradvelp1.a13);
            const float a12=      -float(invacorr3.a21*velp1.y + invacorr3.a22*gradvelp1.a21 + invacorr3.a23*gradvelp1.a22 + invacorr3.a24*gradvelp1.a23);
            const float a13=      -float(invacorr3.a21*velp1.z + invacorr3.a22*gradvelp1.a31 + invacorr3.a23*gradvelp1.a32 + invacorr3.a24*gradvelp1.a33);
            const float a21=      -float(invacorr3.a31*velp1.x + invacorr3.a32*gradvelp1.a11 + invacorr3.a33*gradvelp1.a12 + invacorr3.a34*gradvelp1.a13);
            const float a22=      -float(invacorr3.a31*velp1.y + invacorr3.a32*gradvelp1.a21 + invacorr3.a33*gradvelp1.a22 + invacorr3.a34*gradvelp1.a23);
            const float a23=      -float(invacorr3.a31*velp1.z + invacorr3.a32*gradvelp1.a31 + invacorr3.a33*gradvelp1.a32 + invacorr3.a34*gradvelp1.a33);
            const float a31=      -float(invacorr3.a41*velp1.x + invacorr3.a42*gradvelp1.a11 + invacorr3.a43*gradvelp1.a12 + invacorr3.a44*gradvelp1.a13);
            const float a32=      -float(invacorr3.a41*velp1.y + invacorr3.a42*gradvelp1.a21 + invacorr3.a43*gradvelp1.a22 + invacorr3.a44*gradvelp1.a23);
            const float a33=      -float(invacorr3.a41*velp1.z + invacorr3.a42*gradvelp1.a31 + invacorr3.a43*gradvelp1.a32 + invacorr3.a44*gradvelp1.a33);
            velrhopfinal.x=(velghost_x + a11*dpos.x + a21*dpos.y + a31*dpos.z);
            velrhopfinal.y=(velghost_y + a12*dpos.x + a22*dpos.y + a32*dpos.z);
            velrhopfinal.z=(velghost_z + a13*dpos.x + a23*dpos.y + a33*dpos.z);
          }
        }
        else if(a_corr3.a11>0){ // Determinant is small but a11 is nonzero, 0th order ANGELO
          if(computerhop)velrhopfinal.w=float(rhopp1/a_corr3.a11);
          if(computevel){
            velrhopfinal.x=float(velp1.x/a_corr3.a11);
            velrhopfinal.y=float(velp1.y/a_corr3.a11);
            velrhopfinal.z=float(velp1.z/a_corr3.a11);
          }
        }
      }
      velrhop[p1]=velrhopfinal;
    }
  }
}


//------------------------------------------------------------------------------
/// Perform interaction between ghost inlet/outlet nodes and fluid particles. GhostNodes-Fluid
/// Realiza interaccion entre ghost inlet/outlet nodes y particulas de fluido. GhostNodes-Fluid
//------------------------------------------------------------------------------
template<bool sim2d,TpKernel tker> __global__ void KerInteractionInOutExtrap_FastSingle(
  unsigned inoutcount,const int* inoutpart,const byte* cfgzone
  ,byte computerhopmask,byte computevelmask
  ,const float4* planes,const float* width,const float3* dirdata,float determlimit
  ,int scelldiv,int4 nc,int3 cellzero,const int2* beginendcellfluid
  ,const double2* posxy,const double* posz,const typecode* code
  ,const unsigned* idp,float4* velrhop)
{
  const unsigned cp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(cp<inoutcount){
    const unsigned p1=inoutpart[cp];
    const byte izone=byte(CODE_GetIzoneFluidInout(code[p1]));
    const byte cfg=cfgzone[izone];
    const bool computerhop=((cfg&computerhopmask)!=0);
    const bool computevel= ((cfg&computevelmask )!=0);
    if(computerhop || computevel){
      //-Calculates ghost node position.
      double3 pos_p1=make_double3(posxy[p1].x,posxy[p1].y,posz[p1]);
      if(CODE_IsPeriodic(code[p1]))pos_p1=KerInteraction_PosNoPeriodic(pos_p1);
      const double displane=cumath::DistPlane(planes[izone],pos_p1)*2;
      const float3 rdirdata=dirdata[izone];
      const double3 posp1=make_double3(pos_p1.x+displane*rdirdata.x, pos_p1.y+displane*rdirdata.y, pos_p1.z+displane*rdirdata.z); //-Ghost node position.

      //-Initializes variables for calculation.
      float rhopp1=0;
      float3 gradrhopp1=make_float3(0,0,0);
      float3 velp1=make_float3(0,0,0);
      tmatrix3f gradvelp1; cumath::Tmatrix3fReset(gradvelp1); //-Only for velocity.
      tmatrix3f a_corr2; if(sim2d) cumath::Tmatrix3fReset(a_corr2); //-Only for 2D.
      tmatrix4f a_corr3; if(!sim2d)cumath::Tmatrix4fReset(a_corr3); //-Only for 3D.
    
      //-Obtains neighborhood search limits.
      int ini1,fin1,ini2,fin2,ini3,fin3;
      cunsearch::InitCte(posp1.x,posp1.y,posp1.z,scelldiv,nc,cellzero,ini1,fin1,ini2,fin2,ini3,fin3);

      //-Interaction with fluids.
      for(int c3=ini3;c3<fin3;c3+=nc.w)for(int c2=ini2;c2<fin2;c2+=nc.x){
        unsigned pini,pfin=0;  cunsearch::ParticleRange(c2,c3,ini1,fin1,beginendcellfluid,pini,pfin);
        if(pfin)for(unsigned p2=pini;p2<pfin;p2++){
          const double2 p2xy=posxy[p2];
          const float drx=float(posp1.x-p2xy.x);
          const float dry=float(posp1.y-p2xy.y);
          const float drz=float(posp1.z-posz[p2]);
          const float rr2=drx*drx+dry*dry+drz*drz;
          if(rr2<=CTE.kernelsize2 && rr2>=ALMOSTZERO && CODE_IsFluidNotInout(code[p2])){//-Only with fluid particles but not inout particles.
            //-Computes kernel.
            float fac;
            const float wab=cufsph::GetKernel_WabFac<tker>(rr2,fac);
            const float frx=fac*drx,fry=fac*dry,frz=fac*drz; //-Gradients.

            const float4 velrhopp2=velrhop[p2];
            //===== Get mass and volume of particle p2 =====
            float massp2=CTE.massf;
            float volp2=massp2/velrhopp2.w;

            //===== Density and its gradient =====
            rhopp1+=massp2*wab;
            gradrhopp1.x+=massp2*frx;
            gradrhopp1.y+=massp2*fry;
            gradrhopp1.z+=massp2*frz;

            //===== Kernel values multiplied by volume =====
            const float vwab=wab*volp2;
            const float vfrx=frx*volp2;
            const float vfry=fry*volp2;
            const float vfrz=frz*volp2;

            //===== Velocity and its gradient =====
            if(computevel){
              velp1.x+=vwab*velrhopp2.x;
              velp1.y+=vwab*velrhopp2.y;
              velp1.z+=vwab*velrhopp2.z;
              gradvelp1.a11+=vfrx*velrhopp2.x;    // du/dx
              gradvelp1.a12+=vfry*velrhopp2.x;    // du/dy
              gradvelp1.a13+=vfrz*velrhopp2.x;    // du/dz
              gradvelp1.a21+=vfrx*velrhopp2.y;    // dv/dx
              gradvelp1.a22+=vfry*velrhopp2.y;    // dv/dx
              gradvelp1.a23+=vfrz*velrhopp2.y;    // dv/dx
              gradvelp1.a31+=vfrx*velrhopp2.z;    // dw/dx
              gradvelp1.a32+=vfry*velrhopp2.z;    // dw/dx
              gradvelp1.a33+=vfrz*velrhopp2.z;    // dw/dx
            }

            //===== Matrix A for correction =====
            if(sim2d){
              a_corr2.a11+=vwab;  a_corr2.a12+=drx*vwab;  a_corr2.a13+=drz*vwab;
              a_corr2.a21+=vfrx;  a_corr2.a22+=drx*vfrx;  a_corr2.a23+=drz*vfrx;
              a_corr2.a31+=vfrz;  a_corr2.a32+=drx*vfrz;  a_corr2.a33+=drz*vfrz;
            }
            else{
              a_corr3.a11+=vwab;  a_corr3.a12+=drx*vwab;  a_corr3.a13+=dry*vwab;  a_corr3.a14+=drz*vwab;
              a_corr3.a21+=vfrx;  a_corr3.a22+=drx*vfrx;  a_corr3.a23+=dry*vfrx;  a_corr3.a24+=drz*vfrx;
              a_corr3.a31+=vfry;  a_corr3.a32+=drx*vfry;  a_corr3.a33+=dry*vfry;  a_corr3.a34+=drz*vfry;
              a_corr3.a41+=vfrz;  a_corr3.a42+=drx*vfrz;  a_corr3.a43+=dry*vfrz;  a_corr3.a44+=drz*vfrz;
            }
          }
        }
      }

      //-Store the results.
      //--------------------
      float4 velrhopfinal=velrhop[p1];
      const float3 dpos=make_float3(float(pos_p1.x-posp1.x),float(pos_p1.y-posp1.y),float(pos_p1.z-posp1.z)); //-Inlet/outlet particle position - ghost node position.
      if(sim2d){
        const double determ=cumath::Determinant3x3dbl(a_corr2);
        if(fabs(determ)>=determlimit){//-Use 1e-3f (first_order) or 1e+3f (zeroth_order).
          const tmatrix3f invacorr2=cumath::InverseMatrix3x3dbl(a_corr2,determ);
          //-GHOST NODE DENSITY IS MIRRORED BACK TO THE INFLOW OR OUTFLOW PARTICLES.
          if(computerhop){
            const float rhoghost=float(invacorr2.a11*rhopp1 + invacorr2.a12*gradrhopp1.x + invacorr2.a13*gradrhopp1.z);
            const float grx=    -float(invacorr2.a21*rhopp1 + invacorr2.a22*gradrhopp1.x + invacorr2.a23*gradrhopp1.z);
            const float grz=    -float(invacorr2.a31*rhopp1 + invacorr2.a32*gradrhopp1.x + invacorr2.a33*gradrhopp1.z);
            velrhopfinal.w=(rhoghost + grx*dpos.x + grz*dpos.z);
          }
          //-GHOST NODE VELOCITY ARE MIRRORED BACK TO THE OUTFLOW PARTICLES.
          if(computevel){
            const float velghost_x=float(invacorr2.a11*velp1.x + invacorr2.a12*gradvelp1.a11 + invacorr2.a13*gradvelp1.a13);
            const float velghost_z=float(invacorr2.a11*velp1.z + invacorr2.a12*gradvelp1.a31 + invacorr2.a13*gradvelp1.a33);
            const float a11=-float(invacorr2.a21*velp1.x + invacorr2.a22*gradvelp1.a11 + invacorr2.a23*gradvelp1.a13);
            const float a13=-float(invacorr2.a21*velp1.z + invacorr2.a22*gradvelp1.a31 + invacorr2.a23*gradvelp1.a33);
            const float a31=-float(invacorr2.a31*velp1.x + invacorr2.a32*gradvelp1.a11 + invacorr2.a33*gradvelp1.a13);
            const float a33=-float(invacorr2.a31*velp1.z + invacorr2.a32*gradvelp1.a31 + invacorr2.a33*gradvelp1.a33);
            velrhopfinal.x=(velghost_x + a11*dpos.x + a31*dpos.z);
            velrhopfinal.z=(velghost_z + a13*dpos.x + a33*dpos.z);
            velrhopfinal.y=0;
          }
        }
        else if(a_corr2.a11>0){//-Determinant is small but a11 is nonzero, 0th order ANGELO.
          if(computerhop)velrhopfinal.w=float(rhopp1/a_corr2.a11);
          if(computevel){
            velrhopfinal.x=float(velp1.x/a_corr2.a11);
            velrhopfinal.z=float(velp1.z/a_corr2.a11);
            velrhopfinal.y=0;
          }
        }
      }
      else{
        const double determ=cumath::Determinant4x4dbl(a_corr3);
        if(fabs(determ)>=determlimit){
          const tmatrix4f invacorr3=cumath::InverseMatrix4x4dbl(a_corr3,determ);
          //-GHOST NODE DENSITY IS MIRRORED BACK TO THE INFLOW OR OUTFLOW PARTICLES.
          if(computerhop){
            const float rhoghost=float(invacorr3.a11*rhopp1 + invacorr3.a12*gradrhopp1.x + invacorr3.a13*gradrhopp1.y + invacorr3.a14*gradrhopp1.z);
            const float grx=    -float(invacorr3.a21*rhopp1 + invacorr3.a22*gradrhopp1.x + invacorr3.a23*gradrhopp1.y + invacorr3.a24*gradrhopp1.z);
            const float gry=    -float(invacorr3.a31*rhopp1 + invacorr3.a32*gradrhopp1.x + invacorr3.a33*gradrhopp1.y + invacorr3.a34*gradrhopp1.z);
            const float grz=    -float(invacorr3.a41*rhopp1 + invacorr3.a42*gradrhopp1.x + invacorr3.a43*gradrhopp1.y + invacorr3.a44*gradrhopp1.z);
            velrhopfinal.w=(rhoghost + grx*dpos.x + gry*dpos.y + grz*dpos.z);
          }
          //-GHOST NODE VELOCITY ARE MIRRORED BACK TO THE OUTFLOW PARTICLES.
          if(computevel){
            const float velghost_x=float(invacorr3.a11*velp1.x + invacorr3.a12*gradvelp1.a11 + invacorr3.a13*gradvelp1.a12 + invacorr3.a14*gradvelp1.a13);
            const float velghost_y=float(invacorr3.a11*velp1.y + invacorr3.a12*gradvelp1.a21 + invacorr3.a13*gradvelp1.a22 + invacorr3.a14*gradvelp1.a23);
            const float velghost_z=float(invacorr3.a11*velp1.z + invacorr3.a12*gradvelp1.a31 + invacorr3.a13*gradvelp1.a32 + invacorr3.a14*gradvelp1.a33);
            const float a11=      -float(invacorr3.a21*velp1.x + invacorr3.a22*gradvelp1.a11 + invacorr3.a23*gradvelp1.a12 + invacorr3.a24*gradvelp1.a13);
            const float a12=      -float(invacorr3.a21*velp1.y + invacorr3.a22*gradvelp1.a21 + invacorr3.a23*gradvelp1.a22 + invacorr3.a24*gradvelp1.a23);
            const float a13=      -float(invacorr3.a21*velp1.z + invacorr3.a22*gradvelp1.a31 + invacorr3.a23*gradvelp1.a32 + invacorr3.a24*gradvelp1.a33);
            const float a21=      -float(invacorr3.a31*velp1.x + invacorr3.a32*gradvelp1.a11 + invacorr3.a33*gradvelp1.a12 + invacorr3.a34*gradvelp1.a13);
            const float a22=      -float(invacorr3.a31*velp1.y + invacorr3.a32*gradvelp1.a21 + invacorr3.a33*gradvelp1.a22 + invacorr3.a34*gradvelp1.a23);
            const float a23=      -float(invacorr3.a31*velp1.z + invacorr3.a32*gradvelp1.a31 + invacorr3.a33*gradvelp1.a32 + invacorr3.a34*gradvelp1.a33);
            const float a31=      -float(invacorr3.a41*velp1.x + invacorr3.a42*gradvelp1.a11 + invacorr3.a43*gradvelp1.a12 + invacorr3.a44*gradvelp1.a13);
            const float a32=      -float(invacorr3.a41*velp1.y + invacorr3.a42*gradvelp1.a21 + invacorr3.a43*gradvelp1.a22 + invacorr3.a44*gradvelp1.a23);
            const float a33=      -float(invacorr3.a41*velp1.z + invacorr3.a42*gradvelp1.a31 + invacorr3.a43*gradvelp1.a32 + invacorr3.a44*gradvelp1.a33);
            velrhopfinal.x=(velghost_x + a11*dpos.x + a21*dpos.y + a31*dpos.z);
            velrhopfinal.y=(velghost_y + a12*dpos.x + a22*dpos.y + a32*dpos.z);
            velrhopfinal.z=(velghost_z + a13*dpos.x + a23*dpos.y + a33*dpos.z);
          }
        }
        else if(a_corr3.a11>0){ // Determinant is small but a11 is nonzero, 0th order ANGELO
          if(computerhop)velrhopfinal.w=float(rhopp1/a_corr3.a11);
          if(computevel){
            velrhopfinal.x=float(velp1.x/a_corr3.a11);
            velrhopfinal.y=float(velp1.y/a_corr3.a11);
            velrhopfinal.z=float(velp1.z/a_corr3.a11);
          }
        }
      }
      velrhop[p1]=velrhopfinal;
    }
  }
}

//==============================================================================
/// Perform interaction between ghost inlet/outlet nodes and fluid particles. GhostNodes-Fluid
/// Realiza interaccion entre ghost inlet/outlet nodes y particulas de fluido. GhostNodes-Fluid
//==============================================================================
template<TpKernel tker> void Interaction_InOutExtrapT(byte doublemode
  ,bool simulate2d,unsigned inoutcount,const int* inoutpart,const byte* cfgzone
  ,byte computerhopmask,byte computevelmask,const float4* planes,const float* width
  ,const float3* dirdata,float determlimit,const StDivDataGpu& dvd
  ,const double2* posxy,const double* posz,const typecode* code
  ,const unsigned* idp,float4* velrhop)
{
  const int2* beginendcellfluid=dvd.beginendcell+dvd.cellfluid;
  //-Interaction GhostBoundaryNodes-Fluid.
  if(inoutcount){
    const unsigned bsize=128;
    dim3 sgrid=GetSimpleGridSize(inoutcount,bsize);
    if(simulate2d){ const bool sim2d=true;
      switch(doublemode){
        case 1:  KerInteractionInOutExtrap_FastSingle<sim2d,tker> <<<sgrid,bsize>>> (inoutcount,inoutpart,cfgzone,computerhopmask,computevelmask,planes,width,dirdata,determlimit,dvd.scelldiv,dvd.nc,dvd.cellzero,beginendcellfluid,posxy,posz,code,idp,velrhop);  break;
        case 2:  KerInteractionInOutExtrap_Single    <sim2d,tker> <<<sgrid,bsize>>> (inoutcount,inoutpart,cfgzone,computerhopmask,computevelmask,planes,width,dirdata,determlimit,dvd.scelldiv,dvd.nc,dvd.cellzero,beginendcellfluid,posxy,posz,code,idp,velrhop);  break;
        case 3:  KerInteractionInOutExtrap_Double    <sim2d,tker> <<<sgrid,bsize>>> (inoutcount,inoutpart,cfgzone,computerhopmask,computevelmask,planes,width,dirdata,determlimit,dvd.scelldiv,dvd.nc,dvd.cellzero,beginendcellfluid,posxy,posz,code,idp,velrhop);  break;
      }
    }
    else{           const bool sim2d=false;
      switch(doublemode){
        case 1:  KerInteractionInOutExtrap_FastSingle<sim2d,tker> <<<sgrid,bsize>>> (inoutcount,inoutpart,cfgzone,computerhopmask,computevelmask,planes,width,dirdata,determlimit,dvd.scelldiv,dvd.nc,dvd.cellzero,beginendcellfluid,posxy,posz,code,idp,velrhop);  break;
        case 2:  KerInteractionInOutExtrap_Single    <sim2d,tker> <<<sgrid,bsize>>> (inoutcount,inoutpart,cfgzone,computerhopmask,computevelmask,planes,width,dirdata,determlimit,dvd.scelldiv,dvd.nc,dvd.cellzero,beginendcellfluid,posxy,posz,code,idp,velrhop);  break;
        case 3:  KerInteractionInOutExtrap_Double    <sim2d,tker> <<<sgrid,bsize>>> (inoutcount,inoutpart,cfgzone,computerhopmask,computevelmask,planes,width,dirdata,determlimit,dvd.scelldiv,dvd.nc,dvd.cellzero,beginendcellfluid,posxy,posz,code,idp,velrhop);  break;
      }
    }
  }
}

//==============================================================================
/// Perform interaction between ghost inlet/outlet nodes and fluid particles. GhostNodes-Fluid
/// Realiza interaccion entre ghost inlet/outlet nodes y particulas de fluido. GhostNodes-Fluid
//==============================================================================
void Interaction_InOutExtrap(byte doublemode,bool simulate2d,TpKernel tkernel
  ,unsigned inoutcount,const int* inoutpart,const byte* cfgzone
  ,byte computerhopmask,byte computevelmask,const float4* planes
  ,const float* width,const float3* dirdata,float determlimit
  ,const StDivDataGpu& dvd,const double2* posxy,const double* posz
  ,const typecode* code,const unsigned* idp,float4* velrhop)
{
  switch(tkernel){
    case KERNEL_Wendland:
      Interaction_InOutExtrapT<KERNEL_Wendland>(doublemode,simulate2d
        ,inoutcount,inoutpart,cfgzone,computerhopmask,computevelmask
        ,planes,width,dirdata,determlimit,dvd,posxy,posz,code,idp,velrhop);
    break;
#ifndef DISABLE_KERNELS_EXTRA
    case KERNEL_Cubic:
      Interaction_InOutExtrapT<KERNEL_Cubic>(doublemode,simulate2d
        ,inoutcount,inoutpart,cfgzone,computerhopmask,computevelmask
        ,planes,width,dirdata,determlimit,dvd,posxy,posz,code,idp,velrhop);
    break;
#endif
    default: throw "Kernel unknown at Interaction_InOutExtrap().";
  }
}

//------------------------------------------------------------------------------
/// Updates velocity of inout fluid according to circle jet velocity profile.
//------------------------------------------------------------------------------
__global__ void KerInOutUpdateJetVel(unsigned izone,float4 plane,float3 ptplane
  ,float3 opencenter,float radiusfr,float dp
  ,float3 direction,float inputvel,unsigned np,const int* plist
  ,const typecode* code,const double2* posxy,const double* posz,float4* velrhop)
{
  const unsigned cp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(cp<np){
    const unsigned p=plist[cp];
    if(izone==CODE_GetIzoneFluidInout(code[p])){
      const double2 psxy=posxy[p];
      const float psx=float(psxy.x);
      const float psy=float(psxy.y);
      const float psz=float(posz[p]);
      const float pladis=cugeo::PlaneDistSign(plane,psx,psy,psz);
      float4 rvel=velrhop[p];
      if(pladis<=-dp){
        rvel.x=direction.x*inputvel;
        rvel.y=direction.y*inputvel;
        rvel.z=direction.z*inputvel;
      }
      else{
        const float pscenx=ptplane.x+(direction.x*pladis);
        const float psceny=ptplane.y+(direction.y*pladis);
        const float pscenz=ptplane.z+(direction.z*pladis);
        const float vcenx=psx-pscenx;
        const float vceny=psy-psceny;
        const float vcenz=psz-pscenz;
        const float ps2x=opencenter.x+vcenx*radiusfr;
        const float ps2y=opencenter.y+vceny*radiusfr;
        const float ps2z=opencenter.z+vcenz*radiusfr;
        const float3 v=cugeo::VecModule(ps2x-psx,ps2y-psy,ps2z-psz,inputvel);
        rvel.x=v.x;
        rvel.y=v.y;
        rvel.z=v.z;
      }
      velrhop[p]=rvel;
    }
  }
}

//==============================================================================
/// Updates velocity of inout fluid according to circle jet velocity profile.
//==============================================================================
void InOutUpdateJetVel(unsigned izone,const tplane3d& plane
  ,const tdouble3& ptplane,const tdouble3& opencenter,double radiusfr,double dp
  ,const tdouble3& direction,float inputvel,unsigned np,const int* plist
  ,const typecode* code,const double2* posxy,const double* posz,float4* velrhop)
{
  if(np){
    const float4 planef=Float4(ToTFloat4(TDouble4(plane.a,plane.b,plane.c,plane.d)));
    const float3 ptplanef=Float3(ToTFloat3(ptplane));
    const float3 opencenterf=Float3(ToTFloat3(opencenter));
    const float3 directionf=Float3(ToTFloat3(direction));
    dim3 sgrid=GetSimpleGridSize(np,SPHBSIZE);
    KerInOutUpdateJetVel <<<sgrid,SPHBSIZE>>> (izone,planef,ptplanef,opencenterf
      ,float(radiusfr),float(dp),directionf,inputvel,np,plist,code,posxy,posz,velrhop);
  }
}


//##############################################################################
//# Kernels to interpolate velocity (JSphInOutGridDataTime).
//# Kernels para interpolar valores de velocidad (JSphInOutGridDataTime).
//##############################################################################
//------------------------------------------------------------------------------
/// Interpolate data between time0 and time1.
//------------------------------------------------------------------------------
__global__ void KerInOutInterpolateTime(unsigned npt,double fxtime
  ,const float* vel0,const float* vel1,float* vel)
{
  const unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<npt){
    const float v0=vel0[p];
    vel[p]=float(fxtime*(vel1[p]-v0)+v0);
  }
}

//==============================================================================
/// Interpolate data between time0 and time1.
//==============================================================================
void InOutInterpolateTime(unsigned npt,double time,double t0,double t1
  ,const float* velx0,const float* velx1,float* velx
  ,const float* velz0,const float* velz1,float* velz)
{
  if(npt){
    const double fxtime=((time-t0)/(t1-t0));
    dim3 sgrid=GetSimpleGridSize(npt,SPHBSIZE);
    KerInOutInterpolateTime <<<sgrid,SPHBSIZE>>> (npt,fxtime,velx0,velx1,velx);
    if(velz0)KerInOutInterpolateTime <<<sgrid,SPHBSIZE>>> (npt,fxtime,velz0,velz1,velz);
  }
}

//------------------------------------------------------------------------------
/// Interpolate velocity in time and Z-position of selected partiles in a list.
//------------------------------------------------------------------------------
__global__ void KerInOutInterpolateZVel(unsigned izone,double posminz,double dpz
  ,int nz1,const float* velx,const float* velz,unsigned np,const int* plist
  ,const double* posz,const typecode* code,float4* velrhop,float velcorr)
{
  const unsigned cp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(cp<np){
    const unsigned p=plist[cp];
    if(izone==CODE_GetIzoneFluidInout(code[p])){
      const double pz=posz[p]-posminz;
      int cz=int(pz/dpz);
      cz=max(cz,0);
      cz=min(cz,nz1);
      const double fz=(pz/dpz-cz);  //const double fz=(pz-Dpz*cz)/Dpz;
      //-Interpolation in Z.
      const unsigned cp=cz;
      const float v00=velx[cp];
      const float v01=(cz<nz1? velx[cp+1]: v00);
      const float v=float(fz*(v01-v00)+v00);
      velrhop[p]=make_float4(v-velcorr,0,0,velrhop[p].w);
      if(velz!=NULL){
        const float v00=velz[cp];
        const float v01=(cz<nz1? velz[cp+1]:    v00);
        const float v=float(fz*(v01-v00)+v00);
        velrhop[p].z=v;
      }
    }
  }
}

//==============================================================================
/// Interpolate velocity in time and Z-position of selected partiles in a list.
//==============================================================================
void InOutInterpolateZVel(unsigned izone,double posminz,double dpz,int nz1
  ,const float* velx,const float* velz,unsigned np,const int* plist
  ,const double* posz,const typecode* code,float4* velrhop,float velcorr)
{
  if(np){
    dim3 sgrid=GetSimpleGridSize(np,SPHBSIZE);
    KerInOutInterpolateZVel <<<sgrid,SPHBSIZE>>> (izone,posminz,dpz,nz1,velx,velz,np,plist,posz,code,velrhop,velcorr);
  }
}

//------------------------------------------------------------------------------
/// Removes interpolated Z velocity of inlet/outlet particles.
//------------------------------------------------------------------------------
__global__ void KerInOutInterpolateResetZVel(unsigned izone,unsigned np
  ,const int* plist,const typecode* code,float4* velrhop)
{
  const unsigned cp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(cp<np){
    const unsigned p=plist[cp];
    if(izone==CODE_GetIzoneFluidInout(code[p]))velrhop[p].z=0;
  }
}

//==============================================================================
/// Removes interpolated Z velocity of inlet/outlet particles.
//==============================================================================
void InOutInterpolateResetZVel(unsigned izone,unsigned np,const int* plist
  ,const typecode* code,float4* velrhop)
{
  if(np){
    dim3 sgrid=GetSimpleGridSize(np,SPHBSIZE);
    KerInOutInterpolateResetZVel <<<sgrid,SPHBSIZE>>> (izone,np,plist,code,velrhop);
  }
}


//<vs_meeshdat_ini>
//##############################################################################
//# Kernels to interpolate data (JSphInOutZsurf and JMeshTDatasDsVel).
//# Kernels para interpolar valores (JSphInOutZsurf and JMeshTDatasDsVel).
//##############################################################################
//------------------------------------------------------------------------------
/// Interpolate data between time0 and time1.
//------------------------------------------------------------------------------
__global__ void KerInOutInterpolateDataTime(unsigned np,float tf
  ,const float* data0,const float* data1,float* res)
{
  const unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of value.
  if(p<np){
    const float v0=data0[p];
    res[p]=tf*(data1[p]-v0)+v0;
  }
}

//==============================================================================
/// Interpolate data between time0 and time1.
//==============================================================================
void InOutInterpolateDataTime(unsigned np,float tf
  ,const float* data0,const float* data1,float* res)
{
  if(np){
    dim3 sgrid=GetSimpleGridSize(np,SPHBSIZE);
    KerInOutInterpolateDataTime <<<sgrid,SPHBSIZE>>> (np,tf,data0,data1,res);
  }
}

//------------------------------------------------------------------------------
/// Interpolate data between time0 and time1.
//------------------------------------------------------------------------------
__global__ void KerInOutInterpolateDataTime(unsigned np,float tf
  ,const float3* data0,const float3* data1,float3* res)
{
  const unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of value.
  if(p<np){
    const float3 v0=data0[p];
    const float3 v1=data1[p];
    res[p]=make_float3(tf*(v1.x-v0.x)+v0.x, tf*(v1.y-v0.y)+v0.y, tf*(v1.z-v0.z)+v0.z);
  }
}

//==============================================================================
/// Interpolate data between time0 and time1.
//==============================================================================
void InOutInterpolateDataTime(unsigned np,float tf
  ,const float3* data0,const float3* data1,float3* res)
{
  if(np){
    dim3 sgrid=GetSimpleGridSize(np,SPHBSIZE);
    KerInOutInterpolateDataTime <<<sgrid,SPHBSIZE>>> (np,tf,data0,data1,res);
  }
}

//------------------------------------------------------------------------------
/// Interpolate data between positions using a 0-D mesh data.
//------------------------------------------------------------------------------
__global__ void KerInOutIntpVelFr0_f1(byte izone,float velcorr,float3 vdir
  ,const float* data1
  ,unsigned np,const int* plist,const typecode* code,float4* velrhop)
{
  const unsigned cp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of value.
  if(cp<np){
    const unsigned p=plist[cp];
    if(izone==byte(CODE_GetIzoneFluidInout(code[p]))){
      const float vv=data1[0]-velcorr;
      velrhop[p]=make_float4(vv*vdir.x, vv*vdir.y, vv*vdir.z, velrhop[p].w);
    }
  }
}
//------------------------------------------------------------------------------
/// Interpolate data between positions using a 0-D mesh data.
//------------------------------------------------------------------------------
__global__ void KerInOutIntpVelFr0_f3(byte izone,float3 vc3,const float3* data3
  ,unsigned np,const int* plist,const typecode* code,float4* velrhop)
{
  const unsigned cp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of value.
  if(cp<np){
    const unsigned p=plist[cp];
    if(izone==byte(CODE_GetIzoneFluidInout(code[p]))){
      const float3 v3=data3[0];
      velrhop[p]=make_float4(v3.x-vc3.x, v3.y-vc3.y, v3.z-vc3.z, velrhop[p].w);
    }
  }
}

//==============================================================================
/// Interpolate data between positions using a 0-D mesh data.
//==============================================================================
void InOutIntpVelFr0(byte izone,float velcorr,tfloat3 vdir
  ,const float* data1,const float3* data3
  ,unsigned np,const int* plist,const typecode* code,float4* velrhop)
{
  if(np){
    const tfloat3 velcorr3=vdir*velcorr;
    dim3 sgrid=GetSimpleGridSize(np,SPHBSIZE);
    if(data1)KerInOutIntpVelFr0_f1 <<<sgrid,SPHBSIZE>>> (izone,velcorr,Float3(vdir),data1,np,plist,code,velrhop);
    else     KerInOutIntpVelFr0_f3 <<<sgrid,SPHBSIZE>>> (izone,Float3(velcorr3)    ,data3,np,plist,code,velrhop);
  }
}


//------------------------------------------------------------------------------
/// Computes cell and factor position according to given data.
//------------------------------------------------------------------------------
__device__ float KerInOutIntpPosCellDist(const double3& ps,const double4& pla
  ,unsigned cmax,uint2& cell)
{
  double d=(ps.x*pla.x + ps.y*pla.y + ps.z*pla.z + pla.w);
  d=max(d,0.);
  const unsigned c=unsigned(d);
  cell.x=min(c,cmax);
  cell.y=min(c+1,cmax);
  return(c<cmax? float(d-c): 0);
}

//------------------------------------------------------------------------------
/// Interpolate data between positions using a 1-D mesh data.
//------------------------------------------------------------------------------
__global__ void KerInOutIntpVelFr1_f1(byte izone,float velcorr,float3 vdir
  ,const float* data,double4 pla1,unsigned cmax1
  ,unsigned np,const int* plist,const typecode* code
  ,const double2* posxy,const double* posz
  ,float4* velrhop)
{
  const unsigned cp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of value.
  if(cp<np){
    const unsigned p=plist[cp];
    if(izone==byte(CODE_GetIzoneFluidInout(code[p]))){
      double2 pxy=posxy[p];
      const double3 ps=make_double3(pxy.x,pxy.y,posz[p]);
      uint2 cell;
      const float fdis=KerInOutIntpPosCellDist(ps,pla1,cmax1,cell);
      const float v0=data[cell.x];
      const float v1=data[cell.y];
      const float vv=((v1-v0)*fdis+v0) -velcorr;
      velrhop[p]=make_float4(vv*vdir.x, vv*vdir.y, vv*vdir.z, velrhop[p].w);
    }
  }
}

//------------------------------------------------------------------------------
/// Interpolate data between positions using a 1-D mesh data.
//------------------------------------------------------------------------------
__global__ void KerInOutIntpVelFr1_f3(byte izone,float3 vc3
  ,const float3* data,double4 pla1,unsigned cmax1
  ,unsigned np,const int* plist,const typecode* code
  ,const double2* posxy,const double* posz
  ,float4* velrhop)
{
  const unsigned cp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of value.
  if(cp<np){
    const unsigned p=plist[cp];
    if(izone==byte(CODE_GetIzoneFluidInout(code[p]))){
      double2 pxy=posxy[p];
      const double3 ps=make_double3(pxy.x,pxy.y,posz[p]);
      uint2 cell;
      const float fdis=KerInOutIntpPosCellDist(ps,pla1,cmax1,cell);
      const float3 v0=data[cell.x];
      const float3 v1=data[cell.y];
      const float vvx=((v1.x-v0.x)*fdis+v0.x) -vc3.x;
      const float vvy=((v1.y-v0.y)*fdis+v0.y) -vc3.y;
      const float vvz=((v1.z-v0.z)*fdis+v0.z) -vc3.z;
      velrhop[p]=make_float4(vvx, vvy, vvz, velrhop[p].w);
    }
  }
}

//==============================================================================
/// Interpolate data between positions using a 1-D mesh data.
//==============================================================================
void InOutIntpVelFr1(byte izone,float velcorr,tfloat3 vdir
  ,const float* data1,const float3* data3
  ,tplane3d pla1,unsigned cmax1
  ,unsigned np,const int* plist,const typecode* code
  ,const double2* posxy,const double* posz,float4* velrhop)
{
  if(np){
    const tfloat3 velcorr3=vdir*velcorr;
    dim3 sgrid=GetSimpleGridSize(np,SPHBSIZE);
    if(data1)KerInOutIntpVelFr1_f1 <<<sgrid,SPHBSIZE>>> (izone,velcorr,Float3(vdir),data1,Double4(pla1),cmax1,np,plist,code,posxy,posz,velrhop);
    else     KerInOutIntpVelFr1_f3 <<<sgrid,SPHBSIZE>>> (izone,Float3(velcorr3)    ,data3,Double4(pla1),cmax1,np,plist,code,posxy,posz,velrhop);
  }
}


//------------------------------------------------------------------------------
/// Interpolate data between positions using a 2-D mesh data.
//------------------------------------------------------------------------------
__global__ void KerInOutIntpVelFr2_f1(byte izone,float velcorr,float3 vdir
  ,const float* data,double4 pla1,unsigned cmax1,double4 pla2,unsigned cmax2
  ,unsigned frnum1
  ,unsigned np,const int* plist,const typecode* code
  ,const double2* posxy,const double* posz
  ,float4* velrhop)
{
  const unsigned cp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of value.
  if(cp<np){
    const unsigned p=plist[cp];
    if(izone==byte(CODE_GetIzoneFluidInout(code[p]))){
      double2 pxy=posxy[p];
      const double3 ps=make_double3(pxy.x,pxy.y,posz[p]);
      uint2 cell1,cell2;
      const float fdis1=KerInOutIntpPosCellDist(ps,pla1,cmax1,cell1);
      const float fdis2=KerInOutIntpPosCellDist(ps,pla2,cmax2,cell2);
      const float v000=data[cell1.x + frnum1*cell2.x];
      const float v010=data[cell1.y + frnum1*cell2.x];
      const float v001=data[cell1.x + frnum1*cell2.y];
      const float v011=data[cell1.y + frnum1*cell2.y];
      const float v0x0=(v010-v000)*fdis1+v000;
      const float v0x1=(v011-v001)*fdis1+v001;
      const float vv=((v0x1-v0x0)*fdis2+v0x0) -velcorr;
      velrhop[p]=make_float4(vv*vdir.x, vv*vdir.y, vv*vdir.z, velrhop[p].w);
    }
  }
}

//------------------------------------------------------------------------------
/// Interpolate data between positions using a 2-D mesh data.
//------------------------------------------------------------------------------
__global__ void KerInOutIntpVelFr2_f3(byte izone,float3 vc3
  ,const float3* data,double4 pla1,unsigned cmax1,double4 pla2,unsigned cmax2
  ,unsigned frnum1
  ,unsigned np,const int* plist,const typecode* code
  ,const double2* posxy,const double* posz
  ,float4* velrhop)
{
  const unsigned cp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of value.
  if(cp<np){
    const unsigned p=plist[cp];
    if(izone==byte(CODE_GetIzoneFluidInout(code[p]))){
      double2 pxy=posxy[p];
      const double3 ps=make_double3(pxy.x,pxy.y,posz[p]);
      uint2 cell1,cell2;
      const float fdis1=KerInOutIntpPosCellDist(ps,pla1,cmax1,cell1);
      const float fdis2=KerInOutIntpPosCellDist(ps,pla2,cmax2,cell2);
      const float3 v000=data[cell1.x + frnum1*cell2.x];
      const float3 v010=data[cell1.y + frnum1*cell2.x];
      const float3 v001=data[cell1.x + frnum1*cell2.y];
      const float3 v011=data[cell1.y + frnum1*cell2.y];
      const float v0x0_x=(v010.x-v000.x)*fdis1+v000.x;
      const float v0x0_y=(v010.y-v000.y)*fdis1+v000.y;
      const float v0x0_z=(v010.z-v000.z)*fdis1+v000.z;
      const float v0x1_x=(v011.x-v001.x)*fdis1+v001.x;
      const float v0x1_y=(v011.y-v001.y)*fdis1+v001.y;
      const float v0x1_z=(v011.z-v001.z)*fdis1+v001.z;
      const float vvx=((v0x1_x-v0x0_x)*fdis2+v0x0_x) -vc3.x;
      const float vvy=((v0x1_y-v0x0_y)*fdis2+v0x0_y) -vc3.y;
      const float vvz=((v0x1_z-v0x0_z)*fdis2+v0x0_z) -vc3.z;
      velrhop[p]=make_float4(vvx, vvy, vvz, velrhop[p].w);
    }
  }
}

//==============================================================================
/// Interpolate data between positions using a 2-D mesh data.
//==============================================================================
void InOutIntpVelFr2(byte izone,float velcorr,tfloat3 vdir
  ,const float* data1,const float3* data3,unsigned frnum1
  ,tplane3d pla1,unsigned cmax1,tplane3d pla2,unsigned cmax2
  ,unsigned np,const int* plist,const typecode* code
  ,const double2* posxy,const double* posz,float4* velrhop)
{
  if(np){
    const tfloat3 velcorr3=vdir*velcorr;
    dim3 sgrid=GetSimpleGridSize(np,SPHBSIZE);
    if(data1)KerInOutIntpVelFr2_f1 <<<sgrid,SPHBSIZE>>> (izone,velcorr,Float3(vdir),data1,Double4(pla1),cmax1,Double4(pla2),cmax2,frnum1,np,plist,code,posxy,posz,velrhop);
    else     KerInOutIntpVelFr2_f3 <<<sgrid,SPHBSIZE>>> (izone,Float3(velcorr3)    ,data3,Double4(pla1),cmax1,Double4(pla2),cmax2,frnum1,np,plist,code,posxy,posz,velrhop);
  }
}


//------------------------------------------------------------------------------
/// Interpolate data between positions using a 3-D mesh data.
//------------------------------------------------------------------------------
__global__ void KerInOutIntpVelFr3_f1(byte izone,float velcorr,float3 vdir
  ,const float* tdat,double4 pla1,unsigned cmax1,double4 pla2,unsigned cmax2
  ,double4 pla3,unsigned cmax3,unsigned npt1,unsigned npt12
  ,unsigned np,const int* plist,const typecode* code
  ,const double2* posxy,const double* posz
  ,float4* velrhop)
{
  const unsigned cp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of value.
  if(cp<np){
    const unsigned p=plist[cp];
    if(izone==byte(CODE_GetIzoneFluidInout(code[p]))){
      double2 pxy=posxy[p];
      const double3 ps=make_double3(pxy.x,pxy.y,posz[p]);
      uint2 cell1,cell2,cell3;
      const float fdis1=KerInOutIntpPosCellDist(ps,pla1,cmax1,cell1);
      const float fdis2=KerInOutIntpPosCellDist(ps,pla2,cmax2,cell2);
      const float fdis3=KerInOutIntpPosCellDist(ps,pla3,cmax3,cell3);
      float resx,resy;
      {
        const unsigned mod3=npt12*cell3.x;
        const float v000=tdat[cell1.x + npt1*cell2.x + mod3];
        const float v100=tdat[cell1.y + npt1*cell2.x + mod3];
        const float v010=tdat[cell1.x + npt1*cell2.y + mod3];
        const float v110=tdat[cell1.y + npt1*cell2.y + mod3];
        const float vx00=(v100-v000)*fdis1+v000;
        const float vx10=(v110-v010)*fdis1+v010;
        resx=(vx10-vx00)*fdis2+vx00;
      }
      {
        const unsigned mod3=npt12*cell3.y;
        const float v000=tdat[cell1.x + npt1*cell2.x + mod3];
        const float v100=tdat[cell1.y + npt1*cell2.x + mod3];
        const float v010=tdat[cell1.x + npt1*cell2.y + mod3];
        const float v110=tdat[cell1.y + npt1*cell2.y + mod3];
        const float vx00=(v100-v000)*fdis1+v000;
        const float vx10=(v110-v010)*fdis1+v010;
        resy=(vx10-vx00)*fdis2+vx00;
      }
      const float vv=((resy-resx)*fdis3+resx) -velcorr;
      velrhop[p]=make_float4(vv*vdir.x, vv*vdir.y, vv*vdir.z, velrhop[p].w);
    }
  }
}

//------------------------------------------------------------------------------
/// Interpolate data between positions using a 3-D mesh data.
//------------------------------------------------------------------------------
__global__ void KerInOutIntpVelFr3_f3(byte izone,float3 vc3
  ,const float3* tdat,double4 pla1,unsigned cmax1,double4 pla2,unsigned cmax2
  ,double4 pla3,unsigned cmax3,unsigned npt1,unsigned npt12
  ,unsigned np,const int* plist,const typecode* code
  ,const double2* posxy,const double* posz
  ,float4* velrhop)
{
  const unsigned cp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of value.
  if(cp<np){
    const unsigned p=plist[cp];
    if(izone==byte(CODE_GetIzoneFluidInout(code[p]))){
      double2 pxy=posxy[p];
      const double3 ps=make_double3(pxy.x,pxy.y,posz[p]);
      uint2 cell1,cell2,cell3;
      const float fdis1=KerInOutIntpPosCellDist(ps,pla1,cmax1,cell1);
      const float fdis2=KerInOutIntpPosCellDist(ps,pla2,cmax2,cell2);
      const float fdis3=KerInOutIntpPosCellDist(ps,pla3,cmax3,cell3);
      float res0_x,res0_y,res0_z;
      {
        const unsigned mod3=npt12*cell3.x;
        const float3 v000=tdat[cell1.x + npt1*cell2.x + mod3];
        const float3 v100=tdat[cell1.y + npt1*cell2.x + mod3];
        const float3 v010=tdat[cell1.x + npt1*cell2.y + mod3];
        const float3 v110=tdat[cell1.y + npt1*cell2.y + mod3];
        const float vx00_x=(v100.x-v000.x)*fdis1+v000.x;
        const float vx00_y=(v100.y-v000.y)*fdis1+v000.y;
        const float vx00_z=(v100.z-v000.z)*fdis1+v000.z;
        const float vx10_x=(v110.x-v010.x)*fdis1+v010.x;
        const float vx10_y=(v110.y-v010.y)*fdis1+v010.y;
        const float vx10_z=(v110.z-v010.z)*fdis1+v010.z;
        res0_x=(vx10_x-vx00_x)*fdis2+vx00_x;
        res0_y=(vx10_y-vx00_y)*fdis2+vx00_y;
        res0_z=(vx10_z-vx00_z)*fdis2+vx00_z;
      }
      float res1_x,res1_y,res1_z;
      {
        const unsigned mod3=npt12*cell3.y;
        const float3 v000=tdat[cell1.x + npt1*cell2.x + mod3];
        const float3 v100=tdat[cell1.y + npt1*cell2.x + mod3];
        const float3 v010=tdat[cell1.x + npt1*cell2.y + mod3];
        const float3 v110=tdat[cell1.y + npt1*cell2.y + mod3];
        const float vx00_x=(v100.x-v000.x)*fdis1+v000.x;
        const float vx00_y=(v100.y-v000.y)*fdis1+v000.y;
        const float vx00_z=(v100.z-v000.z)*fdis1+v000.z;
        const float vx10_x=(v110.x-v010.x)*fdis1+v010.x;
        const float vx10_y=(v110.y-v010.y)*fdis1+v010.y;
        const float vx10_z=(v110.z-v010.z)*fdis1+v010.z;
        res1_x=(vx10_x-vx00_x)*fdis2+vx00_x;
        res1_y=(vx10_y-vx00_y)*fdis2+vx00_y;
        res1_z=(vx10_z-vx00_z)*fdis2+vx00_z;
      }
      const float vvx=((res1_x-res0_x)*fdis3+res0_x) -vc3.x;
      const float vvy=((res1_y-res0_y)*fdis3+res0_y) -vc3.y;
      const float vvz=((res1_z-res0_z)*fdis3+res0_z) -vc3.z;
      velrhop[p]=make_float4(vvx, vvy, vvz, velrhop[p].w);
    }
  }
}

//==============================================================================
/// Interpolate data between positions using a 3-D mesh data.
//==============================================================================
void InOutIntpVelFr3(byte izone,float velcorr,tfloat3 vdir
  ,const float* data1,const float3* data3,unsigned npt1,unsigned npt12
  ,tplane3d pla1,unsigned cmax1,tplane3d pla2,unsigned cmax2,tplane3d pla3,unsigned cmax3
  ,unsigned np,const int* plist,const typecode* code
  ,const double2* posxy,const double* posz,float4* velrhop)
{
  if(np){
    const tfloat3 velcorr3=vdir*velcorr;
    dim3 sgrid=GetSimpleGridSize(np,SPHBSIZE);
    if(data1)KerInOutIntpVelFr3_f1 <<<sgrid,SPHBSIZE>>> (izone,velcorr,Float3(vdir),data1,Double4(pla1),cmax1,Double4(pla2),cmax2,Double4(pla3),cmax3,npt1,npt12,np,plist,code,posxy,posz,velrhop);
    else     KerInOutIntpVelFr3_f3 <<<sgrid,SPHBSIZE>>> (izone,Float3(velcorr3)    ,data3,Double4(pla1),cmax1,Double4(pla2),cmax2,Double4(pla3),cmax3,npt1,npt12,np,plist,code,posxy,posz,velrhop);
  }
}
//<vs_meeshdat_end>


}


//##############################################################################
//# Kernels for mDBC and mDBC2.
//# Kernels para mDBC y mDBC2.
//##############################################################################
// #include "JSphGpu_mdbc_iker.cu"

#include "JSphGpu_mdbc_iker.h"
#include <cfloat>

namespace cusph{

//##############################################################################
//# Kernels for mDBC and mDBC2.
//# Kernels para mDBC y mDBC2.
//##############################################################################
//------------------------------------------------------------------------------
/// Perform interaction between ghost node of selected boundary and fluid.
/// Only for SlipMode==SLIP_Vel0 (DBC vel=0)
//------------------------------------------------------------------------------
template<TpKernel tker,bool sim2d>
  __global__ void KerInteractionMdbcCorrection_Fast(unsigned n,unsigned nbound
  ,double3 mapposmin,float poscellsize,const float4* poscell
  ,int scelldiv,int4 nc,int3 cellzero,const int2* beginendcellfluid
  ,const double2* posxy,const double* posz,const typecode* code
  ,const unsigned* idp,const float3* boundnor,float4* velrho)
{
  const unsigned p1=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p1<n){
    const float determlimit=1e-3f;
    const float3 bnormalp1=boundnor[p1];
    if(bnormalp1.x!=0 || bnormalp1.y!=0 || bnormalp1.z!=0){
      float rhofinal=FLT_MAX;

      //-Calculates ghost node position.
      double3 gposp1=make_double3(posxy[p1].x+bnormalp1.x,posxy[p1].y+bnormalp1.y,posz[p1]+bnormalp1.z);
      gposp1=(CTE.periactive!=0? KerUpdatePeriodicPos(gposp1): gposp1); //-Corrected interface Position.
      const float4 gpscellp1=KerComputePosCell(gposp1,mapposmin,poscellsize);

      //-Initializes variables for calculation.
      float rhop1=0;
      float3 gradrhop1=make_float3(0,0,0);
      tmatrix3f a_corr2; if(sim2d) cumath::Tmatrix3fReset(a_corr2); //-Only for 2D.
      tmatrix4f a_corr3; if(!sim2d)cumath::Tmatrix4fReset(a_corr3); //-Only for 3D.
    
      //-Obtains neighborhood search limits.
      int ini1,fin1,ini2,fin2,ini3,fin3;
      cunsearch::InitCte(gposp1.x,gposp1.y,gposp1.z,scelldiv,nc,cellzero,ini1,fin1,ini2,fin2,ini3,fin3);

      //-Boundary-Fluid interaction.
      for(int c3=ini3;c3<fin3;c3+=nc.w)for(int c2=ini2;c2<fin2;c2+=nc.x){
        unsigned pini,pfin=0;  cunsearch::ParticleRange(c2,c3,ini1,fin1,beginendcellfluid,pini,pfin);
        if(pfin)for(unsigned p2=pini;p2<pfin;p2++){
          const float4 pscellp2=poscell[p2];
          float drx=gpscellp1.x-pscellp2.x + CTE.poscellsize*(PSCEL_GetfX(gpscellp1.w)-PSCEL_GetfX(pscellp2.w));
          float dry=gpscellp1.y-pscellp2.y + CTE.poscellsize*(PSCEL_GetfY(gpscellp1.w)-PSCEL_GetfY(pscellp2.w));
          float drz=gpscellp1.z-pscellp2.z + CTE.poscellsize*(PSCEL_GetfZ(gpscellp1.w)-PSCEL_GetfZ(pscellp2.w));
          const float rr2=drx*drx+dry*dry+drz*drz;
          if(rr2<=CTE.kernelsize2 && CODE_IsFluid(code[p2])){//-Only with fluid particles (including inout).
            //-Computes kernel.
            float fac;
            const float wab=cufsph::GetKernel_WabFac<tker>(rr2,fac);
            const float frx=fac*drx,fry=fac*dry,frz=fac*drz; //-Gradients.

            //===== Get mass and volume of particle p2 =====
            const float4 velrhop2=velrho[p2];
            float massp2=CTE.massf;
            const float volp2=massp2/velrhop2.w;

            //===== Density and its gradient =====
            rhop1+=massp2*wab;
            gradrhop1.x+=massp2*frx;
            gradrhop1.y+=massp2*fry;
            gradrhop1.z+=massp2*frz;

            //===== Kernel values multiplied by volume =====
            const float vwab=wab*volp2;
            const float vfrx=frx*volp2;
            const float vfry=fry*volp2;
            const float vfrz=frz*volp2;

            //===== Matrix A for correction =====
            if(sim2d){
              a_corr2.a11+=vwab;  a_corr2.a12+=drx*vwab;  a_corr2.a13+=drz*vwab;
              a_corr2.a21+=vfrx;  a_corr2.a22+=drx*vfrx;  a_corr2.a23+=drz*vfrx;
              a_corr2.a31+=vfrz;  a_corr2.a32+=drx*vfrz;  a_corr2.a33+=drz*vfrz;
            }
            else{
              a_corr3.a11+=vwab;  a_corr3.a12+=drx*vwab;  a_corr3.a13+=dry*vwab;  a_corr3.a14+=drz*vwab;
              a_corr3.a21+=vfrx;  a_corr3.a22+=drx*vfrx;  a_corr3.a23+=dry*vfrx;  a_corr3.a24+=drz*vfrx;
              a_corr3.a31+=vfry;  a_corr3.a32+=drx*vfry;  a_corr3.a33+=dry*vfry;  a_corr3.a34+=drz*vfry;
              a_corr3.a41+=vfrz;  a_corr3.a42+=drx*vfrz;  a_corr3.a43+=dry*vfrz;  a_corr3.a44+=drz*vfrz;
            }
          }
        }
      }

      //-Store the results.
      //--------------------
      {
        const float3 dpos=make_float3(-bnormalp1.x,-bnormalp1.y,-bnormalp1.z); //-Boundary particle position - ghost node position.
        if(sim2d){
          const double determ=cumath::Determinant3x3dbl(a_corr2);
          if(fabs(determ)>=determlimit){//-Use 1e-3f (first_order) or 1e+3f (zeroth_order).
            const tmatrix3f invacorr2=cumath::InverseMatrix3x3dbl(a_corr2,determ);
            //-GHOST NODE DENSITY IS MIRRORED BACK TO THE BOUNDARY PARTICLES.
            const float rhoghost=float(invacorr2.a11*rhop1 + invacorr2.a12*gradrhop1.x + invacorr2.a13*gradrhop1.z);
            const float grx=    -float(invacorr2.a21*rhop1 + invacorr2.a22*gradrhop1.x + invacorr2.a23*gradrhop1.z);
            const float grz=    -float(invacorr2.a31*rhop1 + invacorr2.a32*gradrhop1.x + invacorr2.a33*gradrhop1.z);
            rhofinal=(rhoghost + grx*dpos.x + grz*dpos.z);
          }
          else if(a_corr2.a11>0){//-Determinant is small but a11 is nonzero, 0th order ANGELO.
            rhofinal=float(rhop1/a_corr2.a11);
          }
        }
        else{
          const double determ=cumath::Determinant4x4dbl(a_corr3);
          if(fabs(determ)>=determlimit){
            const tmatrix4f invacorr3=cumath::InverseMatrix4x4dbl(a_corr3,determ);
            //-GHOST NODE DENSITY IS MIRRORED BACK TO THE BOUNDARY PARTICLES.
            const float rhoghost=float(invacorr3.a11*rhop1 + invacorr3.a12*gradrhop1.x + invacorr3.a13*gradrhop1.y + invacorr3.a14*gradrhop1.z);
            const float grx=    -float(invacorr3.a21*rhop1 + invacorr3.a22*gradrhop1.x + invacorr3.a23*gradrhop1.y + invacorr3.a24*gradrhop1.z);
            const float gry=    -float(invacorr3.a31*rhop1 + invacorr3.a32*gradrhop1.x + invacorr3.a33*gradrhop1.y + invacorr3.a34*gradrhop1.z);
            const float grz=    -float(invacorr3.a41*rhop1 + invacorr3.a42*gradrhop1.x + invacorr3.a43*gradrhop1.y + invacorr3.a44*gradrhop1.z);
            rhofinal=(rhoghost + grx*dpos.x + gry*dpos.y + grz*dpos.z);
          }
          else if(a_corr3.a11>0){//-Determinant is small but a11 is nonzero, 0th order ANGELO.
            rhofinal=float(rhop1/a_corr3.a11);
          }
        }
        //-Store the results.
        rhofinal=(rhofinal!=FLT_MAX? rhofinal: CTE.rhopzero);
        //-SlipMode==SLIP_Vel0 (DBC vel=0)
        velrho[p1].w=rhofinal;
      }
    }
  }
}


//==============================================================================
/// Calculates extrapolated data on boundary particles from fluid domain for mDBC.
/// Calcula datos extrapolados en el contorno para mDBC.
//==============================================================================
template<TpKernel tker,bool sim2d> void Interaction_MdbcCorrectionT2(unsigned n
  ,unsigned nbound,const StDivDataGpu& dvd,const tdouble3& mapposmin
  ,const double2* posxy,const double* posz,const float4* poscell
  ,const typecode* code,const unsigned* idp,const float3* boundnor
  ,float4* velrho,hipStream_t stm)
{
  const int2* beginendcellfluid=dvd.beginendcell+dvd.cellfluid;
  //-Interaction GhostBoundaryNodes-Fluid.
  if(n){
    const unsigned bsbound=128;
    dim3 sgridb=cusph::GetSimpleGridSize(n,bsbound);
    KerInteractionMdbcCorrection_Fast <tker,sim2d> <<<sgridb,bsbound,0,stm>>>
      (n,nbound,Double3(mapposmin),dvd.poscellsize
      ,poscell,dvd.scelldiv,dvd.nc,dvd.cellzero,beginendcellfluid
      ,posxy,posz,code,idp,boundnor,velrho);
  }
}
//==============================================================================
template<TpKernel tker> void Interaction_MdbcCorrectionT(bool simulate2d
  ,unsigned n,unsigned nbound,const StDivDataGpu& dvd,const tdouble3& mapposmin
  ,const double2* posxy,const double* posz,const float4* poscell
  ,const typecode* code,const unsigned* idp,const float3* boundnor
  ,float4* velrho,hipStream_t stm)
{
  if(simulate2d){
    Interaction_MdbcCorrectionT2 <tker,true > (n,nbound,dvd
      ,mapposmin,posxy,posz,poscell,code,idp,boundnor,velrho,stm);
  }
  else{
    Interaction_MdbcCorrectionT2 <tker,false> (n,nbound,dvd
      ,mapposmin,posxy,posz,poscell,code,idp,boundnor,velrho,stm);
  }
}
//==============================================================================
/// Calculates extrapolated data on boundary particles from fluid domain for mDBC.
/// Calcula datos extrapolados en el contorno para mDBC.
//==============================================================================
void Interaction_MdbcCorrection(TpKernel tkernel,bool simulate2d,unsigned n
  ,unsigned nbound,const StDivDataGpu& dvd,const tdouble3& mapposmin
  ,const double2* posxy,const double* posz,const float4* poscell
  ,const typecode* code,const unsigned* idp,const float3* boundnor
  ,float4* velrho,hipStream_t stm)
{
  switch(tkernel){
    case KERNEL_Wendland:{ const TpKernel tker=KERNEL_Wendland;
      Interaction_MdbcCorrectionT <tker> (simulate2d,n,nbound
        ,dvd,mapposmin,posxy,posz,poscell,code,idp,boundnor,velrho,stm);
    }break;
#ifndef DISABLE_KERNELS_EXTRA
    case KERNEL_Cubic:{ const TpKernel tker=KERNEL_Cubic;
      Interaction_MdbcCorrectionT <tker> (simulate2d,n,nbound
        ,dvd,mapposmin,posxy,posz,poscell,code,idp,boundnor,velrho,stm);
    }break;
#endif
    default: throw "Kernel unknown at Interaction_MdbcCorrection().";
  }
}

//<vs_m2dbc_ini>
//##############################################################################
//# Kernels for mDBC2 interaction.
//# Kernels para interaccion mDBC2.
//##############################################################################
//------------------------------------------------------------------------------
/// Perform Pressure cloning for mDBC2.
//------------------------------------------------------------------------------
//  const float pressfinal=KerMdbc2PressClone(sim2d,rhoghost,bnormalp1,gravity,motace,dpos);
__device__ float KerMdbc2PressClone(bool sim2d,const float rhoghost,float3 bnormalp1
  ,const float3 gravity,const float3 motacep1,const float3 dpos)
{
  float pressfinal=0.f;
  if(sim2d){
    const float pghost=float(CTE.cs0*CTE.cs0*(rhoghost-CTE.rhopzero));
    const float norm=sqrt(bnormalp1.x*bnormalp1.x + bnormalp1.z*bnormalp1.z);
    const float normx=bnormalp1.x/norm; 
    const float normz=bnormalp1.z/norm;
    const float normpos=dpos.x*normx + dpos.z*normz;
    const float3 force=make_float3(gravity.x-motacep1.x,0,gravity.z-motacep1.z);
    const float normforce=CTE.rhopzero*(force.x*normx + force.z*normz);
    pressfinal=pghost+normforce*normpos;
  }
  else{
    const float pghost=float(CTE.cs0*CTE.cs0*(rhoghost-CTE.rhopzero));
    const float norm=sqrt(bnormalp1.x*bnormalp1.x + bnormalp1.y*bnormalp1.y + bnormalp1.z*bnormalp1.z);
    const float normx=bnormalp1.x/norm;
    const float normy=bnormalp1.y/norm;
    const float normz=bnormalp1.z/norm;
    const float normpos=dpos.x*normx + dpos.y*normy + dpos.z*normz;
    const float3 force=make_float3(gravity.x-motacep1.x,gravity.y-motacep1.y,gravity.z-motacep1.z);
    const float normforce=CTE.rhopzero*(force.x*normx + force.y*normy + force.z*normz);
    pressfinal=pghost+normforce*normpos;
  }
  return(pressfinal);
}
//------------------------------------------------------------------------------
/// Calculates the infinity norm of a 3x3 matrix.
//------------------------------------------------------------------------------
__device__ float KerMdbc2InfNorm3x3(tmatrix3f mat){
  const float row1=float(fabs(mat.a11) + fabs(mat.a12) + fabs(mat.a13));
  const float row2=float(fabs(mat.a21) + fabs(mat.a22) + fabs(mat.a23));
  const float row3=float(fabs(mat.a31) + fabs(mat.a32) + fabs(mat.a33));
  const float infnorm=max(row1,max(row2,row3));
  return(infnorm);
}
//------------------------------------------------------------------------------
/// Calculates the infinity norm of a 4x4 matrix.
//------------------------------------------------------------------------------
__device__ float KerMdbc2InfNorm4x4(tmatrix4f mat){
  const float row1=float(fabs(mat.a11) + fabs(mat.a12) + fabs(mat.a13) + fabs(mat.a14));
  const float row2=float(fabs(mat.a21) + fabs(mat.a22) + fabs(mat.a23) + fabs(mat.a24));
  const float row3=float(fabs(mat.a31) + fabs(mat.a32) + fabs(mat.a33) + fabs(mat.a34));
  const float row4=float(fabs(mat.a41) + fabs(mat.a42) + fabs(mat.a43) + fabs(mat.a44));
  const float infnorm=max(row1,max(row2,max(row3,row4)));
  return(infnorm);
}
//------------------------------------------------------------------------------
/// Calculates tangent velocity
//------------------------------------------------------------------------------
__device__ float3 KerMdbc2TangenVel(const float3& boundnor,const float3& velfinal){
  const float snormal=sqrt(boundnor.x*boundnor.x + boundnor.y*boundnor.y + boundnor.z*boundnor.z);
  const float bnormalx=boundnor.x/snormal;
  const float bnormaly=boundnor.y/snormal;
  const float bnormalz=boundnor.z/snormal;
	const float veldotnorm=velfinal.x*bnormalx + velfinal.y*bnormaly + velfinal.z*bnormalz;
  const float3 tangentvel=make_float3(velfinal.x-veldotnorm*bnormalx,
                                      velfinal.y-veldotnorm*bnormaly,
                                      velfinal.z-veldotnorm*bnormalz);
  return(tangentvel);
}
//------------------------------------------------------------------------------
/// Perform interaction between ghost node of selected boundary and fluid.
//------------------------------------------------------------------------------
template<TpKernel tker,bool sim2d,TpSlipMode tslip,bool sp>
  __global__ void KerInteractionMdbc2Correction_Fast
  (unsigned n,unsigned nbound,float3 gravity
  ,double3 mapposmin,float poscellsize,const float4* poscell
  ,int scelldiv,int4 nc,int3 cellzero,const int2* beginendcellfluid
  ,const double2* posxy,const double* posz,const typecode* code
  ,const unsigned* idp,const float3* boundnor,const float3* motionvel
  ,const float3* motionace,float4* velrho,byte* boundmode,float3* tangenvel)
{
  const unsigned p1=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p1<n){
    const float3 bnormalp1=boundnor[p1];
    if(bnormalp1.x!=0 || bnormalp1.z!=0 || bnormalp1.y!=0){
      float rhofinal=FLT_MAX;
      float3 velrhofinal=make_float3(0,0,0);
      float sumwab=0;
      float submerged=0;                   //-Not for Vel0.
      const float3 motacep1=motionace[p1]; //-Not for Vel0.

      //-Calculates ghost node position.
      double3 gposp1=make_double3(posxy[p1].x+bnormalp1.x,posxy[p1].y+bnormalp1.y,posz[p1]+bnormalp1.z);
      //-Corrected interface Position.
      gposp1=(CTE.periactive!=0? KerUpdatePeriodicPos(gposp1): gposp1); 
      const float4 gpscellp1=KerComputePosCell(gposp1,mapposmin,poscellsize);

      //-Initializes variables for calculation.
      float rhop1=0;
      float3 gradrhop1=make_float3(0,0,0);
      float3 velp1=make_float3(0,0,0);                              // -Only for velocity
      tmatrix3f a_corr2; if(sim2d) cumath::Tmatrix3fReset(a_corr2); //-Only for 2D.
      tmatrix4f a_corr3; if(!sim2d)cumath::Tmatrix4fReset(a_corr3); //-Only for 3D.
    
      //-Obtains neighborhood search limits.
      int ini1,fin1,ini2,fin2,ini3,fin3;
      cunsearch::InitCte(gposp1.x,gposp1.y,gposp1.z,scelldiv,nc,cellzero,ini1,fin1,ini2,fin2,ini3,fin3);

      //-Boundary-Fluid interaction.
      for(int c3=ini3;c3<fin3;c3+=nc.w)for(int c2=ini2;c2<fin2;c2+=nc.x){
        unsigned pini,pfin=0;  cunsearch::ParticleRange(c2,c3,ini1,fin1,beginendcellfluid,pini,pfin);
        if(pfin)for(unsigned p2=pini;p2<pfin;p2++){
          const float4 pscellp2=poscell[p2];
          float drx=gpscellp1.x-pscellp2.x + CTE.poscellsize*(PSCEL_GetfX(gpscellp1.w)-PSCEL_GetfX(pscellp2.w));
          float dry=gpscellp1.y-pscellp2.y + CTE.poscellsize*(PSCEL_GetfY(gpscellp1.w)-PSCEL_GetfY(pscellp2.w));
          float drz=gpscellp1.z-pscellp2.z + CTE.poscellsize*(PSCEL_GetfZ(gpscellp1.w)-PSCEL_GetfZ(pscellp2.w));
          const float rr2=drx*drx+dry*dry+drz*drz;
          if(rr2<=CTE.kernelsize2 && CODE_IsFluid(code[p2])){//-Only with fluid particles (including inout).
            //-Computes kernel.
            float fac;
            const float wab=cufsph::GetKernel_WabFac<tker>(rr2,fac);
            const float frx=fac*drx,fry=fac*dry,frz=fac*drz; //-Gradients.

            //===== Get mass and volume of particle p2 =====
            const float4 velrhop2=velrho[p2];
            float massp2=CTE.massf;
            const float volp2=massp2/velrhop2.w;

            //===== Check if ghost node is submerged =====
            submerged-=volp2*(drx*frx + dry*fry + drz*frz);

            //===== Density and its gradient =====
            rhop1+=massp2*wab;
            gradrhop1.x+=massp2*frx;
            gradrhop1.y+=massp2*fry;
            gradrhop1.z+=massp2*frz;

            //===== Kernel values multiplied by volume =====
            const float vwab=wab*volp2;
            sumwab+=vwab;
            const float vfrx=frx*volp2;
            const float vfry=fry*volp2;
            const float vfrz=frz*volp2;

            //===== Velocity =====
            velp1.x+=vwab*velrhop2.x;
            velp1.y+=vwab*velrhop2.y;
            velp1.z+=vwab*velrhop2.z;

            //===== Matrix A for correction =====
            if(sim2d){
              a_corr2.a11+=vwab;  a_corr2.a12-=drx*vwab;  a_corr2.a13-=drz*vwab;
              a_corr2.a21+=vfrx;  a_corr2.a22-=drx*vfrx;  a_corr2.a23-=drz*vfrx;
              a_corr2.a31+=vfrz;  a_corr2.a32-=drx*vfrz;  a_corr2.a33-=drz*vfrz;
            }
            else{
              a_corr3.a11+=vwab;  a_corr3.a12-=drx*vwab;  a_corr3.a13-=dry*vwab;  a_corr3.a14-=drz*vwab;
              a_corr3.a21+=vfrx;  a_corr3.a22-=drx*vfrx;  a_corr3.a23-=dry*vfrx;  a_corr3.a24-=drz*vfrx;
              a_corr3.a31+=vfry;  a_corr3.a32-=drx*vfry;  a_corr3.a33-=dry*vfry;  a_corr3.a34-=drz*vfry;
              a_corr3.a41+=vfrz;  a_corr3.a42-=drx*vfrz;  a_corr3.a43-=dry*vfrz;  a_corr3.a44-=drz*vfrz;
            }
          }
        }
      }

      //-Store the results.
      //--------------------
      if(submerged>0.f){
        boundmode[p1]=BMODE_MDBC2;
        const float3 dpos=make_float3(-bnormalp1.x,-bnormalp1.y,-bnormalp1.z); //-Boundary particle position - ghost node position.
        if(sim2d){//-2D simulation.
          if(sumwab<0.1f){ //-If kernel sum is small use shepherd density and vel0.
            //-Trying to avoid too negative pressures.
            rhofinal=max(CTE.rhopzero,(rhop1/a_corr2.a11));
          }
          else{//-Chech if matrix is invertible and well conditioned.
            const double determ=(sp? cumath::Determinant3x3(a_corr2): 
                                     cumath::Determinant3x3dbl(a_corr2));
            if(fabs(determ)>=0.001){//-Use 1e-3f (first_order).
              const tmatrix3f invacorr2=(sp? cumath::InverseMatrix3x3(a_corr2,determ):
                                             cumath::InverseMatrix3x3dbl(a_corr2,determ));
              //-Calculate the scaled condition number
              const float infnorma   =KerMdbc2InfNorm3x3(a_corr2);
              const float infnormainv=KerMdbc2InfNorm3x3(invacorr2);
              const float condinf=CTE.dp*CTE.dp * infnorma * infnormainv;
              if(condinf<=50 && !CODE_IsFlexStrucFlex(code[p1]))//-If matrix is well conditioned use matrix inverse for density and shepherd for velocity.
                rhofinal=float(invacorr2.a11*rhop1 + invacorr2.a12*gradrhop1.x + invacorr2.a13*gradrhop1.z);
              else//-If ill conditioned use shepherd.
                rhofinal=float(rhop1/a_corr2.a11);
            }
            else//-If not invertible use shepherd for density.
              rhofinal=float(rhop1/a_corr2.a11);
          }
          //-Final density according to press.
          const float pressfinal=KerMdbc2PressClone(sim2d,rhofinal,bnormalp1,gravity,motacep1,dpos);
          rhofinal=CTE.rhopzero+float(pressfinal/(CTE.cs0*CTE.cs0));
          //-velocity with Shepherd Sum.
          velrhofinal.x=float(velp1.x/a_corr2.a11);
          velrhofinal.y=0.f;
          velrhofinal.z=float(velp1.z/a_corr2.a11);
        }
        else{//-3D simulation.
          //-Density with pressure cloning.
          if(sumwab<0.1f){//-If kernel sum is small use shepherd for density.
            //-Trying to avoid too negative pressures for empty kernel.
            rhofinal=max(CTE.rhopzero,float(rhop1/a_corr3.a11));
          }
          else{
            const double determ=(sp? cumath::Determinant4x4(a_corr3): 
                                     cumath::Determinant4x4dbl(a_corr3));
            if(fabs(determ)>=0.001){
              const tmatrix4f invacorr3=(sp? cumath::InverseMatrix4x4(a_corr3,determ):
                                             cumath::InverseMatrix4x4dbl(a_corr3,determ));
              //-Calculate the scaled condition number.
              const float infnorma   =KerMdbc2InfNorm4x4(a_corr3);
              const float infnormainv=KerMdbc2InfNorm4x4(invacorr3);
              const float condinf=CTE.dp*CTE.dp*infnorma*infnormainv;
              if(condinf<=50 && !CODE_IsFlexStrucFlex(code[p1]))//-If matrix is well conditioned use matrix inverse for density.
                rhofinal=float(invacorr3.a11*rhop1 + invacorr3.a12*gradrhop1.x + invacorr3.a13*gradrhop1.y + invacorr3.a14*gradrhop1.z);
              else//-Matrix is not well conditioned, use shepherd for density.
                rhofinal=float(rhop1/a_corr3.a11);
            }
            else//-Matrix is not invertible use shepherd for density.
              rhofinal=float(rhop1/a_corr3.a11);
          }
          //-Final density according to press.
          const float pressfinal=KerMdbc2PressClone(sim2d,rhofinal,bnormalp1,gravity,motacep1,dpos);
          rhofinal=CTE.rhopzero+float(pressfinal/(CTE.cs0*CTE.cs0));
          //-Velocity with Shepherd sum.
          velrhofinal.x=float(velp1.x/a_corr3.a11);
          velrhofinal.y=float(velp1.y/a_corr3.a11);
          velrhofinal.z=float(velp1.z/a_corr3.a11);
        }

        //-Store the results.
        if(tslip==SLIP_NoSlip){//-No-Slip: vel = 2*motion - ghost
          const float3 v=motionvel[p1];
          const float3 v2=make_float3(v.x+v.x - velrhofinal.x,
                                      v.y+v.y - velrhofinal.y,
                                      v.z+v.z - velrhofinal.z); 
          #ifndef MDBC2_KEEPVEL
            velrho[p1]=make_float4(v2.x,v2.y,v2.z,rhofinal);
          #else
            velrho[p1].w=rhofinal;
          #endif
          tangenvel[p1]=KerMdbc2TangenVel(bnormalp1,v2);
        }
        else if (tslip==SLIP_FreeSlip) {//-Free slip: vel = ghost vel
          // copy velocity from ghost node
          const float3 v2 = make_float3(velrhofinal.x, velrhofinal.y, velrhofinal.z);
          #ifndef MDBC2_KEEPVEL
            velrho[p1] = make_float4(v2.x, v2.y, v2.z, rhofinal);
          #else
            velrho[p1].w = rhofinal;
          #endif
            tangenvel[p1] = KerMdbc2TangenVel(bnormalp1, v2);
        }
      }
      else{//-If unsubmerged switch off boundary particle.
        boundmode[p1]=BMODE_MDBC2OFF;
        const float3 v=motionvel[p1];
        #ifndef MDBC2_KEEPVEL
          velrho[p1]=make_float4(v.x,v.y,v.z,CTE.rhopzero);
        #else
          velrho[p1].w=CTE.rhopzero;
        #endif
        tangenvel[p1]=KerMdbc2TangenVel(bnormalp1,v);
      }
    }
  }
}

//==============================================================================
/// Calculates extrapolated data on boundary particles from fluid domain for mDBC.
/// Calcula datos extrapolados en el contorno para mDBC.
//==============================================================================
template<TpKernel tker,bool sim2d,TpSlipMode tslip> void Interaction_Mdbc2CorrectionT2
  (unsigned n,unsigned nbound,const tfloat3 gravity
  ,const StDivDataGpu& dvd,const tdouble3& mapposmin,const double2* posxy
  ,const double* posz,const float4* poscell,const typecode* code
  ,const unsigned* idp,const float3* boundnor,const float3* motionvel
  ,const float3* motionace,float4* velrho,byte* boundmode,float3* tangenvel
  ,hipStream_t stm)
{
  const int2* beginendcellfluid=dvd.beginendcell+dvd.cellfluid;
  //-Interaction GhostBoundaryNodes-Fluid.
  if(n){
    const unsigned bsbound=128;
    dim3 sgridb=cusph::GetSimpleGridSize(n,bsbound);
    const bool usefloat=false;
    KerInteractionMdbc2Correction_Fast <tker,sim2d,tslip,usefloat> <<<sgridb,bsbound,0,stm>>>
      (n,nbound,Float3(gravity),Double3(mapposmin),dvd.poscellsize
      ,poscell,dvd.scelldiv,dvd.nc,dvd.cellzero,beginendcellfluid
      ,posxy,posz,code,idp,boundnor,motionvel,motionace
      ,velrho,boundmode,tangenvel);
  }
}
//==============================================================================
template<TpKernel tker> void Interaction_Mdbc2CorrectionT(bool simulate2d
  ,TpSlipMode slipmode,unsigned n,unsigned nbound,const tfloat3 gravity
  ,const StDivDataGpu& dvd,const tdouble3& mapposmin,const double2* posxy
  ,const double* posz,const float4* poscell,const typecode* code
  ,const unsigned* idp,const float3* boundnor,const float3* motionvel
  ,const float3* motionace,float4* velrho,byte* boundmode,float3* tangenvel
  ,hipStream_t stm)
{
  switch(slipmode){
    case SLIP_NoSlip: { const TpSlipMode tslip = SLIP_NoSlip;
      if (simulate2d) {
          const bool sim2d = true;
          Interaction_Mdbc2CorrectionT2 <tker, sim2d, tslip>(n, nbound, gravity
              , dvd, mapposmin, posxy, posz, poscell, code, idp, boundnor, motionvel
              , motionace, velrho, boundmode, tangenvel, stm);
      }
      else {
          const bool sim2d = false;
          Interaction_Mdbc2CorrectionT2 <tker, sim2d, tslip>(n, nbound, gravity
              , dvd, mapposmin, posxy, posz, poscell, code, idp, boundnor, motionvel
              , motionace, velrho, boundmode, tangenvel, stm);
      }
    }break;
    case SLIP_FreeSlip: { const TpSlipMode tslip = SLIP_FreeSlip;
      if (simulate2d) {
          const bool sim2d = true;
          Interaction_Mdbc2CorrectionT2 <tker, sim2d, tslip>(n, nbound, gravity
              , dvd, mapposmin, posxy, posz, poscell, code, idp, boundnor, motionvel
              , motionace, velrho, boundmode, tangenvel, stm);
      }
      else {
          const bool sim2d = false;
          Interaction_Mdbc2CorrectionT2 <tker, sim2d, tslip>(n, nbound, gravity
              , dvd, mapposmin, posxy, posz, poscell, code, idp, boundnor, motionvel
              , motionace, velrho, boundmode, tangenvel, stm);
      }
    }break;
    default: throw "SlipMode unknown at Interaction_Mdbc2CorrectionT().";
  }
}
//==============================================================================
/// Calculates extrapolated data on boundary particles from fluid domain for mDBC.
/// Calcula datos extrapolados en el contorno para mDBC.
//==============================================================================
void Interaction_Mdbc2Correction(TpKernel tkernel,bool simulate2d
  ,TpSlipMode slipmode,unsigned n,unsigned nbound,const tfloat3 gravity
  ,const StDivDataGpu& dvd,const tdouble3& mapposmin,const double2* posxy
  ,const double* posz,const float4* poscell,const typecode* code
  ,const unsigned* idp,const float3* boundnor,const float3* motionvel
  ,const float3* motionace,float4* velrho,byte* boundmode,float3* tangenvel
  ,hipStream_t stm)
{
  switch(tkernel){
    case KERNEL_Wendland:{ const TpKernel tker=KERNEL_Wendland;
      Interaction_Mdbc2CorrectionT <tker> (simulate2d,slipmode,n,nbound,gravity
        ,dvd,mapposmin,posxy,posz,poscell,code,idp,boundnor,motionvel,motionace
        ,velrho,boundmode,tangenvel,stm);
    }break;
#ifndef DISABLE_KERNELS_EXTRA
    case KERNEL_Cubic:{ const TpKernel tker=KERNEL_Cubic;
      Interaction_Mdbc2CorrectionT <tker> (simulate2d,slipmode,n,nbound,gravity
        ,dvd,mapposmin,posxy,posz,poscell,code,idp,boundnor,motionvel,motionace
        ,velrho,boundmode,tangenvel,stm);
    }break;
#endif
    default: throw "Kernel unknown at Interaction_Mdbc2Correction().";
  }
}

//------------------------------------------------------------------------------
/// Copy motion velocity and compute acceleration of moving particles.
//------------------------------------------------------------------------------
__global__ void KerCopyMotionVelAce(unsigned n,double dt,const unsigned* ridpmot
  ,const float4* velrho,float3* motionvel,float3* motionace)
{
  const unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    const int pid=ridpmot[p];
    if(pid>=0){
      //-Computes acceleration and copy new velocity.
      const float3 mvel0=motionvel[pid];
      const float4 v=velrho[pid];
      motionace[pid]=make_float3(float((double(v.x)-mvel0.x)/dt),
                                 float((double(v.y)-mvel0.y)/dt),
                                 float((double(v.z)-mvel0.z)/dt));
      motionvel[pid]=make_float3(v.x,v.y,v.z);
    }
  }
}

//==============================================================================
/// Copy motion velocity and compute acceleration of moving particles.
//==============================================================================
void CopyMotionVelAce(unsigned nmoving,double dt,const unsigned* ridpmot
  ,const float4* velrho,float3* motionvel,float3* motionace)
{
  dim3 sgrid=GetSimpleGridSize(nmoving,SPHBSIZE);
  KerCopyMotionVelAce <<<sgrid,SPHBSIZE>>> (nmoving,dt,ridpmot,velrho
    ,motionvel,motionace);
}
//<vs_m2dbc_end>


}

//##############################################################################
//# Kernels for PreLoop and Advanced shifting.
//# Kernels para PreLoop y Advanced shifting.
//##############################################################################
// #include "JSphGpu_preloop_iker.cu"

#include "JSphGpu_preloop_iker.h"

namespace cusph{

//------------------------------------------------------------------------------
/// Saves particle index to access to the parent of periodic particles.
//------------------------------------------------------------------------------
__global__ void KerPeriodicSaveParent(unsigned n,unsigned pini
  ,const unsigned* listp,unsigned* periparent)
{
  const unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    const unsigned pnew=p+pini;
    const unsigned rp=listp[p];
    const unsigned pcopy=(rp&0x7FFFFFFF);
    periparent[pnew]=pcopy;
  }
}
//==============================================================================
/// Saves particle index to access to the parent of periodic particles.
//==============================================================================
void PeriodicSaveParent(unsigned n,unsigned pini,const unsigned* listp
  ,unsigned* periparent)
{
  if(n){
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    KerPeriodicSaveParent <<<sgrid,SPHBSIZE>>> (n,pini,listp,periparent);
  }
}

//------------------------------------------------------------------------------
/// Updates PreLoop variables in periodic particles.
//------------------------------------------------------------------------------
 __global__ void KerPeriPreLoopCorr(unsigned n,unsigned pinit
  ,const unsigned* periparent,unsigned* fstype,float4* shiftvel)
{
  const unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    const unsigned p1=p+pinit;//-Number of particle.
    const unsigned pp=periparent[p1];
    if(pp!=UINT_MAX)fstype[p]=fstype[pp];
    if(pp!=UINT_MAX)shiftvel[p]=shiftvel[pp];
  }
}
//==============================================================================
/// Updates PreLoop variables in periodic particles.
//==============================================================================
void PeriPreLoopCorr(unsigned n,unsigned pinit
  ,const unsigned* periparent,unsigned* fstype,float4* shiftvel)
{
  if(n){
    const unsigned bsize=256;
    dim3 sgrid=GetSimpleGridSize(n,bsize);
    //-Calculates Kernel Gradient Correction.
    KerPeriPreLoopCorr <<<sgrid,bsize>>> (n,pinit,periparent,fstype,shiftvel);
  }
}

//------------------------------------------------------------------------------
/// Interaction of a particle with a set of particles. (Fluid/Float-Fluid/Float/Bound)
/// Realiza la interaccion de una particula con un conjunto de ellas. (Fluid/Float-Fluid/Float/Bound)
//------------------------------------------------------------------------------
template<TpKernel tker> __device__ void KerComputeNormalsBox(bool boundp2
  ,unsigned p1,const unsigned& pini,const unsigned& pfin,const float4* poscell
  ,const float4* velrhop,const typecode* code,float massp2,const float4& pscellp1
  ,float& fs_treshold,float3& gradc,tmatrix3f& lcorr,unsigned& neigh,float& pou
  ,const float* ftomassp)
{
  const float w0=cufsph::GetKernel_Wab<tker>(CTE.dp*CTE.dp);
  for(int p2=pini;p2<pfin;p2++){
  const float4 pscellp2=poscell[p2];
    const float drx=pscellp1.x-pscellp2.x + CTE.poscellsize*(PSCEL_GetfX(pscellp1.w)-PSCEL_GetfX(pscellp2.w));
    const float dry=pscellp1.y-pscellp2.y + CTE.poscellsize*(PSCEL_GetfY(pscellp1.w)-PSCEL_GetfY(pscellp2.w));
    const float drz=pscellp1.z-pscellp2.z + CTE.poscellsize*(PSCEL_GetfZ(pscellp1.w)-PSCEL_GetfZ(pscellp2.w));
    const double rr2=drx*drx+dry*dry+drz*drz;
    if(rr2<=CTE.kernelsize2 && rr2>=ALMOSTZERO){
      //-Computes kernel.
      const float fac=cufsph::GetKernel_Fac<tker>(rr2);
      const float frx=fac*drx,fry=fac*dry,frz=fac*drz; //-Gradients.
      // float4 velrhop2=velrhop[p2];
      const float rhopp2= float(velrhop[p2].w);
      //-Velocity derivative (Momentum equation).

      bool ftp2;
      float ftmassp2;    //-Contains mass of floating body or massf if fluid. | Contiene masa de particula floating o massp2 si es bound o fluid.
      const typecode cod=code[p2];
      ftp2=CODE_IsFloating(cod);
      ftmassp2=(ftp2? ftomassp[CODE_GetTypeValue(cod)]: massp2);

      const float vol2=(ftp2 ? float(ftmassp2/rhopp2) : float(massp2/rhopp2));
      neigh++;

      const float dot3=drx*frx+dry*fry+drz*frz;
      gradc.x+=vol2*frx;
      gradc.y+=vol2*fry;
      gradc.z+=vol2*frz;

      fs_treshold-=vol2*dot3;
      lcorr.a11+=-drx*frx*vol2; lcorr.a12+=-drx*fry*vol2; lcorr.a13+=-drx*frz*vol2;
      lcorr.a21+=-dry*frx*vol2; lcorr.a22+=-dry*fry*vol2; lcorr.a23+=-dry*frz*vol2;
      lcorr.a31+=-drz*frx*vol2; lcorr.a32+=-drz*fry*vol2; lcorr.a33+=-drz*frz*vol2;

      const float wab=cufsph::GetKernel_Wab<tker>(rr2);
      pou+=wab*vol2;       
    }
  }
}

//==============================================================================
/// Perform interaction between particles: Fluid/Float-Fluid/Float or Fluid/Float-Bound
/// Realiza interaccion entre particulas: Fluid/Float-Fluid/Float or Fluid/Float-Bound
//==============================================================================
template<TpKernel tker> __global__ void KerComputeNormals(unsigned n,unsigned pinit
  ,int scelldiv,int4 nc,int3 cellzero,const int2* begincell,unsigned cellfluid
  ,const unsigned* dcell,const float4* poscell,const float4* velrhop
  ,const typecode* code,unsigned* fstype,float3* fsnormal,bool simulate2d
  ,float4* shiftvel,const float* ftomassp,const unsigned* listp)
{
  const unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    const unsigned p1=listp[p];   
    //-Obtains basic data of particle p1.
    const float4 pscellp1=poscell[p1];
    const float4 velrhop1=velrhop[p1];
      
    float fs_treshold=0;                              //-Divergence of the position.
    float3 gradc=make_float3(0,0,0);                  //-Gradient of the concentration
    float pou=0;                                      //-Partition of unity.
    unsigned neigh=0;                                 //-Number of neighbours.
      
    tmatrix3f lcorr; cumath::Tmatrix3fReset(lcorr);             //-Correction matrix.
    tmatrix3f lcorr_inv; cumath::Tmatrix3fReset(lcorr_inv);     //-Inverse of the correction matrix.

    //-Calculate approx. number of neighbours when uniform distribution (in the future on the constant memory?)
    float Nzero=0;
    if(simulate2d)Nzero=(3.141592)*CTE.kernelsize2/(CTE.dp*CTE.dp);
    else          Nzero=(4.f/3.f)*(3.141592)*CTE.kernelsize2*CTE.kernelsize/(CTE.dp*CTE.dp*CTE.dp);

    //-Copy the value of shift to gradc. For single resolution is zero, but in Vres take in account virtual stencil.
    gradc=make_float3(shiftvel[p1].x,shiftvel[p1].y,shiftvel[p1].z); 
    
    //-Obtains neighborhood search limits.
    int ini1,fin1,ini2,fin2,ini3,fin3;
    cunsearch::InitCte(dcell[p1],scelldiv,nc,cellzero,ini1,fin1,ini2,fin2,ini3,fin3);

    //-Interaction with fluids.
    ini3+=cellfluid; fin3+=cellfluid;
    for(int c3=ini3;c3<fin3;c3+=nc.w)for(int c2=ini2;c2<fin2;c2+=nc.x){
      unsigned pini,pfin=0;  cunsearch::ParticleRange(c2,c3,ini1,fin1,begincell,pini,pfin);
      if(pfin){
        KerComputeNormalsBox<tker>(false,p1,pini,pfin,poscell,velrhop,code,CTE.massf
          ,pscellp1,fs_treshold,gradc,lcorr,neigh,pou,ftomassp);
      }
    }

    //-Interaction with boundaries.
    ini3-=cellfluid; fin3-=cellfluid;
    for(int c3=ini3;c3<fin3;c3+=nc.w)for(int c2=ini2;c2<fin2;c2+=nc.x){
      unsigned pini,pfin=0;  cunsearch::ParticleRange(c2,c3,ini1,fin1,begincell,pini,pfin);
      if(pfin){
        KerComputeNormalsBox<tker>(false,p1,pini,pfin,poscell,velrhop,code,CTE.massb
          ,pscellp1,fs_treshold,gradc,lcorr,neigh,pou,ftomassp);
      }
    }
    //-Find particles that are probably on the free-surface.
    unsigned fstypep1=0;
    if(simulate2d){
      if(fs_treshold<1.7)fstypep1=2;
      if(fs_treshold<1.1 && Nzero/float(neigh)<0.4f)fstypep1=3;
    }
    else{
      if(fs_treshold<2.75)fstypep1=2;
      if(fs_treshold<1.8 && Nzero/float(neigh)<0.4f)fstypep1=3;
    }
    fstype[p1]=fstypep1;

    //-Add the contribution of the particle itself
    pou+=cufsph::GetKernel_Wab<tker>(0.f)*CTE.massf/velrhop1.w;

    //-Calculation of the inverse of the correction matrix (Don't think there is a better way, create function for Determinant2x2 for clarity?).
    if(simulate2d){
      tmatrix2f lcorr2d;
      tmatrix2f lcorr2d_inv;
      lcorr2d.a11=lcorr.a11; lcorr2d.a12=lcorr.a13;
      lcorr2d.a21=lcorr.a31; lcorr2d.a22=lcorr.a33;
      float lcorr_det=(lcorr2d.a11*lcorr2d.a22-lcorr2d.a12*lcorr2d.a21);
      lcorr2d_inv.a11=lcorr2d.a22/lcorr_det; lcorr2d_inv.a12=-lcorr2d.a12/lcorr_det;
      lcorr2d_inv.a22=lcorr2d.a11/lcorr_det; lcorr2d_inv.a21=-lcorr2d.a21/lcorr_det;
      lcorr_inv.a11=lcorr2d_inv.a11;  lcorr_inv.a13=lcorr2d_inv.a12;
      lcorr_inv.a31=lcorr2d_inv.a21;  lcorr_inv.a33=lcorr2d_inv.a22;
    }
    else{
      const float determ=cumath::Determinant3x3(lcorr);
      lcorr_inv=cumath::InverseMatrix3x3(lcorr,determ);
    }

    //-Correction of the gradient of concentration.
    float3 gradc1=make_float3(0,0,0);    
    gradc1.x=gradc.x*lcorr_inv.a11+gradc.y*lcorr_inv.a12+gradc.z*lcorr_inv.a13;
    gradc1.y=gradc.x*lcorr_inv.a21+gradc.y*lcorr_inv.a22+gradc.z*lcorr_inv.a23;
    gradc1.z=gradc.x*lcorr_inv.a31+gradc.y*lcorr_inv.a32+gradc.z*lcorr_inv.a33;    
    float gradc_norm=sqrt(gradc1.x*gradc1.x+gradc1.y*gradc1.y+gradc1.z*gradc1.z);
    //-Set normal.
    fsnormal[p1].x=-gradc1.x/gradc_norm;
    fsnormal[p1].y=-gradc1.y/gradc_norm;
    fsnormal[p1].z=-gradc1.z/gradc_norm;
  }
}

 
//------------------------------------------------------------------------------
/// Obtain the list of particle that are probably on the free-surface.
//------------------------------------------------------------------------------
__global__ void KerCountFreeSurface(unsigned n,unsigned pini
  ,const unsigned* fstype,unsigned* listp)
{
  extern __shared__ unsigned slist[];
  if(!threadIdx.x)slist[0]=0;
  __syncthreads();
  const unsigned pp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(pp<n){
    const unsigned p=pp+pini;
    if(fstype[p]>1) slist[atomicAdd(slist,1)+1]=p;
  }
  __syncthreads();
  const unsigned ns=slist[0];
  __syncthreads();
  if(!threadIdx.x && ns)slist[0]=atomicAdd((listp+n),ns);
  __syncthreads();
  if(threadIdx.x<ns){
    const unsigned cp=slist[0]+threadIdx.x;
    listp[cp]=slist[threadIdx.x+1];
  }
}

//==============================================================================
/// Compute free-surface particles and their normals.
//==============================================================================
void ComputeFSNormals(TpKernel tkernel,bool simulate2d,unsigned bsfluid
  ,unsigned fluidini,unsigned fluidnum,StDivDataGpu& dvd,const unsigned* dcell
  ,const double2* posxy,const double* posz,const float4* poscell
  ,const float4* velrho,const typecode* code,const float* ftomassp
  ,float4* shiftvel,unsigned* fstype,float3* fsnormal,unsigned* listp
  ,hipStream_t stm)
{
  //-Creates list with free-surface particle (normal and periodic).
  unsigned count=0;
  if(fluidnum){
    hipMemset(listp+fluidnum,0,sizeof(unsigned));
    dim3 sgridf=GetSimpleGridSize(fluidnum,bsfluid);
    const unsigned smem=(bsfluid+1)*sizeof(unsigned); 
    KerCountFreeSurface <<<sgridf,bsfluid,smem,stm>>> (fluidnum,fluidini,fstype,listp);
  }
  hipMemcpy(&count,listp+fluidnum,sizeof(unsigned),hipMemcpyDeviceToHost);
  hipDeviceSynchronize();

  if(count){
    dim3 sgridf=GetSimpleGridSize(count,bsfluid);
    switch(tkernel){
      case KERNEL_Wendland:{ const TpKernel tker=KERNEL_Wendland;
        KerComputeNormals <tker> <<<sgridf,bsfluid,0,stm>>> (count,fluidini
          ,dvd.scelldiv,dvd.nc,dvd.cellzero,dvd.beginendcell,dvd.cellfluid,dcell
          ,poscell,velrho,code,fstype,fsnormal,simulate2d,shiftvel,ftomassp,listp);
      }break;
     #ifndef DISABLE_KERNELS_EXTRA
      case KERNEL_Cubic:{ const TpKernel tker=KERNEL_Cubic;
        KerComputeNormals <tker> <<<sgridf,bsfluid,0,stm>>> (count,fluidini
          ,dvd.scelldiv,dvd.nc,dvd.cellzero,dvd.beginendcell,dvd.cellfluid,dcell
          ,poscell,velrho,code,fstype,fsnormal,simulate2d,shiftvel,ftomassp,listp);
      }break;
     #endif
      default: throw "Kernel unknown at ComputeFSNormals().";
    }
  }
  hipDeviceSynchronize();
}

//------------------------------------------------------------------------------
/// Interaction of a particle with a set of particles. (Fluid/Float-Fluid/Float/Bound)
/// Realiza la interaccion de una particula con un conjunto de ellas. (Fluid/Float-Fluid/Float/Bound)
//------------------------------------------------------------------------------
__device__ void KerScanUmbrellaRegionBox(bool boundp2,unsigned p1
  ,const unsigned& pini,const unsigned& pfin,const float4* poscell
  ,const float4& pscellp1,bool& fs_flag,const float3* fsnormal,bool simulate2d)
{
  for(int p2=pini;p2<pfin;p2++){
    const float4 pscellp2=poscell[p2];
    const float drx=pscellp1.x-pscellp2.x + CTE.poscellsize*(PSCEL_GetfX(pscellp1.w)-PSCEL_GetfX(pscellp2.w));
    const float dry=pscellp1.y-pscellp2.y + CTE.poscellsize*(PSCEL_GetfY(pscellp1.w)-PSCEL_GetfY(pscellp2.w));
    const float drz=pscellp1.z-pscellp2.z + CTE.poscellsize*(PSCEL_GetfZ(pscellp1.w)-PSCEL_GetfZ(pscellp2.w));
    const double rr2=drx*drx+dry*dry+drz*drz;
    if(rr2<=CTE.kernelsize2 && rr2>=ALMOSTZERO){
      const float3 posq=make_float3(fsnormal[p1].x*CTE.kernelh,fsnormal[p1].y*CTE.kernelh,fsnormal[p1].z*CTE.kernelh);
      if(rr2>2.f*CTE.kernelh*CTE.kernelh){
        const float drxq=-drx-posq.x;
        const float dryq=-dry-posq.y;
        const float drzq=-drz-posq.z;
        const float rrq=sqrt(drxq*drxq+dryq*dryq+drzq*drzq);
        if(rrq<CTE.kernelh)fs_flag=true;
      }
      else{
        if(simulate2d){
          const float drxq=-drx-posq.x;
          const float drzq=-drz-posq.z;
          const float3 normalq=make_float3(drxq*fsnormal[p1].x,0,drzq*fsnormal[p1].z);
          const float3 tangq=make_float3(-drxq*fsnormal[p1].z,0,drzq*fsnormal[p1].x);
          const float normalqnorm=sqrt(normalq.x*normalq.x+normalq.z*normalq.z);
          const float tangqnorm=sqrt(tangq.x*tangq.x+tangq.z*tangq.z);
          if(normalqnorm+tangqnorm<CTE.kernelh) fs_flag=true;
        }
        else{
          float rrr=1.f/sqrt(rr2);
          const float arccosin=acos((-drx*fsnormal[p1].x*rrr-dry*fsnormal[p1].y*rrr-drz*fsnormal[p1].z*rrr));
          if(arccosin<0.785398f)fs_flag=true;
        }
      }
    }
  }
}

//==============================================================================
/// Interaction of Fluid-Fluid/Bound & Bound-Fluid.
/// Interaccion Fluid-Fluid/Bound & Bound-Fluid.
//==============================================================================
__global__ void KerScanUmbrellaRegion(unsigned n,unsigned pinit
  ,int scelldiv,int4 nc,int3 cellzero,const int2* begincell,unsigned cellfluid
  ,const unsigned* dcell,const float4 *poscell,const typecode* code
  ,bool simulate2d,const float3* fsnormal,const unsigned* listp,unsigned* fstype)
{
  const unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    const unsigned p1=listp[p];   
    bool fs_flag=false;
    //-Obtains basic data of particle p1.
    const float4 pscellp1=poscell[p1];

    //-Obtains neighborhood search limits.
    int ini1,fin1,ini2,fin2,ini3,fin3;
    cunsearch::InitCte(dcell[p1],scelldiv,nc,cellzero,ini1,fin1,ini2,fin2,ini3,fin3);

    //-Interaction with fluids.
    ini3+=cellfluid; fin3+=cellfluid;
    for(int c3=ini3;c3<fin3;c3+=nc.w)for(int c2=ini2;c2<fin2;c2+=nc.x){
      unsigned pini,pfin=0;  cunsearch::ParticleRange(c2,c3,ini1,fin1,begincell,pini,pfin);
      if(pfin)
        KerScanUmbrellaRegionBox(false,p1,pini,pfin,poscell,pscellp1,fs_flag,fsnormal,simulate2d);
    }

    //-Interaction with boundaries.
    ini3-=cellfluid; fin3-=cellfluid;
    for(int c3=ini3;c3<fin3;c3+=nc.w)for(int c2=ini2;c2<fin2;c2+=nc.x){
      unsigned pini,pfin=0;  cunsearch::ParticleRange(c2,c3,ini1,fin1,begincell,pini,pfin);
      if(pfin)
        KerScanUmbrellaRegionBox(true,p1,pini,pfin,poscell,pscellp1,fs_flag,fsnormal,simulate2d);
    }
    //-If particle was present in umbrella region, change the code of the particle.
    if(fs_flag && fstype[p1]==2)fstype[p1]=0;
    //-Periodic particle are internal by default.
    if(CODE_IsPeriodic(code[p1]))fstype[p1]=0;
  }
}

//==============================================================================
/// Scan Umbrella region to identify free-surface particle.
//==============================================================================
void ComputeUmbrellaRegion(TpKernel tkernel,bool simulate2d
  ,unsigned bsfluid,unsigned fluidini,unsigned fluidnum,StDivDataGpu& dvd
  ,const unsigned* dcell,const float4* poscell,const typecode* code
  ,const float3* fsnormal,unsigned* listp,unsigned* fstype,hipStream_t stm)
{
  //-Obtain the list of particle that are probably on the free-surface (in ComputeUmbrellaRegion maybe is unnecessary).
  unsigned count=0;
  if(fluidnum){
    hipMemset(listp+fluidnum,0,sizeof(unsigned));
    dim3 sgridf=GetSimpleGridSize(fluidnum,bsfluid);
    const unsigned smem=(bsfluid+1)*sizeof(unsigned); 
    KerCountFreeSurface <<<sgridf,bsfluid,smem,stm>>> (fluidnum,fluidini,fstype,listp);
  }
  hipMemcpy(&count,listp+fluidnum,sizeof(unsigned),hipMemcpyDeviceToHost);
  hipDeviceSynchronize();
  //-Scan Umbrella region on selected free-surface particles.
  if(count){
    dim3 sgridf=GetSimpleGridSize(count,bsfluid);
    KerScanUmbrellaRegion <<<sgridf,bsfluid,0,stm>>> (count,fluidini
      ,dvd.scelldiv,dvd.nc,dvd.cellzero,dvd.beginendcell,dvd.cellfluid
      ,dcell,poscell,code,simulate2d,fsnormal,listp,fstype);
  }
  hipDeviceSynchronize();
}

//==============================================================================
/// Interaction of a particle with a set of particles. (Fluid/Float-Fluid/Float/Bound)
//==============================================================================
template<TpKernel tker,bool simulate2d,bool shiftadv>
__device__ void KerPreLoopInteractionBox(bool boundp2,unsigned p1
  ,const unsigned& pini,const unsigned& pfin,const float4* poscell
  ,const float4* velrhop,const typecode* code,float massp2,const float4& pscellp1
  ,const float4& velrhop1,const float* ftomassp,float4& shiftposf1,unsigned* fs
  ,float3* fsnormal,bool& nearfs,float& mindist,float& maxarccos,bool& bound_inter
  ,float3& fsnormalp1,float& pou)
{
  for(int p2=pini;p2<pfin;p2++){
    const float4 pscellp2=poscell[p2];
    const float drx=pscellp1.x-pscellp2.x + CTE.poscellsize*(PSCEL_GetfX(pscellp1.w)-PSCEL_GetfX(pscellp2.w));
    const float dry=pscellp1.y-pscellp2.y + CTE.poscellsize*(PSCEL_GetfY(pscellp1.w)-PSCEL_GetfY(pscellp2.w));
    const float drz=pscellp1.z-pscellp2.z + CTE.poscellsize*(PSCEL_GetfZ(pscellp1.w)-PSCEL_GetfZ(pscellp2.w));
    const double rr2=drx*drx+dry*dry+drz*drz;
    if(rr2<=CTE.kernelsize2 && rr2>=ALMOSTZERO){
      const float fac=cufsph::GetKernel_Fac<tker>(rr2);
      const float frx=fac*drx,fry=fac*dry,frz=fac*drz; //-Gradients.
      float4 velrhop2=velrhop[p2];

      bool ftp2;
      float ftmassp2;    //-Contains mass of floating body or massf if fluid. | Contiene masa de particula floating o massp2 si es bound o fluid.
      const typecode cod=code[p2];
      ftp2=CODE_IsFloating(cod);
      ftmassp2=(ftp2? ftomassp[CODE_GetTypeValue(cod)]: massp2);
        
      if(shiftadv){
        const float massrho=(boundp2 ? CTE.massb/velrhop2.w : (ftmassp2)/velrhop2.w);

        //-Compute gradient of concentration and partition of unity.        
        shiftposf1.x+=massrho*frx;    
        shiftposf1.y+=massrho*fry;
        shiftposf1.z+=massrho*frz;          
        const float wab=cufsph::GetKernel_Wab<tker>(rr2);
        shiftposf1.w+=wab*massrho;

        //-Check if the particle is too close to solid or floating object
        if((boundp2 || ftp2)) bound_inter=true;

        //-Check if it close to the free-surface, calculate distance from free-surface and smoothing of free-surface normals.
        if(fs[p2]>1 && fs[p2]<3 && !boundp2 ) {
          nearfs=true;
          mindist=min(sqrt(rr2),mindist);
          pou+=wab*massrho;
          fsnormalp1.x+=fsnormal[p2].x*wab*massrho;
          fsnormalp1.y+=fsnormal[p2].y*wab*massrho;
          fsnormalp1.z+=fsnormal[p2].z*wab*massrho;
        }
        //-Check maximum curvature.
        if(fs[p1]>1 && fs[p2]>1){
          const float norm1=sqrt(fsnormal[p1].x*fsnormal[p1].x+fsnormal[p1].y*fsnormal[p1].y+fsnormal[p1].z*fsnormal[p1].z);
          const float norm2=sqrt(fsnormal[p2].x*fsnormal[p2].x+fsnormal[p2].y*fsnormal[p2].y+fsnormal[p2].z*fsnormal[p2].z);
          maxarccos=max(maxarccos,(acos((fsnormal[p1].x*fsnormal[p2].x+fsnormal[p2].y*fsnormal[p1].y+fsnormal[p2].z*fsnormal[p1].z))));
        }
      }
    }
  }
}

//==============================================================================
/// Interaction of Fluid-Fluid/Bound & Bound-Fluid for models before
/// InteractionForces
//==============================================================================
template<TpKernel tker,bool simulate2d,bool shiftadv>
__global__ void KerPreLoopInteraction(unsigned n,unsigned pinit,int scelldiv
  ,int4 nc,int3 cellzero,const int2* begincell,unsigned cellfluid
  ,const unsigned* dcell,const float4* poscell,const float4* velrhop
  ,const typecode* code,const float* ftomassp,float4* shiftvel,unsigned* fstype
  ,float3* fsnormal,float* fsmindist)
{
  const unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    const unsigned p1=p+pinit;      //-Number of particle.
    //-Obtains basic data of particle p1.
    const float4 pscellp1=poscell[p1];
    const float4 velrhop1=velrhop[p1];
    const float pressp1=cufsph::ComputePressCte(velrhop1.w);
    bool    nearfs=false;                     //-Bool for detecting near free-surface particles. <shiftImproved>
    float4  shiftposp1=make_float4(0,0,0,0);
      
    float mindist=CTE.kernelh;                //-Set Min Distance from free-surface to kernel radius. <shiftImproved>
    float maxarccos=0.0;                      //-Variable for identify high-curvature free-surface particle <shiftImproved>
    bool bound_inter=false;                   //-Variable for identify free-surface that interact with boundary <shiftImproved>
    float3 fsnormalp1=make_float3(0,0,0);     //-Normals for near free-surface particles <shiftImproved>
    unsigned fsp1=fstype[p1];                 //-Free-surface identification code: 0-internal, 1-close to free-surface, 2 free-surface, 3-isolated.
    float   pou=false;                        //-Partition of unity for normal correction.                      <ShiftingAdvanced>
    
    //-Obtains neighborhood search limits.
    int ini1,fin1,ini2,fin2,ini3,fin3;
    cunsearch::InitCte(dcell[p1],scelldiv,nc,cellzero,ini1,fin1,ini2,fin2,ini3,fin3);

    //-Interaction with fluids.
    ini3+=cellfluid; fin3+=cellfluid;
    for(int c3=ini3;c3<fin3;c3+=nc.w)for(int c2=ini2;c2<fin2;c2+=nc.x){
      unsigned pini,pfin=0;  cunsearch::ParticleRange(c2,c3,ini1,fin1,begincell,pini,pfin);
      if(pfin){
        KerPreLoopInteractionBox<tker,simulate2d,shiftadv> (false,p1,pini,pfin
          ,poscell,velrhop,code,CTE.massf,pscellp1,velrhop1,ftomassp,shiftposp1
          ,fstype,fsnormal,nearfs,mindist,maxarccos,bound_inter,fsnormalp1,pou);
      } 
    }

    //-Interaction with bound.
    ini3-=cellfluid; fin3-=cellfluid;
    for(int c3=ini3;c3<fin3;c3+=nc.w)for(int c2=ini2;c2<fin2;c2+=nc.x){
      unsigned pini,pfin=0;  cunsearch::ParticleRange(c2,c3,ini1,fin1,begincell,pini,pfin);
      if(pfin){
        KerPreLoopInteractionBox<tker,simulate2d,shiftadv> (true,p1,pini,pfin
          ,poscell,velrhop,code,CTE.massb,pscellp1,velrhop1,ftomassp,shiftposp1
          ,fstype,fsnormal,nearfs,mindist,maxarccos,bound_inter,fsnormalp1,pou);
      }
    }

    if(shiftadv){
      shiftposp1.w+=cufsph::GetKernel_Wab<tker>(0.0)*CTE.massf/velrhop1.w;
      fsmindist[p1]=mindist;
      //-Assign correct code to near free-surface particle and correct their normals by Shepard's Correction.
      if(fsp1==0 && nearfs){
        if(pou>1e-6){
          fsnormalp1=make_float3(fsnormalp1.x,fsnormalp1.y,fsnormalp1.z);
          float norm=sqrt(fsnormalp1.x*fsnormalp1.x+fsnormalp1.y*fsnormalp1.y+fsnormalp1.z*fsnormalp1.z);
          fsnormal[p1]=make_float3(fsnormalp1.x/norm,fsnormalp1.y/norm,fsnormalp1.z/norm);
        }      
        fstype[p1]=1;
        if(bound_inter) fstype[p1]=3;
      }
      //-Check if free-surface particle interact with bound or has high-curvature.
      if(fsp1==2 && (bound_inter||maxarccos>0.52)) shiftposp1=make_float4(0,0,0,shiftposp1.w);
      //-Compute shifting when <shiftImproved> true
      shiftvel[p1]=shiftposp1;
    }
  }
}
  
//==============================================================================
/// Interaction of Fluid-Fluid/Bound & Bound-Fluid for models before
/// InteractionForces
//==============================================================================
template<TpKernel tker,bool simulate2d,bool shiftadv> void PreLoopInteractionT3
  (unsigned bsfluid,unsigned fluidnum,unsigned fluidini,StDivDataGpu& dvd
  ,const unsigned* dcell,const float4* poscell,const float4* velrho
  ,const typecode* code,const float* ftomassp,float4* shiftvel,unsigned* fstype
  ,float3* fsnormal,float* fsmindist,hipStream_t stm)
{
  if(fluidnum){
    dim3 sgridf=GetSimpleGridSize(fluidnum,bsfluid);
    KerPreLoopInteraction <tker,simulate2d,shiftadv> <<<sgridf,bsfluid,0,stm>>> 
      (fluidnum,fluidini,dvd.scelldiv,dvd.nc,dvd.cellzero,dvd.beginendcell,dvd.cellfluid
      ,dcell,poscell,velrho,code,ftomassp,shiftvel,fstype,fsnormal,fsmindist);
  }
}
//==============================================================================
template<TpKernel tker,bool simulate2d> void PreLoopInteractionT2(bool shiftadv
  ,unsigned bsfluid,unsigned fluidnum,unsigned fluidini,StDivDataGpu& dvd
  ,const unsigned* dcell,const float4* poscell,const float4* velrho
  ,const typecode* code,const float* ftomassp,float4* shiftvel,unsigned* fstype
  ,float3* fsnormal,float* fsmindist,hipStream_t stm)
{
  if(shiftadv){
    PreLoopInteractionT3 <tker,simulate2d,true > (bsfluid
        ,fluidnum,fluidini,dvd,dcell,poscell,velrho,code,ftomassp
        ,shiftvel,fstype,fsnormal,fsmindist,stm);
  }
  else{
    PreLoopInteractionT3 <tker,simulate2d,false> (bsfluid
        ,fluidnum,fluidini,dvd,dcell,poscell,velrho,code,ftomassp
        ,shiftvel,fstype,fsnormal,fsmindist,stm);
  }
}
//==============================================================================
template<TpKernel tker> void PreLoopInteractionT(bool simulate2d,bool shiftadv
  ,unsigned bsfluid,unsigned fluidnum,unsigned fluidini,StDivDataGpu& dvd
  ,const unsigned* dcell,const float4* poscell,const float4* velrho
  ,const typecode* code,const float* ftomassp,float4* shiftvel,unsigned* fstype
  ,float3* fsnormal,float* fsmindist,hipStream_t stm)
{
  if(simulate2d){
    PreLoopInteractionT2 <tker,true > (shiftadv,bsfluid
        ,fluidnum,fluidini,dvd,dcell,poscell,velrho,code,ftomassp
        ,shiftvel,fstype,fsnormal,fsmindist,stm);
  }
  else{
    PreLoopInteractionT2 <tker,false> (shiftadv,bsfluid
        ,fluidnum,fluidini,dvd,dcell,poscell,velrho,code,ftomassp
        ,shiftvel,fstype,fsnormal,fsmindist,stm);
  }
}
//==============================================================================
void PreLoopInteraction(TpKernel tkernel,bool simulate2d,bool shiftadv
  ,unsigned bsfluid,unsigned fluidnum,unsigned fluidini,StDivDataGpu& dvd
  ,const unsigned* dcell,const float4* poscell,const float4* velrho
  ,const typecode* code,const float* ftomassp,float4* shiftvel,unsigned* fstype
  ,float3* fsnormal,float* fsmindist,hipStream_t stm)
{
  switch(tkernel){
    case KERNEL_Wendland:{ const TpKernel tker=KERNEL_Wendland;
      PreLoopInteractionT <tker> (simulate2d,shiftadv,bsfluid
        ,fluidnum,fluidini,dvd,dcell,poscell,velrho,code,ftomassp
        ,shiftvel,fstype,fsnormal,fsmindist,stm);
    }break;
   #ifndef DISABLE_KERNELS_EXTRA
    case KERNEL_Cubic:{ const TpKernel tker=KERNEL_Cubic;
      PreLoopInteractionT <tker> (simulate2d,shiftadv,bsfluid
        ,fluidnum,fluidini,dvd,dcell,poscell,velrho,code,ftomassp
        ,shiftvel,fstype,fsnormal,fsmindist,stm);
    }break;
   #endif
    default: throw "Kernel unknown at PreLoopInteraction().";
  }
  hipDeviceSynchronize();
}

//------------------------------------------------------------------------------
/// Compute shifting velocity for advanced shifting model.
//------------------------------------------------------------------------------
__global__ void KerComputeShiftingVel(unsigned n,unsigned pinit,bool sim2d
  ,float shiftcoef,bool ale,float dt,const unsigned* fstype
  ,const float3* fsnormal,const float* fsmindist,float4* shiftvel)
{
  const unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(p<n){
    const unsigned p1=p+pinit;      //-Number of particle.
    //-Obtains basic data of particle p1.
    const unsigned fstypep1=fstype[p1];
    const float4   shiftp1=shiftvel[p1];
    const float    fsmindistp1=fsmindist[p1];
    const float3   fsnormalp1=fsnormal[p1];
    const float    theta=min(1.f,max(0.f,(fsmindistp1-CTE.kernelsize)/(0.5*CTE.kernelsize-CTE.kernelsize)));
    float4         shift_final=make_float4(0,0,0,0);
    const float    normshift=(fsnormalp1.x*shiftp1.x+fsnormalp1.y*shiftp1.y+fsnormalp1.z*shiftp1.z);

    if(fstypep1==0){
      shift_final=shiftp1;
    }
    else if(fstypep1==1 || fstypep1==2){
      if(ale){
        shift_final.x=shiftp1.x-theta*fsnormalp1.x*normshift;
        shift_final.y=shiftp1.y-theta*fsnormalp1.y*normshift;
        shift_final.z=shiftp1.z-theta*fsnormalp1.z*normshift;
      }
      else{
        shift_final=make_float4(0,0,0,0);
      }
    }
    else if(fstypep1==3){
      shift_final=make_float4(0,0,0,0);
    }

    if(sim2d)shift_final.y=0.f;

    const float rhovar=abs(shiftp1.x*shift_final.x+shiftp1.y*shift_final.y+shiftp1.z*shift_final.z)*min(CTE.kernelh,fsmindistp1);
    const float eps=1e-5f;
    const float umagn_1=shiftcoef*CTE.kernelh/dt;
    const float umagn_2=abs(eps/(2.f*dt*rhovar));
    const float umagn=min(umagn_1,umagn_2)*(min(CTE.kernelh,fsmindistp1)*dt);
    // const float umagn= shiftcoef*CTE.kernelh*fsmindistp1;
    const float maxdist=0.1f*CTE.dp;
    shift_final.x=(fabs(umagn*shift_final.x)<maxdist? umagn*shift_final.x: (umagn*shift_final.x>=0? maxdist: -maxdist));
    shift_final.y=(fabs(umagn*shift_final.y)<maxdist? umagn*shift_final.y: (umagn*shift_final.y>=0? maxdist: -maxdist));
    shift_final.z=(fabs(umagn*shift_final.z)<maxdist? umagn*shift_final.z: (umagn*shift_final.z>=0? maxdist: -maxdist));
    shiftvel[p1].x=(shift_final.x)/dt;
    shiftvel[p1].y=(shift_final.y)/dt;
    shiftvel[p1].z=(shift_final.z)/dt;
  }
}

//==============================================================================
/// Compute shifting velocity for advanced shifting model.
//==============================================================================
void ComputeShiftingVel(unsigned bsfluid,unsigned fluidnum,unsigned fluidini
  ,bool sim2d,float shiftcoef,bool ale,float dt,const unsigned* fstype
  ,const float3* fsnormal,const float* fsmindist,float4* shiftvel
  ,hipStream_t stm)
{
  if(fluidnum){
    dim3 sgridf=GetSimpleGridSize(fluidnum,bsfluid);
    KerComputeShiftingVel<<<sgridf,bsfluid,0,stm>>> 
      (fluidnum,fluidini,sim2d,shiftcoef,ale,dt,fstype,fsnormal,fsmindist,shiftvel);
  }
}

}

//##############################################################################
//# Kernels for variable resolution (JSphVRes).
//# Kernels para variable resolution (JSphVRes).
//##############################################################################

// #include "JSphGpu_VRes_iker.cu"
#include "JSphGpu_VRes_iker.h"

namespace cusphvres{
#include "FunctionsBasic_iker.h"
#include "FunctionsMath_iker.h"
#include "FunSphKernel_iker.h"
#include "FunctionsGeo3d_iker.h"
#include "FunSphEos_iker.h"

#undef _JCellSearch_iker_
#include "JCellSearch_iker.h"

__device__ void MovePoint(double2 rxy,double rz,double2& rxy_t,double& rz_t,tmatrix4f mat){
  rxy.x-=mat.a14;   rxy.y-=mat.a24;   rz-=mat.a34;
  rxy_t.x=rxy.x*mat.a11+rxy.y*mat.a21+rz*mat.a31/* +mat.a14 */;
  rxy_t.y=rxy.x*mat.a12+rxy.y*mat.a22+rz*mat.a32/* +mat.a24 */;
  rz_t   =rxy.x*mat.a13+rxy.y*mat.a23+rz*mat.a33/* +mat.a34 */;
}

__device__ bool KerBufferInZone(double2 rxy,double rz,double3 boxlimitmin,double3 boxlimitmax)
{
  return(boxlimitmin.x<=rxy.x && rxy.x<=boxlimitmax.x && boxlimitmin.y<=rxy.y && rxy.y<=boxlimitmax.y && boxlimitmin.z<=rz && rz<=boxlimitmax.z);
}

//------------------------------------------------------------------------------
/// Creates list with current buffer particles and normal (no periodic) fluid in
/// buffer zones (update its code).
//------------------------------------------------------------------------------
__global__ void KerBufferCreateList(unsigned n,unsigned pini,const double3 boxlimitmininner,const double3 boxlimitmaxinner
  ,const double3 boxlimitminouter,const double3 boxlimitmaxouter,const bool inner,const double2 *posxy,const double *posz
  ,typecode *code,unsigned *listp,tmatrix4f* mat,bool tracking,unsigned nzone)
{
  extern __shared__ unsigned slist[];
  //float *splanes=(float*)(slist+(n+1));
  if(!threadIdx.x)slist[0]=0;
  __syncthreads();
  const unsigned pp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(pp<n){
    const unsigned p=pp+pini;
    const typecode rcode=code[p];
    if(CODE_IsNormal(rcode) && CODE_IsFluid(rcode)){//-It includes only normal fluid particles (no periodic).
      bool select=CODE_IsFluidBuffer(rcode);//-Particles already selected as InOut.
      if(!select){//-Particulas no periodicas y no marcadas como in/out.
        double2 rxy=posxy[p];
        double rz=posz[p];
        if(tracking){
          double2 rxy_t=make_double2(0,0);
          double rz_t=0.0;
          MovePoint(rxy,rz,rxy_t,rz_t,mat[nzone]);
          rxy=rxy_t; rz=rz_t;
        }
        byte zone=255;
        if(KerBufferInZone(rxy,rz,boxlimitminouter,boxlimitmaxouter)&&!KerBufferInZone(rxy,rz,boxlimitmininner,boxlimitmaxinner))zone=byte(nzone);
          if(zone!=255){
            code[p]=CODE_ToFluidBuffer(rcode,nzone)|CODE_TYPE_FLUID_BUFFERNUM; //-Adds 16 to indicate new particle in zone.
            select=true;
          }
      } else{
    	  const byte izone0=byte(CODE_GetIzoneFluidBuffer(rcode));
    	  		const byte izone=(izone0&CODE_TYPE_FLUID_INOUT015MASK);
    	  		if(izone !=byte(nzone)) select=false;
      }
      if(select)slist[atomicAdd(slist,1)+1]=p; //-Add particle in the list.
    }
  }
  __syncthreads();
  const unsigned ns=slist[0];
  __syncthreads();
  if(!threadIdx.x && ns)slist[0]=atomicAdd((listp+n),ns);
  __syncthreads();
  if(threadIdx.x<ns){
    const unsigned cp=slist[0]+threadIdx.x;
    listp[cp]=slist[threadIdx.x+1];
  }
}


//==============================================================================
/// Creates list with current buffer particles and normal (no periodic) fluid in
/// buffer zones (update its code).
//==============================================================================
unsigned BufferCreateList(bool stable,unsigned n,unsigned pini,const tdouble3 boxlimitmininner,const tdouble3 boxlimitmaxinner,
          const tdouble3 boxlimitminouter,const tdouble3 boxlimitmaxouter,const bool inner,const double2 *posxy,const double *posz
	        ,typecode *code,unsigned *listp,tmatrix4f* mat,bool tracking,unsigned nzone)
{
  unsigned count=0;
  if(n){
    //-listp size list initialized to zero.
    //-Inicializa tamanho de lista listp a cero.
    hipMemset(listp+n,0,sizeof(unsigned));
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    const unsigned smem=(SPHBSIZE+1)*sizeof(unsigned); //-All fluid particles can be in in/out area and one position for counter.
    KerBufferCreateList <<<sgrid,SPHBSIZE,smem>>> (n,pini,Double3(boxlimitmininner),Double3(boxlimitmaxinner),Double3(boxlimitminouter)
    	      ,Double3(boxlimitmaxouter),inner,posxy,posz,code,listp,mat,tracking,nzone);
    hipMemcpy(&count,listp+n,sizeof(unsigned),hipMemcpyDeviceToHost);
    hipDeviceSynchronize();
  }
  return(count);
}

//------------------------------------------------------------------------------
/// Creates list with current buffer particles and normal (no periodic) fluid in
/// buffer zones (update its code).
//------------------------------------------------------------------------------
__global__ void KerBufferCreateListInit(unsigned n,unsigned pini
  ,const double3 boxlimitmininner,const double3 boxlimitmaxinner,const double3 boxlimitminouter,const double3 boxlimitmaxouter,const double3 boxlimitminmid,const double3 boxlimitmaxmid,const bool inner
  ,const double2 *posxy,const double *posz
  ,typecode *code,unsigned *listp,tmatrix4f mat,bool tracking,unsigned nzone)
{
  extern __shared__ unsigned slist[];
  if(!threadIdx.x)slist[0]=0;
  __syncthreads();
  bool inner1=inner;
  const unsigned pp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(pp<n){
    const unsigned p=pp+pini;
    const typecode rcode=code[p];
    if(CODE_IsNormal(rcode) ){//-It includes only normal fluid particles (no periodic).
      bool select=CODE_IsFluidBuffer(rcode);//-Particles already selected as InOut.
      if(!select){//-Particulas no periodicas y no marcadas como in/out.
        double2 rxy=posxy[p];
        double rz=posz[p];
        if(tracking){
          double2 rxy_t=make_double2(0,0);
          double rz_t=0.0;
          MovePoint(rxy,rz,rxy_t,rz_t,mat);
          rxy=rxy_t; rz=rz_t;
        }
        byte zone=255;
        if(KerBufferInZone(rxy,rz,boxlimitminouter,boxlimitmaxouter)&&!KerBufferInZone(rxy,rz,boxlimitmininner,boxlimitmaxinner)&& CODE_IsFluid(rcode))zone=byte(nzone);
          if(zone!=255){
            code[p]=CODE_ToFluidBuffer(rcode,nzone)|CODE_TYPE_FLUID_BUFFERNUM; //-Adds 16 to indicate new particle in zone.
            select=true;
          }
        if((!KerBufferInZone(rxy,rz,boxlimitminouter,boxlimitmaxouter)&& inner) ||(KerBufferInZone(rxy,rz,boxlimitmininner,boxlimitmaxinner)&&!inner)){
        select=false;
        if(!(KerBufferInZone(rxy,rz,boxlimitminmid,boxlimitmaxmid)!=inner1)&& CODE_IsFluid(rcode)) {
        	code[p]=CODE_SetOutIgnore(rcode);//CODE_ToFluidFixed(rcode,nzone);
        }
        else code[p]=CODE_SetOutIgnore(rcode);
        }

      }
      if(select)slist[atomicAdd(slist,1)+1]=p; //-Add particle in the list.
    }
  }
  __syncthreads();
  const unsigned ns=slist[0];
  __syncthreads();
  if(!threadIdx.x && ns)slist[0]=atomicAdd((listp+n),ns);
  __syncthreads();
  if(threadIdx.x<ns){
    const unsigned cp=slist[0]+threadIdx.x;
    listp[cp]=slist[threadIdx.x+1];
  }
}

//==============================================================================
/// Creates list with current buffer particles and normal (no periodic) fluid in
/// buffer zones (update its code).
//==============================================================================
unsigned BufferCreateListInit(bool stable,unsigned n,unsigned pini
  ,const tdouble3 boxlimitmininner,const tdouble3 boxlimitmaxinner,const tdouble3 boxlimitminouter,const tdouble3 boxlimitmaxouter,const tdouble3 boxlimitminmid,const tdouble3 boxlimitmaxmid,const bool inner,const double2 *posxy,const double *posz
  ,typecode *code,unsigned *listp,tmatrix4f mat,bool tracking,unsigned nzone)
{
  unsigned count=0;
  if(n){
    //-listp size list initialized to zero.
    //-Inicializa tamanho de lista listp a cero.
    hipMemset(listp+n,0,sizeof(unsigned));
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    const unsigned smem=(SPHBSIZE+1)*sizeof(unsigned); //-All fluid particles can be in in/out area and one position for counter.
    KerBufferCreateListInit <<<sgrid,SPHBSIZE,smem>>> (n,pini,Double3(boxlimitmininner),Double3(boxlimitmaxinner),Double3(boxlimitminouter)
      ,Double3(boxlimitmaxouter),Double3(boxlimitminmid),Double3(boxlimitmaxmid),inner,posxy,posz,code,listp,mat,tracking,nzone);
    hipMemcpy(&count,listp+n,sizeof(unsigned),hipMemcpyDeviceToHost);
    hipDeviceSynchronize();
  }
  return(count);
}

//------------------------------------------------------------------------------
/// Creates list with current buffer particles and normal (no periodic) fluid in
/// buffer zones (update its code).
//------------------------------------------------------------------------------
__global__ void KerBufferCheckNormals(TpBoundary tboundary,unsigned n,unsigned pini
  ,const double3 boxlimitmininner,const double3 boxlimitmaxinner
  ,const double3 boxlimitminouter,const double3 boxlimitmaxouter
  ,const bool inner,const double2 *posxy,const double *posz,const float3* boundnor
  ,typecode *code,unsigned *listp,tmatrix4f* mat,bool tracking,unsigned nzone)
{
  extern __shared__ unsigned slist[];
  if(!threadIdx.x)slist[0]=0;
  __syncthreads();
  const unsigned pp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(pp<n){
    const unsigned p=pp+pini;
    bool select=false;//-Particles already selected as InOut.
    double2 rxy=posxy[p];
    double rz=posz[p];
    if(tracking){
      double2 rxy_t=make_double2(0,0);
      double rz_t=0.0;
      MovePoint(rxy,rz,rxy_t,rz_t,mat[nzone]);
      rxy=rxy_t; rz=rz_t;
    }
    if(KerBufferInZone(rxy,rz,boxlimitminouter,boxlimitmaxouter)&&!KerBufferInZone(rxy,rz,boxlimitmininner,boxlimitmaxinner)){
      if(tboundary==BC_DBC)select=true;
      else{
        const float3 bnormalp1=boundnor[p];
        if(bnormalp1.x==0 && bnormalp1.y==0 && bnormalp1.z==0){
          select=true;
        }
      }
    }      
    if(select)slist[atomicAdd(slist,1)+1]=p; //-Add particle in the list.
  }
  __syncthreads();
  const unsigned ns=slist[0];
  __syncthreads();
  if(!threadIdx.x && ns)slist[0]=atomicAdd((listp+n),ns);
  __syncthreads();
  if(threadIdx.x<ns){
    const unsigned cp=slist[0]+threadIdx.x;
    listp[cp]=slist[threadIdx.x+1];
  }
}

//==============================================================================
/// Creates list with current inout particles and normal (no periodic) fluid in
/// inlet/outlet zones (update its code).
//==============================================================================

unsigned BufferCheckNormals(TpBoundary tboundary,bool stable,unsigned n,unsigned pini
  ,const tdouble3 boxlimitmininner,const tdouble3 boxlimitmaxinner
  ,const tdouble3 boxlimitminouter,const tdouble3 boxlimitmaxouter
  ,const bool inner,const double2 *posxy,const double *posz,const float3* boundnor
  ,typecode *code,unsigned *listp,tmatrix4f* mat,bool tracking,unsigned nzone)
{
  unsigned count=0;
  if(n){
    //-listp size list initialized to zero.
    //-Inicializa tamanho de lista listp a cero.
    hipMemset(listp+n,0,sizeof(unsigned));
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    const unsigned smem=(SPHBSIZE+1)*sizeof(unsigned); //-All fluid particles can be in in/out area and one position for counter.
    KerBufferCheckNormals <<<sgrid,SPHBSIZE,smem>>> (tboundary,n,pini,Double3(boxlimitmininner),Double3(boxlimitmaxinner),Double3(boxlimitminouter)
      ,Double3(boxlimitmaxouter),inner,posxy,posz,boundnor,code,listp,mat,tracking,nzone);
    hipMemcpy(&count,listp+n,sizeof(unsigned),hipMemcpyDeviceToHost);
    hipDeviceSynchronize();
  }
  return(count);
}

//------------------------------------------------------------------------------
/// Checks particle position.
/// If particle is moved to fluid zone then it changes to fluid particle and
/// it creates a new in/out particle.
/// If particle is moved out the domain then it changes to ignore particle.
//------------------------------------------------------------------------------
__global__ void KerBufferComputeStep(unsigned n,int *inoutpart,const double2 *posxy,const double *posz
  ,typecode *code,const double3 boxlimitmininner,const double3 boxlimitmaxinner,const double3 boxlimitminouter
  ,const double3 boxlimitmaxouter,const bool inner,tmatrix4f* mat,bool tracking,unsigned nzone)
{
	const unsigned cp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
	if(cp<n){
		typecode cod=0;
		const int p=inoutpart[cp];
		const typecode rcode=code[p];
		const byte izone0=byte(CODE_GetIzoneFluidBuffer(rcode));
		const byte izone=(izone0&CODE_TYPE_FLUID_INOUT015MASK); //-Substract 16 to obtain the actual zone (0-15).
		double2 rxy=posxy[p];
		double rz=(posz[p]);
    if(tracking){
          double2 rxy_t=make_double2(0,0);
          double rz_t=0.0;
          MovePoint(rxy,rz,rxy_t,rz_t,mat[nzone]);
          rxy=rxy_t; rz=rz_t;
        }
		if(izone==byte(nzone)){
			if(izone0>=16){     //-Normal fluid particle in zone buffer
				cod= rcode^0x10 ; //-Converts to buffer particle or not.
				code[p]=cod;
			}
			else{//-Previous buffer fluid particle.
				if((!KerBufferInZone(rxy,rz,boxlimitminouter,boxlimitmaxouter)&& inner) ||(KerBufferInZone(rxy,rz,boxlimitmininner,boxlimitmaxinner)&&!inner)){
					cod=CODE_SetOutIgnore(rcode); //-Particle is moved out domain.
					code[p]=cod;
				}
				if((!KerBufferInZone(rxy,rz,boxlimitminouter,boxlimitmaxouter)&& !inner) ||(KerBufferInZone(rxy,rz,boxlimitmininner,boxlimitmaxinner)&&inner)){
					cod=CODE_TYPE_FLUID; //-Particle become normal;
					code[p]=cod;
				}
			}
		}
	}
}

//==============================================================================
/// Checks particle position.
/// If particle is moved to fluid zone then it changes to fluid particle and
/// it creates a new in/out particle.
/// If particle is moved out the domain then it changes to ignore particle.
//==============================================================================
void BufferComputeStep(unsigned n,int *inoutpart,const double2 *posxy,const double *posz
  ,typecode *code,const tdouble3 boxlimitmininner,const tdouble3 boxlimitmaxinner,const tdouble3 boxlimitminouter,const tdouble3 boxlimitmaxouter,const bool inner,tmatrix4f* mat,bool tracking,unsigned nzone)
{
  if(n){
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    KerBufferComputeStep <<<sgrid,SPHBSIZE>>> (n,inoutpart,posxy,posz,code,Double3(boxlimitmininner),Double3(boxlimitmaxinner),Double3(boxlimitminouter)
    	      ,Double3(boxlimitmaxouter),inner,mat,tracking,nzone);
  }
}

//------------------------------------------------------------------------------
/// Solve a linear system of m_dim unknown and m_dim2 right hand sides with a LU Decomposition.
/// Return the reciprocal of the condition number.
//------------------------------------------------------------------------------
template<typename T = float,const int m_dim, const int m_dim2>
__device__ void LUdecomp_Single(T (&A)[m_dim*m_dim],int (&p)[m_dim]
  ,T (&b)[m_dim*m_dim2],T (&sol)[m_dim2],T &treshold){
  //-Compute the norm of matrix A
  T maxs = 0;
  for(int i=0;i<m_dim;i++){
    T sum = 0;
    for(int j=0;j<m_dim;j++) sum+=fabs(A[i*m_dim+j]);
    maxs =max(maxs, sum);
  }

  //- Initialize permutation array
  for(int i=0;i<m_dim;i++) p[i]=i;
  

  T maxv = 0;
  for(int i=0;i<m_dim;i++){
    int imax=i;
      for(int k=i;k<m_dim;k++){
        if(fabs(A[i+k*m_dim])>maxv){
          maxv=fabs(A[i+k*m_dim]);
          imax=k;
        }
      }

      if(imax!=i){
        int tmp =p[i];
        p[i]    =p[imax];
        p[imax] =tmp;

        for(int j=0;j<m_dim;j++) {
          T temp          =A[i*m_dim+j];
          A[i*m_dim+j]    =A[imax*m_dim+j];
          A[imax*m_dim+j] =temp;
        }
      }

        // LU decomposition
      for (int j=i+1;j<m_dim;j++){
        A[j*m_dim+i] /= A[i*m_dim+i];
        for (int k=i+1;k<m_dim;k++) A[j*m_dim+k]-=A[j*m_dim+i]*A[i*m_dim+k];
      }
  }

  T ia[m_dim*m_dim] = {0};

  //-Compute inverse matrix
  for (int j=0;j<m_dim;j++) {
    for (int i=0;i<m_dim;i++) {
      ia[i*m_dim+j]= p[i]==j ? 1.0 : 0.0;
      for (int k = 0;k<i;k++)ia[i*m_dim+j]-=A[i*m_dim+k]*ia[k*m_dim+j];
  }
    for (int i=m_dim-1;i>=0;i--){
      for (int k=i+1;k<m_dim;k++)
        ia[i*m_dim+j] -= A[i*m_dim+k]*ia[k*m_dim+j];
        ia[i*m_dim+j] /= A[i*m_dim+i];
    }
  }

  //-Compute norm of inverse matrix
  T maxs1 = 0;
  for (int i=0;i<m_dim;i++) {
    T sum=0;
    for (int j=0;j<m_dim;j++) sum += fabs(ia[i*m_dim+j]);
    maxs1 =max(maxs1,sum);
  }

  treshold =1.0f/(maxs*maxs1);

  //-Compute Solution array
  for (int k=0;k<m_dim2;k++)
    for (int i=0;i<m_dim;i++) sol[k]+=ia[i]*b[i+k*m_dim];
        
}
//------------------------------------------------------------------------------
/// Perform interaction between buffer particles and fluid particles. Buffer-Fluid
//------------------------------------------------------------------------------
template <TpKernel tker,bool sim2d,TpVresOrder vrorder,TpVresMethod vrmethod, typename T = float>
__global__ void KerInteractionBufferExtrap_Single(
    unsigned bufferpartcount, const int *bufferpart, int scelldiv, int4 nc, int3 cellzero, const int2 *beginendcellfluid, const float4 *poscell, double3 mapposmin,
    const double2 *posxy, const double *posz, const typecode *code, const unsigned *idp,
    const float4 *velrhop, const double2 *posxyb, const double *poszb, float4 *velrhopg, typecode *code1, float mrthreshold)
{
  const unsigned cp=blockIdx.x*blockDim.x+threadIdx.x; // Number of particle.
  if(cp<bufferpartcount){
    const unsigned p1=bufferpart[cp];

    // Calculates ghost node position.
    double3 posp1 = make_double3(posxyb[p1].x, posxyb[p1].y, poszb[p1]);
    const float4 gpscellp1 = cusph::KerComputePosCell(posp1, mapposmin, CTE.poscellsize);

    // Compute size of the reconstruction matrix and right hand sime.
    constexpr unsigned m_dim  = vrorder==VrOrder_2nd ? (sim2d ? 6 : 10) : (sim2d ? 3 : 4);
    constexpr unsigned m_dim2 = sim2d ? 3: 4;

    T C[m_dim]{0};
    T C1[m_dim]{0};
    T D[m_dim2]{0};

    T A[m_dim*m_dim]{0};
    T B[m_dim*m_dim2]{0};

    // Obtains neighborhood search limits.
    int ini1,fin1,ini2,fin2,ini3,fin3;
    cunsearch::InitCte(posp1.x,posp1.y,posp1.z,scelldiv,nc,cellzero,ini1,fin1,ini2,fin2,ini3,fin3);

    //-Interaction with fluids.
    for(int c3=ini3;c3<fin3;c3+=nc.w)for(int c2=ini2;c2<fin2;c2+=nc.x){
      unsigned pini,pfin=0;  cunsearch::ParticleRange(c2,c3,ini1,fin1,beginendcellfluid,pini,pfin);
      if(pfin)for(unsigned p2=pini;p2<pfin;p2++){
        T drx,dry,drz;
        if (sizeof(T)==sizeof(float)){
          const float4 pscellp2 = poscell[p2];
          drx=gpscellp1.x-pscellp2.x+CTE.poscellsize*(PSCEL_GetfX(gpscellp1.w)-PSCEL_GetfX(pscellp2.w));
          dry=gpscellp1.y-pscellp2.y+CTE.poscellsize*(PSCEL_GetfY(gpscellp1.w)-PSCEL_GetfY(pscellp2.w));
          drz=gpscellp1.z-pscellp2.z+CTE.poscellsize*(PSCEL_GetfZ(gpscellp1.w)-PSCEL_GetfZ(pscellp2.w));
        }else{
          const double2 p2xy=posxy[p2];
          drx=T(posp1.x-p2xy.x);
          dry=T(posp1.y-p2xy.y);
          drz=T(posp1.z-posz[p2]);
        }
        const T rr2=drx*drx+dry*dry+drz*drz;
        if (rr2<=CTE.kernelsize2 && rr2>=ALMOSTZERO && CODE_IsFluid(code[p2]) && !CODE_IsFluidBuffer(code[p2]) && !CODE_IsFluidFixed(code[p2])){
        // Computes kernel.
          float fac;
          float facc;
          const T wab=cufsph::GetKernel_WabFacFacc<tker>(rr2,fac,facc);
          const T frx=drx*fac,fry=dry*fac,frz=drz*fac; //-Gradients.
          T frxx=facc*(drx*drx)/sqrt(rr2)+fac*(dry*dry+drz*drz)/rr2;
          T frzz=facc*(drz*drz)/sqrt(rr2)+fac*(dry*dry+drx*drx)/rr2;
          T fryy=facc*(dry*dry)/sqrt(rr2)+fac*(drz*drz+drx*drx)/rr2;
          T frxz=facc*(drx*drz)/sqrt(rr2)-fac*(drx*drz)/rr2;
          T frxy=facc*(drx*dry)/sqrt(rr2)-fac*(drx*dry)/rr2;
          T fryz=facc*(dry*drz)/sqrt(rr2)-fac*(dry*drz)/rr2;

          float4 velrhopp2 = velrhop[p2];
          // Get mass and volume of particle p2
          T massp2=T(CTE.massf);
          T volp2=massp2/T(velrhopp2.w);

          // Set up C and C1 based on order and sim2d
          if(vrmethod==VrMethod_Mls){
            drx=drx/CTE.kernelh;
            dry=dry/CTE.kernelh;
            drz=drz/CTE.kernelh;
          }

          // Set up C and C1 based on order and sim2d
          if (vrorder==VrOrder_2nd){
            if (sim2d){
              T tempC[] ={T(1.0),drx,drz,drx*drx*T(0.5),drx*drz,drz*drz*T(0.5)};
              T tempC1[]={wab,frx,frz,frxx,frxz,frzz};
              for (int i=0;i<m_dim;++i){
                C[i]  = tempC[i];
                if(vrmethod==VrMethod_Liu) C1[i] = volp2*tempC1[i];
                else C1[i] = volp2*wab*tempC[i];
              }
            }else{
              T tempC[]   ={T(1.0),drx,dry,drz,drx*drx*T(0.5),drx*dry,dry*dry*T(0.5),drx*drz,drz*drz*T(0.5),dry*drz};
              T tempC1[]  ={wab,frx,fry,frz,frxx,frxy,fryy,frxz,frzz,fryz};
              for (int i=0;i<m_dim;++i){
                C[i]  = tempC[i];
                if(vrmethod==VrMethod_Liu) C1[i] = volp2*tempC1[i];
                else C1[i] = volp2*wab*tempC[i];
              }
            }            
          }else{
            if (sim2d){
              T tempC[]   = {T(1.0),-drx,-drz};
              T tempC1[]  = {wab,frx,frz};
              for (int i=0;i<m_dim;++i){
                C[i]  = tempC[i];
                if(vrmethod==VrMethod_Liu) C1[i] = volp2*tempC1[i];
                else C1[i] = volp2*wab*tempC[i];
              }
            }else{
              T tempC[] = {T(1.0),drx,dry,drz};
              T tempC1[] = {wab,frx,fry,frz};
              for (int i=0;i<m_dim;++i){
                C[i]  = tempC[i];
                if(vrmethod==VrMethod_Liu) C1[i] = volp2*tempC1[i];
                else C1[i] = volp2*wab*tempC[i];
              }
            }
          }

              // Set up D
          if (sim2d){
            T tempD[] = {T(velrhop[p2].w),T(velrhop[p2].x),T(velrhop[p2].z)};
            for (int i =0;i<m_dim2;++i)D[i]=tempD[i];
          }else{
            T tempD[] = {T(velrhop[p2].w), T(velrhop[p2].x), T(velrhop[p2].y), T(velrhop[p2].z)};
            for (int i =0;i<m_dim2;++i)D[i]=tempD[i];
          }
          // Accumulate A and B matrices
          for (int i=0;i<m_dim;i++)
            for (int j=0;j<m_dim;j++){
              A[i*m_dim+j]+=C1[i]*C[j];
            }
              
          for (int i=0; i<m_dim2;i++)
            for (int j=0;j<m_dim;j++){
              B[i*m_dim+j]+=D[i]*C1[j];
          }
        }
      }        
    }

    T shep = A[0];                                  ///<Shepard summation.
    T sol[m_dim2]{0};                               ///<Solution array;
    T treshold = T(0);                              ///<condition number;
    T cond = T(0);                                  ///<Scaled reciprocal condition number;
    T kernelh2=CTE.kernelh*CTE.kernelh;             ///<Scaling factor;
    T scaleh= T(0);


    if(shep>T(0.05)){
      if (vrorder==VrOrder_2nd){
        if(vrmethod==VrMethod_Liu) scaleh = kernelh2*kernelh2;
        else scaleh=T(1.0);

        int P[m_dim]{0};

        LUdecomp_Single<T,m_dim,m_dim2>(A,P,B,sol,treshold);

        cond=(T(1.0)/treshold)*scaleh;
      }else if (vrorder==VrOrder_1st){
        if(vrmethod==VrMethod_Liu) scaleh = kernelh2*kernelh2;
        else scaleh=T(1.0);
        int P[m_dim]{0};

        LUdecomp_Single<T,m_dim,m_dim2>(A,P,B,sol,treshold);
        cond=(T(1.0)/treshold)*scaleh;
      }
      if (cond>mrthreshold || vrorder==VrOrder_0th)
      {
        for (unsigned i=0;i<m_dim2;i++)
          sol[i]=B[i*m_dim]/shep;
      }

      if (sim2d){
        velrhopg[p1].w = float(sol[0]);
        velrhopg[p1].x = float(sol[1]);
        velrhopg[p1].z = float(sol[2]);
      }else{
        velrhopg[p1].w = float(sol[0]);
        velrhopg[p1].x = float(sol[1]);
        velrhopg[p1].y = float(sol[2]);
        velrhopg[p1].z = float(sol[3]);
      }
    }
    else{
      code1[p1] = CODE_SetOutIgnore(code1[p1]);
    }
  }
}

//==============================================================================
/// Perform interaction between buffer particles and fluid particles. Buffer-Fluid
//==============================================================================
template<TpKernel tker,bool sim2d,TpVresOrder vrorder>
void Interaction_BufferExtrapT(unsigned bufferpartcount,const int *bufferpart,const StInterParmsbg &t,
	const double2 *posxyb,const double *poszb,float4 *velrhop,typecode *code1,bool fastsingle
  ,const TpVresMethod vrmethod,float mrthreshold)
{
  const StDivDataGpu &dvd=t.divdatag;
  const int2* beginendcellfluid=dvd.beginendcell+dvd.cellfluid;
  //-Interaction GhostBoundaryNodes-Fluid.
  if(bufferpartcount){
    const unsigned bsize=128;
    dim3 sgrid=GetSimpleGridSize(bufferpartcount,bsize);
    if(vrmethod==VrMethod_Liu){
      if(fastsingle)  KerInteractionBufferExtrap_Single<tker,sim2d,vrorder,VrMethod_Liu,float> <<<sgrid,bsize>>> (bufferpartcount,bufferpart,dvd.scelldiv,dvd.nc,dvd.cellzero
        ,beginendcellfluid,t.poscell,Double3(t.mapposmin),t.posxy,t.posz,t.code,t.idp,t.velrho,posxyb,poszb,velrhop,code1,mrthreshold);
      else            KerInteractionBufferExtrap_Single<tker,sim2d,vrorder,VrMethod_Liu,double> <<<sgrid,bsize>>> (bufferpartcount,bufferpart,dvd.scelldiv,dvd.nc,dvd.cellzero
        ,beginendcellfluid,t.poscell,Double3(t.mapposmin),t.posxy,t.posz,t.code,t.idp,t.velrho,posxyb,poszb,velrhop,code1,mrthreshold);     
    }else{
      if(fastsingle)  KerInteractionBufferExtrap_Single<tker,sim2d,vrorder,VrMethod_Mls,float> <<<sgrid,bsize>>> (bufferpartcount,bufferpart,dvd.scelldiv,dvd.nc,dvd.cellzero
        ,beginendcellfluid,t.poscell,Double3(t.mapposmin),t.posxy,t.posz,t.code,t.idp,t.velrho,posxyb,poszb,velrhop,code1,mrthreshold);
      else            KerInteractionBufferExtrap_Single<tker,sim2d,vrorder,VrMethod_Mls,double> <<<sgrid,bsize>>> (bufferpartcount,bufferpart,dvd.scelldiv,dvd.nc,dvd.cellzero
        ,beginendcellfluid,t.poscell,Double3(t.mapposmin),t.posxy,t.posz,t.code,t.idp,t.velrho,posxyb,poszb,velrhop,code1,mrthreshold);       
    }  
  }
}


//==============================================================================
/// Perform interaction between buffer particles and fluid particles. Buffer-Fluid
//==============================================================================
template<TpKernel tker>
void Interaction_BufferExtrap_gt0(unsigned bufferpartcount,const int *bufferpart,const StInterParmsbg &t
  ,const double2 *posxyb,const double *poszb,float4* velrhop,typecode *code1,bool fastsingle
  ,const TpVresOrder vrorder,const TpVresMethod vrmethod,float mrthreshold)
{
  if(t.simulate2d){
    switch(vrorder){
      case VrOrder_0th: Interaction_BufferExtrapT<tker,true,VrOrder_0th>(bufferpartcount,bufferpart,t,posxyb,poszb,velrhop,code1,fastsingle,vrmethod,mrthreshold); break;
      case VrOrder_1st: Interaction_BufferExtrapT<tker,true,VrOrder_1st>(bufferpartcount,bufferpart,t,posxyb,poszb,velrhop,code1,fastsingle,vrmethod,mrthreshold); break;
      case VrOrder_2nd: Interaction_BufferExtrapT<tker,true,VrOrder_2nd>(bufferpartcount,bufferpart,t,posxyb,poszb,velrhop,code1,fastsingle,vrmethod,mrthreshold); break;
    }
  }else{
      switch(vrorder){
      case VrOrder_0th: Interaction_BufferExtrapT<tker,false,VrOrder_0th>(bufferpartcount,bufferpart,t,posxyb,poszb,velrhop,code1,fastsingle,vrmethod,mrthreshold); break;
      case VrOrder_1st: Interaction_BufferExtrapT<tker,false,VrOrder_1st>(bufferpartcount,bufferpart,t,posxyb,poszb,velrhop,code1,fastsingle,vrmethod,mrthreshold); break;
      case VrOrder_2nd: Interaction_BufferExtrapT<tker,false,VrOrder_2nd>(bufferpartcount,bufferpart,t,posxyb,poszb,velrhop,code1,fastsingle,vrmethod,mrthreshold); break;
    }
  } 
}

//==============================================================================
/// Perform interaction between buffer particles and fluid particles. Buffer-Fluid
//==============================================================================
void Interaction_BufferExtrap(unsigned bufferpartcount,const int *bufferpart,const StInterParmsbg &t
	,const double2 *posxyb,const double *poszb,float4* velrhop,typecode *code1,bool fastsingle
  ,const TpVresOrder order,const TpVresMethod vrmethod,float mrthreshold)
{
  switch(t.tkernel){
    case KERNEL_Wendland:
      Interaction_BufferExtrap_gt0<KERNEL_Wendland>(bufferpartcount,bufferpart,t,posxyb,poszb,velrhop,code1,fastsingle,order,vrmethod,mrthreshold);
    break;
#ifndef DISABLE_KERNELS_EXTRA
    case KERNEL_Cubic:
    	// Interaction_BufferExtrap_gt0<KERNEL_Wendland>(bufferpartcount,bufferpart,t,posxyb,poszb,velrhop,code1,fastsingle,order,mrthreshold);
    break;
#endif
    default: throw "Kernel unknown at Interaction_InOutExtrap().";
  }
}

//------------------------------------------------------------------------------
/// Perform interaction between ghost inlet/outlet nodes and fluid particles. GhostNodes-Fluid
/// Realiza interaccion entre ghost inlet/outlet nodes y particulas de fluido. GhostNodes-Fluid
//------------------------------------------------------------------------------
template <TpKernel tker,bool sim2d,TpVresOrder vrorder,TpVresMethod vrmethod, typename T = float>
__global__ void KerInteractionBufferExtrap_SingleFlux
  (unsigned bufferpartcount,unsigned pini,int scelldiv,int4 nc,int3 cellzero,const int2 *beginendcellfluid,const float4* poscell, double3 mapposmin
  ,const double2 *posxy,const double *posz,const typecode *code,const unsigned *idp
  ,const float4 *velrhop,const double2 *posxyb,const double *poszb,float3 *normals,float *fluxes
  ,float3* velflux,double dp,double dt,float mrthreshold,const unsigned cellfluid)
{
  const unsigned cp=blockIdx.x*blockDim.x+threadIdx.x; //-Number of particle.
  if(cp<bufferpartcount){
    const unsigned p1=cp+pini;

      //-Calculates ghost node position.
    double3 posp1 = make_double3(posxyb[p1].x, posxyb[p1].y, poszb[p1]);
    const float4 gpscellp1 = cusph::KerComputePosCell(posp1,mapposmin,CTE.poscellsize);

    // Compute size of the reconstruction matrix and right hand sime.
    constexpr unsigned m_dim  = vrorder==VrOrder_2nd ? (sim2d ? 6 : 10) : (sim2d ? 3 : 4);
    constexpr unsigned m_dim2 = sim2d ? 3: 4;

    T C[m_dim]{0};
    T C1[m_dim]{0};
    T D[m_dim2]{0};

    T A[m_dim*m_dim]{0};
    T B[m_dim*m_dim2]{0};



    
    T ShiftTFS=0;
    T mindist=1000000.0;
    T mindp=min(dp,CTE.dp);
    //-Obtains neighborhood search limits.
    int ini1,fin1,ini2,fin2,ini3,fin3;
    cunsearch::InitCte(posp1.x,posp1.y,posp1.z,scelldiv,nc,cellzero,ini1,fin1,ini2,fin2,ini3,fin3);

    //-Interaction with fluids.
    for(int c3=ini3;c3<fin3;c3+=nc.w)for(int c2=ini2;c2<fin2;c2+=nc.x){
      unsigned pini,pfin=0;  cunsearch::ParticleRange(c2,c3,ini1,fin1,beginendcellfluid,pini,pfin);
      if(pfin)for(unsigned p2=pini;p2<pfin;p2++){
        T drx,dry,drz;
        if (sizeof(T)==sizeof(float)){
          const float4 pscellp2 = poscell[p2];
          drx=gpscellp1.x-pscellp2.x+CTE.poscellsize*(PSCEL_GetfX(gpscellp1.w)-PSCEL_GetfX(pscellp2.w));
          dry=gpscellp1.y-pscellp2.y+CTE.poscellsize*(PSCEL_GetfY(gpscellp1.w)-PSCEL_GetfY(pscellp2.w));
          drz=gpscellp1.z-pscellp2.z+CTE.poscellsize*(PSCEL_GetfZ(gpscellp1.w)-PSCEL_GetfZ(pscellp2.w));
        }else{
          const double2 p2xy=posxy[p2];
          drx=T(posp1.x-p2xy.x);
          dry=T(posp1.y-p2xy.y);
          drz=T(posp1.z-posz[p2]);
        }
        const T rr2=drx*drx+dry*dry+drz*drz;
        if (rr2<=CTE.kernelsize2 && rr2>=ALMOSTZERO){
        // Computes kernel.
          float fac;
          float facc;
          const T wab=cufsph::GetKernel_WabFacFacc<tker>(rr2,fac,facc);
          const T frx=drx*fac,fry=dry*fac,frz=drz*fac; //-Gradients.

          float4 velrhopp2 = velrhop[p2];
          // Get mass and volume of particle p2
          T massp2=T(CTE.massf);
          T volp2=massp2/T(velrhopp2.w);
          ShiftTFS-=volp2*(drx*frx+dry*fry+drz*frz);

        }
      }
    }

    cunsearch::InitCte(posp1.x,posp1.y,posp1.z,scelldiv,nc,cellzero,ini1,fin1,ini2,fin2,ini3,fin3);
    ini3+=cellfluid; fin3+=cellfluid;

    //-Interaction with fluids.
    for(int c3=ini3;c3<fin3;c3+=nc.w)for(int c2=ini2;c2<fin2;c2+=nc.x){
      unsigned pini,pfin=0;  cunsearch::ParticleRange(c2,c3,ini1,fin1,beginendcellfluid,pini,pfin);
      if(pfin)for(unsigned p2=pini;p2<pfin;p2++){
        T drx,dry,drz;
        if (sizeof(T)==sizeof(float)){
          const float4 pscellp2 = poscell[p2];
          drx=gpscellp1.x-pscellp2.x+CTE.poscellsize*(PSCEL_GetfX(gpscellp1.w)-PSCEL_GetfX(pscellp2.w));
          dry=gpscellp1.y-pscellp2.y+CTE.poscellsize*(PSCEL_GetfY(gpscellp1.w)-PSCEL_GetfY(pscellp2.w));
          drz=gpscellp1.z-pscellp2.z+CTE.poscellsize*(PSCEL_GetfZ(gpscellp1.w)-PSCEL_GetfZ(pscellp2.w));
        }else{
          const double2 p2xy=posxy[p2];
          drx=T(posp1.x-p2xy.x);
          dry=T(posp1.y-p2xy.y);
          drz=T(posp1.z-posz[p2]);
        }
        const T rr2=drx*drx+dry*dry+drz*drz;
        if (rr2<=CTE.kernelsize2 && rr2>=ALMOSTZERO && CODE_IsFluid(code[p2]) && !CODE_IsFluidBuffer(code[p2]) && !CODE_IsFluidFixed(code[p2])){
        // Computes kernel.
          float fac;
          float facc;
          const T wab=cufsph::GetKernel_WabFacFacc<tker>(rr2,fac,facc);
          const T frx=drx*fac,fry=dry*fac,frz=drz*fac; //-Gradients.
          T frxx=facc*(drx*drx)/sqrt(rr2)+fac*(dry*dry+drz*drz)/rr2;
          T frzz=facc*(drz*drz)/sqrt(rr2)+fac*(dry*dry+drx*drx)/rr2;
          T fryy=facc*(dry*dry)/sqrt(rr2)+fac*(drz*drz+drx*drx)/rr2;
          T frxz=facc*(drx*drz)/sqrt(rr2)-fac*(drx*drz)/rr2;
          T frxy=facc*(drx*dry)/sqrt(rr2)-fac*(drx*dry)/rr2;
          T fryz=facc*(dry*drz)/sqrt(rr2)-fac*(dry*drz)/rr2;

          float4 velrhopp2 = velrhop[p2];
          // Get mass and volume of particle p2
          T massp2=T(CTE.massf);
          T volp2=massp2/T(velrhopp2.w);
          ShiftTFS-=volp2*(drx*frx+dry*fry+drz*frz);
          mindist=min(mindist,rr2);
          if(vrmethod==VrMethod_Mls){
            drx=drx/CTE.kernelh;
            dry=dry/CTE.kernelh;
            drz=drz/CTE.kernelh;
          }

          // Set up C and C1 based on order and sim2d
          if (vrorder==VrOrder_2nd){
            if (sim2d){
              T tempC[] ={T(1.0),drx,drz,drx*drx*T(0.5),drx*drz,drz*drz*T(0.5)};
              T tempC1[]={wab,frx,frz,frxx,frxz,frzz};
              for (int i=0;i<m_dim;++i){
                C[i]  = tempC[i];
                if(vrmethod==VrMethod_Liu) C1[i] = volp2*tempC1[i];
                else C1[i] = volp2*wab*tempC[i];
              }
            }else{
              T tempC[]   ={T(1.0),drx,dry,drz,drx*drx*T(0.5),drx*dry,dry*dry*T(0.5),drx*drz,drz*drz*T(0.5),dry*drz};
              T tempC1[]  ={wab,frx,fry,frz,frxx,frxy,fryy,frxz,frzz,fryz};
              for (int i=0;i<m_dim;++i){
                C[i]  = tempC[i];
                if(vrmethod==VrMethod_Liu) C1[i] = volp2*tempC1[i];
                else C1[i] = volp2*wab*tempC[i];
              }
            }            
          }else{
            if (sim2d){
              T tempC[]   = {T(1.0),-drx,-drz};
              T tempC1[]  = {wab,frx,frz};
              for (int i=0;i<m_dim;++i){
                C[i]  = tempC[i];
                if(vrmethod==VrMethod_Liu) C1[i] = volp2*tempC1[i];
                else C1[i] = volp2*wab*tempC[i];
              }
            }else{
              T tempC[] = {T(1.0),drx,dry,drz};
              T tempC1[] = {wab,frx,fry,frz};
              for (int i=0;i<m_dim;++i){
                C[i]  = tempC[i];
                if(vrmethod==VrMethod_Liu) C1[i] = volp2*tempC1[i];
                else C1[i] = volp2*wab*tempC[i];
              }
            }
          }

              // Set up D
          if (sim2d){
            T tempD[] = {T(velrhop[p2].w),T(velrhop[p2].x),T(velrhop[p2].z)};
            for (int i =0;i<m_dim2;++i)D[i]=tempD[i];
          }else{
            T tempD[] = {T(velrhop[p2].w), T(velrhop[p2].x), T(velrhop[p2].y), T(velrhop[p2].z)};
            for (int i =0;i<m_dim2;++i)D[i]=tempD[i];
          }
          // Accumulate A and B matrices
          for (int i=0;i<m_dim;i++)
            for (int j=0;j<m_dim;j++){
              A[i*m_dim+j]+=C1[i]*C[j];
            }
              
          for (int i=0; i<m_dim2;i++)
            for (int j=0;j<m_dim;j++){
              B[i*m_dim+j]+=D[i]*C1[j];
          }
        }
      }        
    }

    T shep = A[0];                                  ///<Shepard summation.
    T sol[m_dim2]{0};                               ///<Solution array;
    T treshold = T(0);                              ///<condition number;
    T cond = T(0);                                  ///<Scaled reciprocal condition number;
    T kernelh2=CTE.kernelh*CTE.kernelh;             ///<Scaling factor;
    T scaleh= T(0);

    if(shep>T(0.05)){
      if (vrorder==VrOrder_2nd){
        if(vrmethod==VrMethod_Liu) scaleh = kernelh2*kernelh2;
        else scaleh=T(1.0);

        int P[m_dim]{0};

        LUdecomp_Single<T,m_dim,m_dim2>(A,P,B,sol,treshold);

        cond=(T(1.0)/treshold)*scaleh;
      }else if (vrorder==VrOrder_1st){
        if(vrmethod==VrMethod_Liu) scaleh = kernelh2;
        else scaleh=T(1.0);

        int P[m_dim]{0};

        LUdecomp_Single<T,m_dim,m_dim2>(A,P,B,sol,treshold);
        cond=(T(1.0)/treshold)*scaleh;
      }
      if (cond>mrthreshold || vrorder==VrOrder_0th)
      {
        for (unsigned i=0;i<m_dim2;i++)
          sol[i]=B[i*m_dim]/shep;
      }
          // printf("%f %f %f %f\n",sol[0],sol[1],sol[2],dp);

      if (sim2d){
        if((ShiftTFS>1.5 || sqrt(mindist)<mindp) && fluxes[p1]<0.0)
        fluxes[p1]+=max(0.0,-sol[0]*((-float(velflux[p1].x)+sol[1])*normals[p1].x+(-float(velflux[p1].z)+sol[2])*normals[p1].z)*dp*dt);
        else if((ShiftTFS>1.5 || sqrt(mindist)<mindp))
        fluxes[p1]+=-sol[0]*((-float(velflux[p1].x)+sol[1])*normals[p1].x+(-float(velflux[p1].z)+sol[2])*normals[p1].z)*dp*dt;
      }else{
        if      ((ShiftTFS>2.75 || sqrt(mindist)<mindp)  &&fluxes[p1]<0.0)  fluxes[p1]+=max(0.0,-sol[0]*((-velflux[p1].x+sol[1])*normals[p1].x+(-velflux[p1].y+sol[2])*normals[p1].y+(-velflux[p1].z+sol[3])*normals[p1].z)*dp*dp*dt);
        else if ((ShiftTFS>2.75 || sqrt(mindist)<mindp))                    fluxes[p1]+= -sol[0]*((-velflux[p1].x+sol[1])*normals[p1].x+(-velflux[p1].y+sol[2])*normals[p1].y+(-velflux[p1].z+sol[3])*normals[p1].z)*dp*dp*dt;
          
      }
    }
  }
}


//==============================================================================
/// Perform interaction between ghost inlet/outlet nodes and fluid particles. GhostNodes-Fluid
/// Realiza interaccion entre ghost inlet/outlet nodes y particulas de fluido. GhostNodes-Fluid
//==============================================================================
template<TpKernel tker,bool sim2d,TpVresOrder vrorder>
void Interaction_BufferExtrapFluxT(const StInterParmsbg &t,StrDataVresGpu &vres
	,double dp,double dt,bool fastsingle,const TpVresMethod vrmethod,float mrthreshold)
{
  const StDivDataGpu &dvd=t.divdatag;
  const int2* beginendcellfluid=dvd.beginendcell/* +dvd.cellfluid */;
  //-Interaction GhostBoundaryNodes-Fluid.
  const unsigned n=vres.ntot;
  if(n){
    const unsigned bsize=128;
    dim3 sgrid=GetSimpleGridSize(n,bsize);
    if(vrmethod==VrMethod_Liu){
      if(fastsingle)  KerInteractionBufferExtrap_SingleFlux<tker,sim2d,vrorder,VrMethod_Liu,float> <<<sgrid,bsize>>> (vres.ntot,vres.nini,dvd.scelldiv,dvd.nc,dvd.cellzero
        ,beginendcellfluid,t.poscell,Double3(t.mapposmin),t.posxy,t.posz,t.code,t.idp,t.velrho,vres.ptposxy,vres.ptposz,vres.normals,vres.mass,vres.velmot,dp,dt,mrthreshold,dvd.cellfluid);
      else KerInteractionBufferExtrap_SingleFlux<tker,sim2d,vrorder,VrMethod_Liu,double> <<<sgrid,bsize>>> (vres.ntot,vres.nini,dvd.scelldiv,dvd.nc,dvd.cellzero
        ,beginendcellfluid,t.poscell,Double3(t.mapposmin),t.posxy,t.posz,t.code,t.idp,t.velrho,vres.ptposxy,vres.ptposz,vres.normals,vres.mass,vres.velmot,dp,dt,mrthreshold,dvd.cellfluid);    
    }else{
      if(fastsingle)  KerInteractionBufferExtrap_SingleFlux<tker,sim2d,vrorder,VrMethod_Mls,float> <<<sgrid,bsize>>> (vres.ntot,vres.nini,dvd.scelldiv,dvd.nc,dvd.cellzero
        ,beginendcellfluid,t.poscell,Double3(t.mapposmin),t.posxy,t.posz,t.code,t.idp,t.velrho,vres.ptposxy,vres.ptposz,vres.normals,vres.mass,vres.velmot,dp,dt,mrthreshold,dvd.cellfluid);
      else KerInteractionBufferExtrap_SingleFlux<tker,sim2d,vrorder,VrMethod_Mls,double> <<<sgrid,bsize>>> (vres.ntot,vres.nini,dvd.scelldiv,dvd.nc,dvd.cellzero
        ,beginendcellfluid,t.poscell,Double3(t.mapposmin),t.posxy,t.posz,t.code,t.idp,t.velrho,vres.ptposxy,vres.ptposz,vres.normals,vres.mass,vres.velmot,dp,dt,mrthreshold,dvd.cellfluid);    
    }
  }
}

//==============================================================================
/// Perform interaction between ghost inlet/outlet nodes and fluid particles. GhostNodes-Fluid
/// Realiza interaccion entre ghost inlet/outlet nodes y particulas de fluido. GhostNodes-Fluid
//==============================================================================
template<TpKernel tker>
void Interaction_BufferExtrapFlux_gt0(const StInterParmsbg &t,StrDataVresGpu &vres
	,double dp,double dt,bool fastsingle,const TpVresOrder vrorder,const TpVresMethod vrmethod,float mrthreshold)
{
  if(t.simulate2d){
    switch(vrorder){
      case VrOrder_0th: Interaction_BufferExtrapFluxT<tker,true,VrOrder_0th>(t,vres,dp,dt,fastsingle,vrmethod,mrthreshold); break;
      case VrOrder_1st: Interaction_BufferExtrapFluxT<tker,true,VrOrder_1st>(t,vres,dp,dt,fastsingle,vrmethod,mrthreshold); break;
      case VrOrder_2nd: Interaction_BufferExtrapFluxT<tker,true,VrOrder_2nd>(t,vres,dp,dt,fastsingle,vrmethod,mrthreshold); break;
    }
  }else{
      switch(vrorder){
      case VrOrder_0th: Interaction_BufferExtrapFluxT<tker,false,VrOrder_0th>(t,vres,dp,dt,fastsingle,vrmethod,mrthreshold); break;
      case VrOrder_1st: Interaction_BufferExtrapFluxT<tker,false,VrOrder_1st>(t,vres,dp,dt,fastsingle,vrmethod,mrthreshold); break;
      case VrOrder_2nd: Interaction_BufferExtrapFluxT<tker,false,VrOrder_2nd>(t,vres,dp,dt,fastsingle,vrmethod,mrthreshold); break;
    }
  } 
}

//==============================================================================
/// Perform interaction between ghost inlet/outlet nodes and fluid particles. GhostNodes-Fluid
/// Realiza interaccion entre ghost inlet/outlet nodes y particulas de fluido. GhostNodes-Fluid
//==============================================================================
void Interaction_BufferExtrapFlux(const StInterParmsbg &t,StrDataVresGpu &vres
	,double dp,double dt,bool fastsingle,const TpVresOrder vrorder,const TpVresMethod vrmethod,float mrthreshold)
{
  switch(t.tkernel){
    case KERNEL_Wendland:
      Interaction_BufferExtrapFlux_gt0<KERNEL_Wendland>(t,vres,dp,dt,fastsingle,vrorder,vrmethod,mrthreshold);
    break;
#ifndef DISABLE_KERNELS_EXTRA
    case KERNEL_Cubic:
    	// Interaction_BufferExtrapFlux_gt0<KERNEL_Wendland>(bufferpartcount,pini,t,posxyb,poszb,normals,fluxes,dp,dt,velflux,fastsingle,vrorder,mrthreshold);
    break;
#endif
    default: throw "Kernel unknown at Interaction_InOutExtrap().";
  }
}

//------------------------------------------------------------------------------
/// Correct mass accumulated and avoid generation inside boundaries.
//------------------------------------------------------------------------------
__global__ void KerCheckMassFlux(unsigned n,unsigned pini2,double3 mapposmin,float poscellsize
  ,const float4* poscell,int scelldiv,int4 nc,int3 cellzero,const int2* beginendcellfluid
  ,unsigned cellfluid,const double2* posxy,const double* posz,const typecode *code,const double2 *posxyb
  ,const double *poszb,float3 *normals,float *fluxes)
{
  const unsigned cp=blockIdx.x*blockDim.x+threadIdx.x; //-Number of particle.
  if(cp<n){
    const unsigned p1=cp+pini2;

      //-Calculates ghost node position.
    double3 posp1 = make_double3(posxyb[p1].x, posxyb[p1].y, poszb[p1]);
    const float4 gpscellp1 = cusph::KerComputePosCell(posp1,mapposmin,CTE.poscellsize);

    //-Obtains neighborhood search limits.
    int ini1,fin1,ini2,fin2,ini3,fin3;
    cunsearch::InitCte(posp1.x,posp1.y,posp1.z,scelldiv,nc,cellzero,ini1,fin1,ini2,fin2,ini3,fin3);
    // ini3-=cellfluid; fin3-=cellfluid;
    //-Boundary-Fluid interaction.
    for(int c3=ini3;c3<fin3;c3+=nc.w)for(int c2=ini2;c2<fin2;c2+=nc.x){
      unsigned pini,pfin=0;  cunsearch::ParticleRange(c2,c3,ini1,fin1,beginendcellfluid,pini,pfin);
      if(pfin)for(unsigned p2=pini;p2<pfin;p2++){
        const float4 pscellp2=poscell[p2];
        float drx=gpscellp1.x-pscellp2.x + CTE.poscellsize*(PSCEL_GetfX(gpscellp1.w)-PSCEL_GetfX(pscellp2.w));
        float dry=gpscellp1.y-pscellp2.y + CTE.poscellsize*(PSCEL_GetfY(gpscellp1.w)-PSCEL_GetfY(pscellp2.w));
        float drz=gpscellp1.z-pscellp2.z + CTE.poscellsize*(PSCEL_GetfZ(gpscellp1.w)-PSCEL_GetfZ(pscellp2.w));
        const float rr2=drx*drx+dry*dry+drz*drz;
        if(rr2<CTE.dp*CTE.dp){
          const float3 normalp1=normals[p1];
          const float norm = (-normalp1.x*drx+-normalp1.y*dry-normalp1.z*drz)/sqrt(rr2);
          if(acos(norm)>0.785398) fluxes[p1]=0;      
        }
      }
    }
  }
}

//==============================================================================
/// Correct mass accumulated and avoid generation inside boundaries.
//==============================================================================
void CheckMassFlux(unsigned n,unsigned pini
  ,const StDivDataGpu& dvd,const tdouble3& mapposmin,const double2* posxy
  ,const double* posz,const typecode *code,const float4* poscell,const double2 *posxyb
  ,const double *poszb,float3 *normals,float *fluxes)
{
  if(n){
    const unsigned bsbound=128;
    dim3 sgrid=GetSimpleGridSize(n,bsbound);
    KerCheckMassFlux<<<sgrid,bsbound>>>(n,pini,Double3(mapposmin),dvd.poscellsize
      ,poscell,dvd.scelldiv,dvd.nc,dvd.cellzero,dvd.beginendcell,dvd.cellfluid
      ,posxy,posz,code,posxyb,poszb,normals,fluxes);
  }
}

//------------------------------------------------------------------------------
/// Create list for new buffer particles to create.
/// Crea lista de nuevas particulas buffer a crear.
//------------------------------------------------------------------------------
__global__ void KerNewPartListCreate(unsigned n,unsigned pini,unsigned nmax
  ,float *fluxes,int *bufferpart,double massf)
{
  extern __shared__ unsigned slist[];
  if(!threadIdx.x)slist[0]=0;
  __syncthreads();
  const unsigned cp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(cp<n && fluxes[cp+pini]>massf){
	  fluxes[cp+pini]-=massf;
    slist[atomicAdd(slist,1)+1]=cp;
  }
  __syncthreads();
  const unsigned ns=slist[0];
  __syncthreads();
  if(!threadIdx.x && ns)slist[0]= atomicAdd((bufferpart+nmax),ns);
  __syncthreads();
  if(threadIdx.x<ns){
    const unsigned cp2=slist[0]+threadIdx.x;
    if(cp2<nmax)bufferpart[cp2]=slist[threadIdx.x+1];
  }
}

//==============================================================================
/// Create list for new buffer particles to create at end of buffer[].
/// Returns number of new particles to create.
//==============================================================================
unsigned NewPartListCreate(unsigned n,unsigned pini,unsigned nmax
  ,float *fluxes,int *bufferpart,double massf)
{
  unsigned count=0;
  if(n){
    //-bufferpart size list initialized to zero.
    hipMemset(bufferpart+nmax,0,sizeof(unsigned));
    dim3 sgrid=GetSimpleGridSize(n,SPHBSIZE);
    const unsigned smem=(SPHBSIZE+1)*sizeof(unsigned); 
    KerNewPartListCreate <<<sgrid,SPHBSIZE,smem>>> (n,pini,nmax,fluxes,bufferpart,massf);
    hipMemcpy(&count,bufferpart+nmax,sizeof(unsigned),hipMemcpyDeviceToHost);
  }
  return(count);
}


//------------------------------------------------------------------------------
/// Creates new vres buffer particles to replace the particles moved to fluid domain.
//------------------------------------------------------------------------------
__global__ void KerCreateNewPart(unsigned newnp,unsigned pini,int *newpart
  ,unsigned np,unsigned idnext,double2 *posxy,double *posz
  ,unsigned *dcell,typecode *code,unsigned *idp,float4 *velrhop
  ,const float3 *normals,const float dp,double2 *posxyb,double *poszb,unsigned nzone)
{
  const unsigned cp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(cp<newnp){
    const int p=newpart[cp];
    const double dis=0.5*dp;
    const float3 normal=normals[p+pini];
    double2 rposxy=posxyb[p+pini];
    double rposz=poszb[p+pini];
    rposxy.x-=dis*normal.x;
    rposxy.y-=dis*normal.y;
    rposz   -=dis*normal.z;
    const unsigned p2=np+cp;
    code[p2]=CODE_ToFluidBuffer(CODE_TYPE_FLUID,byte(nzone));
    cusph::KerUpdatePos<false>(rposxy,rposz,0,0,0,false,p2,posxy,posz,dcell,code);
    idp[p2]=idnext+cp;
    velrhop[p2]=make_float4(0,0,0,1000);
  }
}

//==============================================================================
/// Creates new buffer particles in VRes simulations.
//==============================================================================
void CreateNewPart(unsigned newnp,unsigned pini,int *newpart
  ,unsigned np,unsigned idnext,double2 *posxy,double *posz
  ,unsigned *dcell,typecode *code,unsigned *idp,float4 *velrhop
  ,const float3 *normals,const float dp,double2 *posxyb,double *poszb
  ,unsigned nzone)
{
  if(newnp){
    dim3 sgrid=GetSimpleGridSize(newnp,SPHBSIZE);
    KerCreateNewPart<<<sgrid,SPHBSIZE>>> (newnp,pini,newpart,np,idnext,posxy,posz
    ,dcell,code,idp,velrhop,normals,dp,posxyb,poszb,nzone);

  }
}

//------------------------------------------------------------------------------
/// Move position and orient normal for accumulation points on the VRes interface.
//------------------------------------------------------------------------------
__global__ void KerMoveBufferZone(unsigned pini,unsigned ntot
  , double2 *posxy,double *posz,float3* normals
  ,float3* velflux,double dt,tmatrix4d mat)
{
  const unsigned cp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
  if(cp<ntot){
    const unsigned p1=cp+pini;
    const double2 rxy=posxy[p1];
    const double rz=posz[p1];
    double2 rxy_t=make_double2(0,0);
    double rz_t=0.0;
    rxy_t.x=rxy.x*mat.a11+rxy.y*mat.a12+rz*mat.a13+mat.a14;
    rxy_t.y=rxy.x*mat.a21+rxy.y*mat.a22+rz*mat.a23+mat.a24;
    rz_t   =rxy.x*mat.a31+rxy.y*mat.a32+rz*mat.a33+mat.a34;
    if(dt!=0)velflux[p1]=make_float3((rxy_t.x-rxy.x)/dt,(rxy_t.y-rxy.y)/dt,(rz_t-rz)/dt);
    posxy[p1]=rxy_t;
    posz[p1]=rz_t;
    float3 normalp1=normals[p1];
    normals[p1].x=normalp1.x*mat.a11+normalp1.y*mat.a12+normalp1.z*mat.a13;
    normals[p1].y=normalp1.x*mat.a21+normalp1.y*mat.a22+normalp1.z*mat.a23;
    normals[p1].z=normalp1.x*mat.a31+normalp1.y*mat.a32+normalp1.z*mat.a33;
  }
}

//========================================================================================
/// Move position and orient normal for accumulation points on the VRes interface.
//=========================================================================================
void MoveBufferZone(unsigned pini,unsigned ntot,
  double2 *posxy,double *posz,float3* normals,
  float3* velflux,double dt,tmatrix4d mat,int zone)
{
	dim3 sgrid=GetSimpleGridSize(ntot,SPHBSIZE);
	KerMoveBufferZone<<<sgrid,SPHBSIZE>>> (pini,ntot,posxy,posz,normals,velflux,dt,mat);
}


//--------------------------------------------------------------------------------------
/// Remove normal component to the interface of the shifting vector for buffer particles.
//---------------------------------------------------------------------------------------
__global__ void KerBufferShiftingGpu(unsigned n,unsigned pini,const double2 *posxy,const double *posz
		  ,float4 *shiftpos,typecode *code,const double3* boxlimitmin,const double3* boxlimitmax
      ,const bool* tracking,const tmatrix4f* mat,const bool* inner)
{
  unsigned p=blockIdx.x*blockDim.x + threadIdx.x;
  if(p<n){
	  const unsigned p1=p+pini;
	  typecode rcode=code[p1];
	  if(CODE_IsFluidBuffer(rcode)){

		  const byte izone0=byte(CODE_GetIzoneFluidBuffer(rcode));
		  const byte izone=(izone0&CODE_TYPE_FLUID_INOUT015MASK);
		  double2 rxy=posxy[p1];
      double rz=posz[p1];
      double3 boxmin=boxlimitmin[izone];
      double3 boxmax=boxlimitmax[izone];
      double3 origin  =make_double3((boxmax.x+boxmin.x)*0.5f,(boxmax.y+boxmin.y)*0.5f,(boxmax.z+boxmin.z)*0.5f);
      double3 boxsize =make_double3(boxmax.x-boxmin.x,boxmax.y-boxmin.y,boxmax.z-boxmin.z);
      
      if(tracking[izone]){
        double2 rxy_t=make_double2(0,0);
        double rz_t=0.0;
        MovePoint(rxy,rz,rxy_t,rz_t,mat[izone]);
        rxy=rxy_t; rz=rz_t;
      }
      
		  double disx =rxy.x  -origin.x;
      double disy =rxy.y  -origin.y;
		  double disz =rz     -origin.z;
      float3 shiftp1=make_float3(shiftpos[p1].x,shiftpos[p1].y,shiftpos[p1].z);

      if((fabs(disx)>boxsize.x/2.0 )){
        float3 normal=make_float3(mat[izone].a11,mat[izone].a21,mat[izone].a31);
        shiftpos[p1].x=shiftp1.x-normal.x*(shiftp1.x*normal.x+shiftp1.y*normal.y+shiftp1.z*normal.z);
        shiftpos[p1].y=shiftp1.y-normal.y*(shiftp1.x*normal.x+shiftp1.y*normal.y+shiftp1.z*normal.z);
        shiftpos[p1].z=shiftp1.z-normal.z*(shiftp1.x*normal.x+shiftp1.y*normal.y+shiftp1.z*normal.z);
      } 
      if((fabs(disy)>boxsize.y/2.0)){
        float3 normal=make_float3(mat[izone].a12,mat[izone].a22,mat[izone].a32);
        shiftpos[p1].x=shiftp1.x-normal.x*(shiftp1.x*normal.x+shiftp1.y*normal.y+shiftp1.z*normal.z);
        shiftpos[p1].y=shiftp1.y-normal.y*(shiftp1.x*normal.x+shiftp1.y*normal.y+shiftp1.z*normal.z);
        shiftpos[p1].z=shiftp1.z-normal.z*(shiftp1.x*normal.x+shiftp1.y*normal.y+shiftp1.z*normal.z);
      }
      if((fabs(disz)>boxsize.z/2.0)){
        float3 normal=make_float3(mat[izone].a13,mat[izone].a23,mat[izone].a33);
        shiftpos[p1].x=shiftp1.x-normal.x*(shiftp1.x*normal.x+shiftp1.y*normal.y+shiftp1.z*normal.z);
        shiftpos[p1].y=shiftp1.y-normal.y*(shiftp1.x*normal.x+shiftp1.y*normal.y+shiftp1.z*normal.z);
        shiftpos[p1].z=shiftp1.z-normal.z*(shiftp1.x*normal.x+shiftp1.y*normal.y+shiftp1.z*normal.z);
      }
    }
  }
}


//========================================================================================
/// Remove normal component to the interface of the shifting vector for buffer particles.
//=========================================================================================
void BufferShiftingGpu(unsigned np,unsigned npb,const double2 *posxy,const double *posz
  ,float4 *shiftpos,typecode *code,StrGeomVresGpu& vresgdata,hipStream_t stm)
{
  const unsigned npf=np-npb;
  if(npf){
    dim3 sgridf=GetSimpleGridSize(npf,SPHBSIZE);
    KerBufferShiftingGpu <<<sgridf,SPHBSIZE>>> (npf,npb,posxy,posz,shiftpos,code
      ,vresgdata.boxlimitmin,vresgdata.boxlimitmax,vresgdata.tracking
      ,vresgdata.matmov,vresgdata.inner);
  }
}
//------------------------------------------------------------------------------
/// Interaction of a particle with a set of particles. (Fluid/Float-Fluid/Float/Bound)
/// Realiza la interaccion de una particula con un conjunto de ellas. (Fluid/Float-Fluid/Float/Bound)
//------------------------------------------------------------------------------
__device__ void KerComputeNormalsBufferBox(unsigned p1,const double3 posp1
  ,float massp2,float& fs_treshold,float3& gradc,tmatrix3f& lcorr,unsigned& neigh
  ,float& pou,const double3 boxlimitmin,const double3 boxlimitmax,const bool inner
  ,const tmatrix4f mat,const bool tracking,const bool sim2d)
{    
      
  float3 minpos=make_float3(0,0,0);
  minpos.x=posp1.x-CTE.kernelsize;
  minpos.y=(sim2d? posp1.y: posp1.y-CTE.kernelsize);
  minpos.z=posp1.z-CTE.kernelsize;
  float3 maxpos=make_float3(0,0,0);
  maxpos.x=posp1.x+CTE.kernelsize;
  maxpos.y=(sim2d?posp1.y: posp1.y+CTE.kernelsize);
  maxpos.z=posp1.z+CTE.kernelsize;

  float dp=CTE.dp;
  for (float rx=minpos.x; rx<=maxpos.x; rx+=dp) for (float ry=minpos.y; ry<=maxpos.y; ry+=dp)
    for (float rz=minpos.z; rz<=maxpos.z; rz+=dp){
      const float drx=float(posp1.x-rx);
      const float dry=float(posp1.y-ry);
      const float drz=float(posp1.z-rz);
      const float rr2=drx*drx+dry*dry+drz*drz;
      float rx1=rx; float ry1=ry; float rz1=rz;
      if(tracking){          
        rx1=(rx-mat.a14)*mat.a11+(ry-mat.a24)*mat.a21+(rz-mat.a34)*mat.a31;
        ry1=(rx-mat.a14)*mat.a12+(ry-mat.a24)*mat.a22+(rz-mat.a34)*mat.a32;
        rz1=(rx-mat.a14)*mat.a13+(ry-mat.a24)*mat.a23+(rz-mat.a34)*mat.a33;
      }
      bool outside=(inner ? !KerBufferInZone(make_double2(rx1,ry1),rz1,boxlimitmin,boxlimitmax) : KerBufferInZone(make_double2(rx1,ry1),rz1,boxlimitmin,boxlimitmax));
      if(rr2<=CTE.kernelsize2 && rr2>=ALMOSTZERO && outside){
        //-Computes kernel.
        const float fac=cufsph::GetKernel_Fac<KERNEL_Wendland>(rr2);
        const float frx=fac*drx,fry=fac*dry,frz=fac*drz; //-Gradients.
            
        const float vol2=massp2/CTE.rhopzero;
        neigh++;

        const float dot3=drx*frx+dry*fry+drz*frz;
        gradc.x+=vol2*frx;
        gradc.y+=vol2*fry;
        gradc.z+=vol2*frz;

        fs_treshold-=vol2*dot3;
        lcorr.a11+=-drx*frx*vol2; lcorr.a12+=-drx*fry*vol2; lcorr.a13+=-drx*frz*vol2;
        lcorr.a21+=-dry*frx*vol2; lcorr.a22+=-dry*fry*vol2; lcorr.a23+=-dry*frz*vol2;
        lcorr.a31+=-drz*frx*vol2; lcorr.a32+=-drz*fry*vol2; lcorr.a33+=-drz*frz*vol2;

        const float wab=cufsph::GetKernel_Wab<KERNEL_Wendland>(rr2);
        pou+=wab*vol2;       
      }
    }
  }



__device__ void KerComputeNormalsBox(bool boundp2,unsigned p1
    ,const unsigned &pini,const unsigned &pfin,const float4 *poscell
    ,const float4* velrhop,const typecode* code,float massp2,const float4 &pscellp1
    ,float& fs_treshold,float3& gradc,tmatrix3f& lcorr,unsigned& neigh,float& pou,const float* ftomassp)
  {
    const float w0=cufsph::GetKernel_Wab<KERNEL_Wendland>(CTE.dp*CTE.dp);
    for(int p2=pini;p2<pfin;p2++){
    const float4 pscellp2=poscell[p2];
      float drx=pscellp1.x-pscellp2.x + CTE.poscellsize*(PSCEL_GetfX(pscellp1.w)-PSCEL_GetfX(pscellp2.w));
      float dry=pscellp1.y-pscellp2.y + CTE.poscellsize*(PSCEL_GetfY(pscellp1.w)-PSCEL_GetfY(pscellp2.w));
      float drz=pscellp1.z-pscellp2.z + CTE.poscellsize*(PSCEL_GetfZ(pscellp1.w)-PSCEL_GetfZ(pscellp2.w));
      const double rr2=drx*drx+dry*dry+drz*drz;
      if(rr2<=CTE.kernelsize2 && rr2>=ALMOSTZERO){
        //-Computes kernel.
        const float fac=cufsph::GetKernel_Fac<KERNEL_Wendland>(rr2);
        const float frx=fac*drx,fry=fac*dry,frz=fac*drz; //-Gradients.
        // float4 velrhop2=velrhop[p2];
        // if(symm)velrhop2.y=-velrhop2.y; //<vs_syymmetry>
        const float rhopp2= float(velrhop[p2].w);
        //-Velocity derivative (Momentum equation).

        bool ftp2;
        float ftmassp2;    //-Contains mass of floating body or massf if fluid. | Contiene masa de particula floating o massp2 si es bound o fluid.
          const typecode cod=code[p2];
          ftp2=CODE_IsFloating(cod);
          ftmassp2=(ftp2? ftomassp[CODE_GetTypeValue(cod)]: massp2);

        const float vol2=(ftp2 ? float(ftmassp2/rhopp2) : float(massp2/rhopp2));
        neigh++;

        const float dot3=drx*frx+dry*fry+drz*frz;
        gradc.x+=vol2*frx;
        gradc.y+=vol2*fry;
        gradc.z+=vol2*frz;

        fs_treshold-=vol2*dot3;
        lcorr.a11+=-drx*frx*vol2; lcorr.a12+=-drx*fry*vol2; lcorr.a13+=-drx*frz*vol2;
        lcorr.a21+=-dry*frx*vol2; lcorr.a22+=-dry*fry*vol2; lcorr.a23+=-dry*frz*vol2;
        lcorr.a31+=-drz*frx*vol2; lcorr.a32+=-drz*fry*vol2; lcorr.a33+=-drz*frz*vol2;

        const float wab=cufsph::GetKernel_Wab<KERNEL_Wendland>(rr2);
        pou+=wab*vol2;       

      }
    }
  }



//==============================================================================
/// Perform interaction between particles: Fluid/Float-Fluid/Float or Fluid/Float-Bound
/// Realiza interaccion entre particulas: Fluid/Float-Fluid/Float or Fluid/Float-Bound
//==============================================================================
    __global__ void KerComputeNormals(unsigned n,unsigned pinit
    ,int scelldiv,int4 nc,int3 cellzero,const int2 *begincell,unsigned cellfluid,const unsigned *dcell
    ,const float4 *poscell,const float4 *velrhop,const typecode *code,unsigned* fstype,float3* fsnormal
    ,bool simulate2d,float4* shiftposfs,const float* ftomassp,const unsigned* listp,const double2* posxy
    ,const double* posz,const double3* boxlimitmin,const double3* boxlimitmax,const bool* inner
    ,const tmatrix4f* mat,const bool* tracking)
  {
    const unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
    if(p<n){
      const unsigned p1=listp[p];   
      //-Obtains basic data of particle p1.
      const float4 pscellp1=poscell[p1];
      const float4 velrhop1=velrhop[p1];
      
      float fs_treshold=0.f;                            //-Divergence of the position.
      float3 gradc=make_float3(0,0,0);                  //-Gradient of the concentration
      float pou=0;                                      //-Partition of unity.
      unsigned neigh=0;                                 //-Number of neighbours.
      
      tmatrix3f lcorr; cumath::Tmatrix3fReset(lcorr);             //-Correction matrix.
      tmatrix3f lcorr_inv; cumath::Tmatrix3fReset(lcorr_inv);     //-Inverse of the correction matrix.

      //-Calculate approx. number of neighbours when uniform distribution (in the future on the constant memory?)
      float Nzero=0.f;
      if(simulate2d){
        Nzero=(3.141592)*CTE.kernelsize2/(CTE.dp*CTE.dp);
      } else{
        Nzero=(4.f/3.f)*(3.141592)*CTE.kernelsize2*CTE.kernelsize/(CTE.dp*CTE.dp*CTE.dp);
      }

      //-Copy the value of shift to gradc. For single resolution is zero, but in Vres take in account virtual stencil.
      gradc=make_float3(shiftposfs[p1].x,shiftposfs[p1].y,shiftposfs[p1].z); 
      

    
      //-Obtains neighborhood search limits.
      int ini1,fin1,ini2,fin2,ini3,fin3;
      cunsearch::InitCte(dcell[p1],scelldiv,nc,cellzero,ini1,fin1,ini2,fin2,ini3,fin3);

      //-Interaction with fluids.
      ini3+=cellfluid; fin3+=cellfluid;
      for(int c3=ini3;c3<fin3;c3+=nc.w)for(int c2=ini2;c2<fin2;c2+=nc.x){
        unsigned pini,pfin=0;  cunsearch::ParticleRange(c2,c3,ini1,fin1,begincell,pini,pfin);
        if(pfin){
          KerComputeNormalsBox (false,p1,pini,pfin,poscell,velrhop,code,CTE.massf,pscellp1,fs_treshold,gradc,lcorr,neigh,pou,ftomassp);
        }
      }

      // -Interaction with boundaries.
      ini3-=cellfluid; fin3-=cellfluid;
      for(int c3=ini3;c3<fin3;c3+=nc.w)for(int c2=ini2;c2<fin2;c2+=nc.x){
        unsigned pini,pfin=0;  cunsearch::ParticleRange(c2,c3,ini1,fin1,begincell,pini,pfin);
        if(pfin){
          KerComputeNormalsBox(false,p1,pini,pfin,poscell,velrhop,code,CTE.massf,pscellp1,fs_treshold,gradc,lcorr,neigh,pou,ftomassp);

        }
      }

      // -Interaction with virtual stencil.
      if(CODE_IsFluidBuffer(code[p1])){
        const byte izone0=byte(CODE_GetIzoneFluidBuffer(code[p1]));
        const byte izone=(izone0&CODE_TYPE_FLUID_INOUT015MASK);
        const double3   boxlimmin =boxlimitmin[izone];
        const double3   boxlimmax =boxlimitmax[izone];
        const bool      inn       =inner      [izone];
        const tmatrix4f rmat      =mat        [izone];
        const bool      track     =tracking   [izone]; 
        const double3   posp1     =make_double3(posxy[p1].x,posxy[p1].y,posz[p1]); 
        KerComputeNormalsBufferBox (p1,posp1,CTE.massf,fs_treshold,gradc,lcorr,neigh,pou,boxlimmin,boxlimmax,inn,rmat,track,simulate2d);
      }

      unsigned fstypep1=0;
      if(simulate2d){
        if(fs_treshold<1.7) fstypep1=2;
        if(fs_treshold<1.1 && Nzero/float(neigh)<0.4f) fstypep1=3;
      } else {
        if(fs_treshold<2.75) fstypep1=2;
        if(fs_treshold<1.8 && Nzero/float(neigh)<0.4f) fstypep1=3;
      }
      fstype[p1]=fstypep1;

      //-Add the contribution of the particle itself
      pou+=cufsph::GetKernel_Wab<KERNEL_Wendland>(0.f)*CTE.massf/velrhop1.w;

      //-Calculation of the inverse of the correction matrix (Don't think there is a better way, create function for Determinant2x2 for clarity?).
      if(simulate2d){
        tmatrix2f lcorr2d;
        tmatrix2f lcorr2d_inv;
        lcorr2d.a11=lcorr.a11; lcorr2d.a12=lcorr.a13;
        lcorr2d.a21=lcorr.a31; lcorr2d.a22=lcorr.a33;
        float lcorr_det=(lcorr2d.a11*lcorr2d.a22-lcorr2d.a12*lcorr2d.a21);
        lcorr2d_inv.a11=lcorr2d.a22/lcorr_det; lcorr2d_inv.a12=-lcorr2d.a12/lcorr_det; lcorr2d_inv.a22=lcorr2d.a11/lcorr_det; lcorr2d_inv.a21=-lcorr2d.a21/lcorr_det;
        lcorr_inv.a11=lcorr2d_inv.a11;  lcorr_inv.a13=lcorr2d_inv.a12;
        lcorr_inv.a31=lcorr2d_inv.a21;  lcorr_inv.a33=lcorr2d_inv.a22;
      } else {
        const float determ = cumath::Determinant3x3(lcorr);
        lcorr_inv = cumath::InverseMatrix3x3(lcorr, determ);
      }

      //-Correction of the gradient of concentration and definition of the normal.
      float3 gradc1=make_float3(0,0,0);    
      gradc1.x=gradc.x*lcorr_inv.a11+gradc.y*lcorr_inv.a12+gradc.z*lcorr_inv.a13;
      gradc1.y=gradc.x*lcorr_inv.a21+gradc.y*lcorr_inv.a22+gradc.z*lcorr_inv.a23;
      gradc1.z=gradc.x*lcorr_inv.a31+gradc.y*lcorr_inv.a32+gradc.z*lcorr_inv.a33;    
      float gradc_norm=sqrt(gradc1.x*gradc1.x+gradc1.y*gradc1.y+gradc1.z*gradc1.z);
      fsnormal[p1].x=-gradc1.x/gradc_norm;
      fsnormal[p1].y=-gradc1.y/gradc_norm;
      fsnormal[p1].z=-gradc1.z/gradc_norm;
    }
  }

 
//------------------------------------------------------------------------------
/// Obtain the list of particle that are probably on the free-surface.
//------------------------------------------------------------------------------
  __global__ void KerCountFreeSurface(unsigned n,unsigned pini
    ,unsigned* fs,unsigned* listp)
  {
    extern __shared__ unsigned slist[];
    //float* splanes=(float*)(slist+(n+1));
    if(!threadIdx.x)slist[0]=0;
    __syncthreads();
    const unsigned pp=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
    if(pp<n){
      const unsigned p=pp+pini;
      if(fs[p]>1) slist[atomicAdd(slist,1)+1]=p;
    }
    __syncthreads();
    const unsigned ns=slist[0];
    __syncthreads();
    if(!threadIdx.x && ns)slist[0]=atomicAdd((listp+n),ns);
    __syncthreads();
    if(threadIdx.x<ns){
      const unsigned cp=slist[0]+threadIdx.x;
      listp[cp]=slist[threadIdx.x+1];
    }
  }

//==============================================================================
/// Compute free-surface particles and their normals.
//==============================================================================
  void ComputeFSNormals(TpKernel tkernel,bool simulate2d,unsigned bsfluid,unsigned fluidini,unsigned fluidnum
    ,StDivDataGpu& dvd,const unsigned* dcell,const double2* posxy,const double* posz
    ,const float4* poscell,const float4* velrho,const typecode* code,const float* ftomassp,float4* shiftposfs
    ,unsigned* fstype,float3* fsnormal,unsigned* listp,StrGeomVresGpu& vresgdata,hipStream_t stm)
  {
    unsigned count=0;

    //-Obtain the list of particle that are probably on the free-surface
    if(fluidnum){
      hipMemset(listp+fluidnum,0,sizeof(unsigned));
      dim3 sgridf=GetSimpleGridSize(fluidnum,bsfluid);
      const unsigned smem=(bsfluid+1)*sizeof(unsigned); 
      KerCountFreeSurface <<<sgridf,bsfluid,smem,stm>>> (fluidnum,fluidini,fstype,listp);
    }
    hipMemcpy(&count,listp+fluidnum,sizeof(unsigned),hipMemcpyDeviceToHost);
    hipDeviceSynchronize();

    if(count){
      dim3 sgridf=GetSimpleGridSize(count,bsfluid);
      KerComputeNormals <<<sgridf,bsfluid,0,stm>>> 
        (count,fluidini,dvd.scelldiv,dvd.nc,dvd.cellzero,dvd.beginendcell,dvd.cellfluid,dcell
            ,poscell,velrho,code,fstype,fsnormal,simulate2d,shiftposfs
            ,ftomassp,listp
            ,posxy,posz,vresgdata.boxdommin,vresgdata.boxdommax,vresgdata.inner,vresgdata.matmov,vresgdata.tracking);
    }
    
    hipDeviceSynchronize();
    
  }


  __device__ void KerScanUmbrellaRegionBufferBox(unsigned p1
    ,const double3 posp1,bool& fs_flag,const float3* fsnormal
    ,const double3 boxlimitmin,const double3 boxlimitmax,const bool inner
    ,const tmatrix4f mat,const bool tracking,const bool sim2d)
  {    
    
    float3 minpos=make_float3(0,0,0);
    minpos.x=posp1.x-CTE.kernelsize;
    minpos.y=(sim2d? posp1.y: posp1.y-CTE.kernelsize);
    minpos.z=posp1.z-CTE.kernelsize;
    float3 maxpos=make_float3(0,0,0);
    maxpos.x=posp1.x+CTE.kernelsize;
    maxpos.y=(sim2d?posp1.y: posp1.y+CTE.kernelsize);
    maxpos.z=posp1.z+CTE.kernelsize;

    float dp=CTE.dp;
    for (float rx=minpos.x; rx<=maxpos.x; rx+=dp) for (float ry=minpos.y; ry<=maxpos.y; ry+=dp)
      for (float rz=minpos.z; rz<=maxpos.z; rz+=dp){
      const float drx=float(posp1.x-rx);
      const float dry=float(posp1.y-ry);
      const float drz=float(posp1.z-rz);
      const float rr2=drx*drx+dry*dry+drz*drz;
      float rx1=rx; float ry1=ry; float rz1=rz;
      if(tracking){          
        rx1=(rx-mat.a14)*mat.a11+(ry-mat.a24)*mat.a21+(rz-mat.a34)*mat.a31;
        ry1=(rx-mat.a14)*mat.a12+(ry-mat.a24)*mat.a22+(rz-mat.a34)*mat.a32;
        rz1=(rx-mat.a14)*mat.a13+(ry-mat.a24)*mat.a23+(rz-mat.a34)*mat.a33;
      }
      bool outside=(inner ? !KerBufferInZone(make_double2(rx1,ry1),rz1,boxlimitmin,boxlimitmax) : KerBufferInZone(make_double2(rx1,ry1),rz1,boxlimitmin,boxlimitmax));
      if(rr2<=CTE.kernelsize2 && rr2>=ALMOSTZERO && outside){

      const float3 posq=make_float3(fsnormal[p1].x*CTE.kernelh/* +posxy[p1].x */,fsnormal[p1].y*CTE.kernelh/* +posxy[p1].y */,fsnormal[p1].z*CTE.kernelh/* +posz[p1] */);

      if (rr2>2.f*CTE.kernelh*CTE.kernelh){
        // const float3 posp2=make_float3(posxy[p2].x,posxy[p2].y,posz[p2]);
        const float drxq=-drx-posq.x;
        const float dryq=-dry-posq.y;
        const float drzq=-drz-posq.z;
        const float rrq=sqrt(drxq*drxq+dryq*dryq+drzq*drzq);
        if(rrq<CTE.kernelh) fs_flag=true;
      } else {
        if(sim2d){
        const float drxq=-drx-posq.x;
        const float drzq=-drz-posq.z;
        const float3 normalq=make_float3(drxq*fsnormal[p1].x,0,drzq*fsnormal[p1].z);
        const float3 tangq=make_float3(-drxq*fsnormal[p1].z,0,drzq*fsnormal[p1].x);
        const float normalqnorm=sqrt(normalq.x*normalq.x+normalq.z*normalq.z);
        const float tangqnorm=sqrt(tangq.x*tangq.x+tangq.z*tangq.z);
        if (normalqnorm+tangqnorm<CTE.kernelh) fs_flag=true;
        } else{
        float rrr=1.f/sqrt(rr2);
        const float arccosin=acos((-drx*fsnormal[p1].x*rrr-dry*fsnormal[p1].y*rrr-drz*fsnormal[p1].z*rrr));
        if (arccosin < 0.785398) fs_flag=true;
        }
      }

    }
  }
  if(fs_flag) return;
  }

//------------------------------------------------------------------------------
/// Interaction of a particle with a set of particles. (Fluid/Float-Fluid/Float/Bound)
/// Realiza la interaccion de una particula con un conjunto de ellas. (Fluid/Float-Fluid/Float/Bound)
//------------------------------------------------------------------------------
 __device__ void KerScanUmbrellaRegionBox(bool boundp2,unsigned p1
    ,const unsigned &pini,const unsigned &pfin,const float4 *poscell,const float4 &pscellp1
    ,bool& fs_flag,const float3* fsnormal,bool simulate2d)
  {
    for(int p2=pini;p2<pfin;p2++){
    const float4 pscellp2=poscell[p2];
      float drx=pscellp1.x-pscellp2.x + CTE.poscellsize*(PSCEL_GetfX(pscellp1.w)-PSCEL_GetfX(pscellp2.w));
      float dry=pscellp1.y-pscellp2.y + CTE.poscellsize*(PSCEL_GetfY(pscellp1.w)-PSCEL_GetfY(pscellp2.w));
      float drz=pscellp1.z-pscellp2.z + CTE.poscellsize*(PSCEL_GetfZ(pscellp1.w)-PSCEL_GetfZ(pscellp2.w));
      const double rr2=drx*drx+dry*dry+drz*drz;
      if(rr2<=CTE.kernelsize2 && rr2>=ALMOSTZERO){

        const float3 posq=make_float3(fsnormal[p1].x*CTE.kernelh,fsnormal[p1].y*CTE.kernelh,fsnormal[p1].z*CTE.kernelh);

        if (rr2>2.f*CTE.kernelh*CTE.kernelh){
          const float drxq=-drx-posq.x;
          const float dryq=-dry-posq.y;
          const float drzq=-drz-posq.z;
          const float rrq=sqrt(drxq*drxq+dryq*dryq+drzq*drzq);
          if(rrq<CTE.kernelh) fs_flag=true;
        } else {
          if(simulate2d){
          const float drxq=-drx-posq.x;
          const float drzq=-drz-posq.z;
          const float3 normalq=make_float3(drxq*fsnormal[p1].x,0,drzq*fsnormal[p1].z);
          const float3 tangq=make_float3(-drxq*fsnormal[p1].z,0,drzq*fsnormal[p1].x);
          const float normalqnorm=sqrt(normalq.x*normalq.x+normalq.z*normalq.z);
          const float tangqnorm=sqrt(tangq.x*tangq.x+tangq.z*tangq.z);
          if (normalqnorm+tangqnorm<CTE.kernelh) fs_flag=true;
          } else{
            float rrr=1.f/sqrt(rr2);
          const float arccosin=acos((-drx*fsnormal[p1].x*rrr-dry*fsnormal[p1].y*rrr-drz*fsnormal[p1].z*rrr));
          if (arccosin < 0.785398) fs_flag=true;
          }
        }

      }
    }
    if(fs_flag) return;
  }
//==============================================================================
/// Interaction of Fluid-Fluid/Bound & Bound-Fluid.
/// Interaccion Fluid-Fluid/Bound & Bound-Fluid.
//==============================================================================
  __global__ void KerScanUmbrellaRegion(unsigned n,unsigned pinit
    ,int scelldiv,int4 nc,int3 cellzero,const int2 *begincell,unsigned cellfluid,const unsigned *dcell
    ,const float4 *poscell,const typecode* code,unsigned* fstype,float3* fsnormal,bool simulate2d,const unsigned* listp
    ,const double2* posxy,const double* posz,const double3* boxlimitmin,const double3* boxlimitmax,const bool* inner,const tmatrix4f* mat,const bool* tracking)
  {
    const unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
    if(p<n){
      const unsigned p1=listp[p];   
      //-Obtains basic data of particle p1.
      const float4 pscellp1=poscell[p1];
      

      bool fs_flag=false;
      int ini1,fin1,ini2,fin2,ini3,fin3;

      cunsearch::InitCte(dcell[p1],scelldiv,nc,cellzero,ini1,fin1,ini2,fin2,ini3,fin3);

      //-Interaction with fluids.
      ini3+=cellfluid; fin3+=cellfluid;
      for(int c3=ini3;c3<fin3;c3+=nc.w)for(int c2=ini2;c2<fin2;c2+=nc.x){
        unsigned pini,pfin=0;  cunsearch::ParticleRange(c2,c3,ini1,fin1,begincell,pini,pfin);
        if(pfin){
          KerScanUmbrellaRegionBox (false,p1,pini,pfin,poscell,pscellp1,fs_flag,fsnormal,simulate2d);
        }
      }

      ini3-=cellfluid; fin3-=cellfluid;
      for(int c3=ini3;c3<fin3;c3+=nc.w)for(int c2=ini2;c2<fin2;c2+=nc.x){
        unsigned pini,pfin=0;  cunsearch::ParticleRange(c2,c3,ini1,fin1,begincell,pini,pfin);
        if(pfin){
          KerScanUmbrellaRegionBox(true,p1,pini,pfin,poscell,pscellp1,fs_flag,fsnormal,simulate2d);
        }
      }

      // -Interaction with virtual stencil.
      if(CODE_IsFluidBuffer(code[p1])){
        const byte izone0=byte(CODE_GetIzoneFluidBuffer(code[p1]));
        const byte izone=(izone0&CODE_TYPE_FLUID_INOUT015MASK);
        const double3   boxlimmin =boxlimitmin[izone];
        const double3   boxlimmax =boxlimitmax[izone];
        const bool      inn       =inner      [izone];
        const tmatrix4f rmat      =mat        [izone];
        const bool      track     =tracking   [izone]; 
        const double3   posp1     =make_double3(posxy[p1].x,posxy[p1].y,posz[p1]); 
        KerScanUmbrellaRegionBufferBox (p1,posp1,fs_flag,fsnormal,boxlimmin,boxlimmax,inn,rmat,track,simulate2d);
      }  
      //-If particle was present in umbrella region, change the code of the particle.
      if(fs_flag && fstype[p1]==2) fstype[p1]=0;
      //-Periodic particle are internal by default.
      if(CODE_IsPeriodic(code[p1])) fstype[p1]=0;  
    }
  }

//==============================================================================
/// Scan Umbrella region to identify free-surface particle.
//==============================================================================
  void ComputeUmbrellaRegion(TpKernel tkernel,bool simulate2d,unsigned bsfluid,unsigned fluidini,unsigned fluidnum
    ,StDivDataGpu& dvd,const unsigned* dcell,const double2* posxy,const double* posz
    ,const float4* poscell,const float4* velrho,const typecode* code,const float* ftomassp,float4* shiftposfs
    ,unsigned* fstype,float3* fsnormal,unsigned* listp,StrGeomVresGpu& vresgdata,hipStream_t stm)
  {
    unsigned count=0;
    //-Obtain the list of particle that are probably on the free-surface (in ComputeUmbrellaRegion maybe is unnecessary).
    if(fluidnum){
      hipMemset(listp+fluidnum,0,sizeof(unsigned));
      dim3 sgridf=GetSimpleGridSize(fluidnum,bsfluid);
      const unsigned smem=(bsfluid+1)*sizeof(unsigned); 
      KerCountFreeSurface <<<sgridf,bsfluid,smem,stm>>> (fluidnum,fluidini,fstype,listp);
    }
    hipMemcpy(&count,listp+fluidnum,sizeof(unsigned),hipMemcpyDeviceToHost);
    hipDeviceSynchronize();

    if(count){
      dim3 sgridf=GetSimpleGridSize(count,bsfluid);
      KerScanUmbrellaRegion<<<sgridf,bsfluid,0,stm>>> 
        (count,fluidini,dvd.scelldiv,dvd.nc,dvd.cellzero,dvd.beginendcell
        ,dvd.cellfluid,dcell,poscell,code,fstype,fsnormal,simulate2d,listp
        ,posxy,posz,vresgdata.boxdommin,vresgdata.boxdommax,vresgdata.inner
        ,vresgdata.matmov,vresgdata.tracking);
    }    
    hipDeviceSynchronize();    
  }

 __device__ void KerComputeShiftingVelBufferBox(unsigned p1,const double3 posp1
    ,float massp2,float4& shiftvel,const double3 boxlimitmin,const double3 boxlimitmax
    ,const bool inner,const tmatrix4f mat,const bool tracking,const bool sim2d)
  {    
      
    float3 minpos=make_float3(0,0,0);
    minpos.x=posp1.x-CTE.kernelsize;
    minpos.y=(sim2d? posp1.y: posp1.y-CTE.kernelsize);
    minpos.z=posp1.z-CTE.kernelsize;
    float3 maxpos=make_float3(0,0,0);
    maxpos.x=posp1.x+CTE.kernelsize;
    maxpos.y=(sim2d?posp1.y: posp1.y+CTE.kernelsize);
    maxpos.z=posp1.z+CTE.kernelsize;

      float dp=CTE.dp;
      for (float rx=minpos.x; rx<=maxpos.x; rx+=dp) for (float ry=minpos.y; ry<=maxpos.y; ry+=dp)
        for (float rz=minpos.z; rz<=maxpos.z; rz+=dp){
          const float drx=float(posp1.x-rx);
          const float dry=float(posp1.y-ry);
          const float drz=float(posp1.z-rz);
          const float rr2=drx*drx+dry*dry+drz*drz;
          float rx1=rx; float ry1=ry; float rz1=rz;
          if(tracking){          
            rx1=(rx-mat.a14)*mat.a11+(ry-mat.a24)*mat.a21+(rz-mat.a34)*mat.a31;
            ry1=(rx-mat.a14)*mat.a12+(ry-mat.a24)*mat.a22+(rz-mat.a34)*mat.a32;
            rz1=(rx-mat.a14)*mat.a13+(ry-mat.a24)*mat.a23+(rz-mat.a34)*mat.a33;
          }
          bool outside=(inner ? !KerBufferInZone(make_double2(rx1,ry1),rz1,boxlimitmin,boxlimitmax) : KerBufferInZone(make_double2(rx1,ry1),rz1,boxlimitmin,boxlimitmax));
          if(rr2<=CTE.kernelsize2 && rr2>=ALMOSTZERO && outside){
            //-Computes kernel.
            const float fac=cufsph::GetKernel_Fac<KERNEL_Wendland>(rr2);
            const float frx=fac*drx,fry=fac*dry,frz=fac*drz; //-Gradients.
            
            const float vol2=massp2/CTE.rhopzero;

            shiftvel.x+=vol2*frx;    
            shiftvel.y+=vol2*fry;
            shiftvel.z+=vol2*frz;          
            const float wab=cufsph::GetKernel_Wab<KERNEL_Wendland>(rr2);
            shiftvel.w+=wab*vol2;      
      }
    }
  }

//==============================================================================
/// Interaction of a particle with a set of particles. (Fluid/Float-Fluid/Float/Bound)
//==============================================================================
  template<TpKernel tker,bool simulate2d,bool shiftadv>
  __device__ void KerPreLoopInteractionBox(bool boundp2,unsigned p1
    ,const unsigned &pini,const unsigned &pfin,const float4 *poscell,const float4 *velrhop
    ,const typecode *code,float massp2,const float4 &pscellp1,const float4 &velrhop1,const float* ftomassp
    ,float4 &shiftposf1,unsigned* fs,float3* fsnormal,bool& nearfs,float &mindist,float& maxarccos
    ,bool& bound_inter,float3& fsnormalp1,float& pou)
  {
    for(int p2=pini;p2<pfin;p2++){
      const float4 pscellp2=poscell[p2];
      float drx=pscellp1.x-pscellp2.x + CTE.poscellsize*(PSCEL_GetfX(pscellp1.w)-PSCEL_GetfX(pscellp2.w));
      float dry=pscellp1.y-pscellp2.y + CTE.poscellsize*(PSCEL_GetfY(pscellp1.w)-PSCEL_GetfY(pscellp2.w));
      float drz=pscellp1.z-pscellp2.z + CTE.poscellsize*(PSCEL_GetfZ(pscellp1.w)-PSCEL_GetfZ(pscellp2.w));
      const double rr2=drx*drx+dry*dry+drz*drz;
      if(rr2<=CTE.kernelsize2 && rr2>=ALMOSTZERO){

        const float fac=cufsph::GetKernel_Fac<KERNEL_Wendland>(rr2);
        const float frx=fac*drx,fry=fac*dry,frz=fac*drz; //-Gradients.
        float4 velrhop2=velrhop[p2];

        bool ftp2;
        float ftmassp2;    //-Contains mass of floating body or massf if fluid. | Contiene masa de particula floating o massp2 si es bound o fluid.
        const typecode cod=code[p2];
        ftp2=CODE_IsFloating(cod);
        ftmassp2=(ftp2? ftomassp[CODE_GetTypeValue(cod)]: massp2);
        
        if(shiftadv){
          const float massrho=(boundp2 ? CTE.massb/velrhop2.w : (ftmassp2)/velrhop2.w);

          //-Compute gradient of concentration and partition of unity.        
          shiftposf1.x+=massrho*frx;    
          shiftposf1.y+=massrho*fry;
          shiftposf1.z+=massrho*frz;          
          const float wab=cufsph::GetKernel_Wab<KERNEL_Wendland>(rr2);
          shiftposf1.w+=wab*massrho;


          //-Check if the particle is too close to solid or floating object
          if((boundp2 || ftp2)) bound_inter=true;

          //-Check if it close to the free-surface, calculate distance from free-surface and smoothing of free-surface normals.
          if(fs[p2]>1 && fs[p2]<3 && !boundp2 ) {
            nearfs=true;
            mindist=min(sqrt(rr2),mindist);
            pou+=wab*massrho;

            fsnormalp1.x+=fsnormal[p2].x*wab*massrho;
            fsnormalp1.y+=fsnormal[p2].y*wab*massrho;
            fsnormalp1.z+=fsnormal[p2].z*wab*massrho;
                        
          }

          //-Check maximum curvature.
          if(fs[p1]>1 && fs[p2]>1){
          const float norm1=sqrt(fsnormal[p1].x*fsnormal[p1].x+fsnormal[p1].y*fsnormal[p1].y+fsnormal[p1].z*fsnormal[p1].z);
          const float norm2=sqrt(fsnormal[p2].x*fsnormal[p2].x+fsnormal[p2].y*fsnormal[p2].y+fsnormal[p2].z*fsnormal[p2].z);
          maxarccos=max(maxarccos,(acos((fsnormal[p1].x*fsnormal[p2].x+fsnormal[p2].y*fsnormal[p1].y+fsnormal[p2].z*fsnormal[p1].z))));
          }
        }
      }
    }
  }

//==============================================================================
/// Interaction of Fluid-Fluid/Bound & Bound-Fluid for models before
/// InteractionForces
//==============================================================================
  template<TpKernel tker,bool simulate2d,bool shiftadv>
  __global__ void KerPreLoopInteraction(unsigned n
    ,unsigned pinit,int scelldiv,int4 nc,int3 cellzero,const int2 *begincell,unsigned cellfluid
    ,const unsigned *dcell,const float4 *poscell,const float4 *velrhop,const typecode *code
    ,const float* ftomassp,float4* shiftvel,unsigned* fstype,float3* fsnormal,float* fsmindist
    ,const double2* posxy,const double* posz,const double3* boxlimitmin,const double3* boxlimitmax
    ,const bool* inner,const tmatrix4f* mat,const bool* tracking)
  {
    const unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of particle.
    if(p<n){
      const unsigned p1=p+pinit;      //-Number of particle.
      //-Obtains basic data of particle p1.
      const float4 pscellp1=poscell[p1];
      const float4 velrhop1=velrhop[p1];
      const float pressp1=cufsph::ComputePressCte(velrhop1.w);
      bool    nearfs=false;                     //-Bool for detecting near free-surface particles. <shiftImproved>
      float4  shiftposp1=make_float4(0,0,0,0);

      
      float mindist=CTE.kernelh;                //-Set Min Distance from free-surface to kernel radius. <shiftImproved>
      float maxarccos=0.0;                      //-Variable for identify high-curvature free-surface particle <shiftImproved>
      bool bound_inter=false;                   //-Variable for identify free-surface that interact with boundary <shiftImproved>
      float3 fsnormalp1=make_float3(0,0,0);     //-Normals for near free-surface particles <shiftImproved>
      unsigned fsp1=fstype[p1];                 //-Free-surface identification code: 0-internal, 1-close to free-surface, 2 free-surface, 3-isolated.
      float   pou=false;                        //-Partition of unity for normal correction.                      <ShiftingAdvanced>
    
      //-Obtains neighborhood search limits.
      int ini1,fin1,ini2,fin2,ini3,fin3;
      cunsearch::InitCte(dcell[p1],scelldiv,nc,cellzero,ini1,fin1,ini2,fin2,ini3,fin3);

      //-Interaction with fluids.
      ini3+=cellfluid; fin3+=cellfluid;
      for(int c3=ini3;c3<fin3;c3+=nc.w)for(int c2=ini2;c2<fin2;c2+=nc.x){
        unsigned pini,pfin=0;  cunsearch::ParticleRange(c2,c3,ini1,fin1,begincell,pini,pfin);
        if(pfin){
          KerPreLoopInteractionBox<tker,simulate2d,shiftadv> (false,p1,pini,pfin,poscell,velrhop
            ,code,CTE.massf,pscellp1,velrhop1,ftomassp,shiftposp1
            ,fstype,fsnormal,nearfs,mindist,maxarccos,bound_inter,fsnormalp1,pou);
        }
      }


      //-Interaction with bound.
      ini3-=cellfluid; fin3-=cellfluid;
      for(int c3=ini3;c3<fin3;c3+=nc.w)for(int c2=ini2;c2<fin2;c2+=nc.x){
        unsigned pini,pfin=0;  cunsearch::ParticleRange(c2,c3,ini1,fin1,begincell,pini,pfin);
        if(pfin){
        KerPreLoopInteractionBox<tker,simulate2d,shiftadv> (true,p1,pini,pfin,poscell,velrhop
            ,code,CTE.massf,pscellp1,velrhop1,ftomassp,shiftposp1
            ,fstype,fsnormal,nearfs,mindist,maxarccos,bound_inter,fsnormalp1,pou);
      }
      }



      if(shiftadv){

        // -Interaction with virtual stencil.
        if(CODE_IsFluidBuffer(code[p1])){
          const byte izone0=byte(CODE_GetIzoneFluidBuffer(code[p1]));
          const byte izone=(izone0&CODE_TYPE_FLUID_INOUT015MASK);
          const double3   boxlimmin =boxlimitmin[izone];
          const double3   boxlimmax =boxlimitmax[izone];
          const bool      inn       =inner      [izone];
          const tmatrix4f rmat      =mat        [izone];
          const bool      track     =tracking   [izone]; 
          const double3   posp1     =make_double3(posxy[p1].x,posxy[p1].y,posz[p1]); 
          KerComputeShiftingVelBufferBox(p1,posp1,CTE.massf,shiftposp1,boxlimmin,boxlimmax,inn,rmat,track,simulate2d);
        }

        shiftposp1.w+=cufsph::GetKernel_Wab<KERNEL_Wendland>(0.0)*CTE.massf/velrhop1.w;


      fsmindist[p1]=mindist;
      //-Assign correct code to near free-surface particle and correct their normals by Shepard's Correction.
      if(fsp1==0 && nearfs){
      if(pou>1e-6){
        fsnormalp1=make_float3(fsnormalp1.x,fsnormalp1.y,fsnormalp1.z);
        float norm=sqrt(fsnormalp1.x*fsnormalp1.x+fsnormalp1.y*fsnormalp1.y+fsnormalp1.z*fsnormalp1.z);
        fsnormal[p1]=make_float3(fsnormalp1.x/norm,fsnormalp1.y/norm,fsnormalp1.z/norm);
      }      
      fstype[p1]=1;
      if(bound_inter) fstype[p1]=3;
      }
      
      //-Check if free-surface particle interact with bound or has high-curvature.
      if(fsp1==2 && (bound_inter||maxarccos>0.52)) shiftposp1=make_float4(0,0,0,shiftposp1.w);
      if(fstype[p1]>0 && CODE_IsFluidBuffer(code[p1])) shiftposp1=make_float4(0,0,0,shiftposp1.w);

      //-Compute shifting when <shiftImproved> true
      shiftvel[p1]=shiftposp1;
      }
      
    }
  }
 
//==============================================================================
/// Interaction of Fluid-Fluid/Bound & Bound-Fluid for models before
/// InteractionForces
//==============================================================================  
  
  template<TpKernel tker,bool simulate2d,bool shiftadv> void PreLoopInteractionT3(unsigned bsfluid
    ,unsigned fluidnum,unsigned fluidini,StDivDataGpu& dvd,const double2* posxy,const double* posz
    ,const unsigned* dcell,const float4* poscell,const float4* velrho,const typecode* code,const float* ftomassp
    ,float4* shiftvel,unsigned* fstype,float3* fsnormal,float* fsmindist,StrGeomVresGpu& vresgdata,hipStream_t stm)
{

  if(fluidnum){
    dim3 sgridf=GetSimpleGridSize(fluidnum,bsfluid);
    KerPreLoopInteraction <tker,simulate2d,shiftadv> <<<sgridf,bsfluid,0,stm>>> 
      (fluidnum,fluidini,dvd.scelldiv,dvd.nc,dvd.cellzero,dvd.beginendcell,dvd.cellfluid
      ,dcell,poscell,velrho,code,ftomassp,shiftvel,fstype,fsnormal,fsmindist
      ,posxy,posz,vresgdata.boxdommin,vresgdata.boxdommax,vresgdata.inner,vresgdata.matmov,vresgdata.tracking); 

  }
}
//==============================================================================
  template<TpKernel tker,bool simulate2d> void PreLoopInteractionT2(bool shiftadv
    ,unsigned bsfluid,unsigned fluidnum,unsigned fluidini,StDivDataGpu& dvd,const double2* posxy,const double* posz
    ,const unsigned* dcell,const float4* poscell,const float4* velrho,const typecode* code,const float* ftomassp
    ,float4* shiftvel,unsigned* fstype,float3* fsnormal,float* fsmindist,StrGeomVresGpu& vresgdata,hipStream_t stm)
{
  if(shiftadv){
    PreLoopInteractionT3 <tker,simulate2d,true > (bsfluid
        ,fluidnum,fluidini,dvd,posxy,posz,dcell,poscell,velrho,code,ftomassp
        ,shiftvel,fstype,fsnormal,fsmindist,vresgdata,stm);
  }
  else{
    PreLoopInteractionT3 <tker,simulate2d,false> (bsfluid
        ,fluidnum,fluidini,dvd,posxy,posz,dcell,poscell,velrho,code,ftomassp
        ,shiftvel,fstype,fsnormal,fsmindist,vresgdata,stm);
  }
}
//==============================================================================
  template<TpKernel tker> void PreLoopInteractionT(bool simulate2d,bool shiftadv
    ,unsigned bsfluid,unsigned fluidnum,unsigned fluidini,StDivDataGpu& dvd,const double2* posxy,const double* posz
    ,const unsigned* dcell,const float4* poscell,const float4* velrho,const typecode* code,const float* ftomassp
    ,float4* shiftvel,unsigned* fstype,float3* fsnormal,float* fsmindist,StrGeomVresGpu& vresgdata,hipStream_t stm)
{
  if(simulate2d){
    PreLoopInteractionT2 <tker,true > (shiftadv,bsfluid
        ,fluidnum,fluidini,dvd,posxy,posz,dcell,poscell,velrho,code,ftomassp
        ,shiftvel,fstype,fsnormal,fsmindist,vresgdata,stm);
  }
  else{
    PreLoopInteractionT2 <tker,false> (shiftadv,bsfluid
        ,fluidnum,fluidini,dvd,posxy,posz,dcell,poscell,velrho,code,ftomassp
        ,shiftvel,fstype,fsnormal,fsmindist,vresgdata,stm);
  }
}

//==============================================================================
  void PreLoopInteraction(TpKernel tkernel,bool simulate2d,bool shiftadv
    ,unsigned bsfluid,unsigned fluidnum,unsigned fluidini,StDivDataGpu& dvd,const double2* posxy,const double* posz
    ,const unsigned* dcell,const float4* poscell,const float4* velrho,const typecode* code,const float* ftomassp
    ,float4* shiftvel,unsigned* fstype,float3* fsnormal,float* fsmindist,StrGeomVresGpu& vresgdata,hipStream_t stm)
  {
    switch(tkernel){
    case KERNEL_Wendland:{ const TpKernel tker=KERNEL_Wendland;
      PreLoopInteractionT <tker> (simulate2d,shiftadv,bsfluid
        ,fluidnum,fluidini,dvd,posxy,posz,dcell,poscell,velrho,code,ftomassp
        ,shiftvel,fstype,fsnormal,fsmindist,vresgdata,stm);
    }break;
  #ifndef DISABLE_KERNELS_EXTRA
    case KERNEL_Cubic:{ const TpKernel tker=KERNEL_Cubic;
      PreLoopInteractionT <tker> (simulate2d,shiftadv,bsfluid
        ,fluidnum,fluidini,dvd,posxy,posz,dcell,poscell,velrho,code,ftomassp
        ,shiftvel,fstype,fsnormal,fsmindist,vresgdata,stm);
    }break;
  #endif
    default: throw "Kernel unknown at Interaction_MdbcCorrection().";
  }
  
    hipDeviceSynchronize();
    
  }

}
