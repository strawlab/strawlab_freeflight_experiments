/* -*- Mode: C -*- */
/* osgCompute - Copyright (C) 2008-2009 SVT Group
*
* This library is free software; you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as
* published by the Free Software Foundation; either version 3 of
* the License, or (at your option) any later version.
*
* This library is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU Lesse General Public License for more details.
*
* The full license is in LICENSE file included with this distribution.
*/

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DEVICE FUNCTIONS //////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//------------------------------------------------------------------------------
inline __device__
float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}

//------------------------------------------------------------------------------
inline __device__
float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}


//------------------------------------------------------------------------------
inline __device__
float4 seed( float* seeds, unsigned int seedCount, unsigned int seedIdx, unsigned int ptclIdx, float3 bbmin, float3 bbmax )
{
    // random seed idx
    unsigned int idx1 = (seedIdx + ptclIdx) % seedCount;
    unsigned int idx2 = (idx1 + ptclIdx) % seedCount;
    unsigned int idx3 = (idx2 + ptclIdx) % seedCount;

    // seeds are within the range [0,1]
    float intFac1 = seeds[idx1];
    float intFac2 = seeds[idx2];
    float intFac3 = seeds[idx3];

    return make_float4(lerp(bbmin.x,bbmax.x,intFac1), lerp(bbmin.y,bbmax.y,intFac3),
        lerp(bbmin.z,bbmax.z,intFac2), 1);
}

//------------------------------------------------------------------------------
inline __device__
unsigned int thIdx()
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int width = gridDim.x * blockDim.x;

    return y*width + x;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GLOBAL FUNCTIONS //////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//------------------------------------------------------------------------------
__global__
void emitKernel( unsigned int numPtcls,
                   float4* ptcls,
                   float* seeds,
                   unsigned int seedIdx,
                   float3 bbmin,
                   float3 bbmax )
{
    // Receive particle pos
    unsigned int ptclIdx = thIdx();
    if( ptclIdx < numPtcls )
    {
        float4 curPtcl = ptcls[ptclIdx];

        // Reseed Particles if they
        // have moved out of the bounding box
        if( curPtcl.x < bbmin.x ||
            curPtcl.y < bbmin.y ||
            curPtcl.z < bbmin.z ||
            curPtcl.x > bbmax.x ||
            curPtcl.y > bbmax.y ||
            curPtcl.z > bbmax.z )
            ptcls[ptclIdx] = seed( seeds, numPtcls, seedIdx, ptclIdx, bbmin, bbmax );
    }
}

//------------------------------------------------------------------------------
__global__
void moveKernel( unsigned int numPtcls,
                 float4* ptcls,
                 float dx, float dy, float dz,
                 float rot_mat_00, float rot_mat_01,
                 float rot_mat_10, float rot_mat_11,
                 float centerx, float centery)
{
    unsigned int ptclIdx = thIdx();
    float4 p1, p2;

    if( ptclIdx < numPtcls )
    {
      p1 = ptcls[ptclIdx];

      // translate so observer is in middle of coordinate system
      p1 = make_float4( p1.x - centerx, p1.y - centery, p1.z, p1.w );

      // rotate
      p2 = make_float4( (rot_mat_00*p1.x + rot_mat_01*p1.y), (rot_mat_10*p1.x + rot_mat_11*p1.y), p1.z, p1.w );

      // reverse the translation
      p2 = make_float4( p2.x + centerx, p2.y + centery, p2.z, p2.w );

      // perform a euler step
      ptcls[ptclIdx] = p2 + make_float4(dx,dy,dz,0);
    }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// HOST FUNCTIONS ////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//------------------------------------------------------------------------------
extern "C" __host__
void emit(unsigned int numPtcls,
            void* ptcls,
            void* seeds,
            unsigned int seedIdx,
            float3 bbmin,
            float3 bbmax )
{
    dim3 blocks( (numPtcls / 128)+1, 1, 1 );
    dim3 threads( 128, 1, 1 );

    emitKernel<<< blocks, threads >>>(
        numPtcls,
        (float4*)ptcls,
        (float*)seeds,
        seedIdx,
        bbmin,
        bbmax );
}

//------------------------------------------------------------------------------
extern "C" __host__
void move( unsigned int numPtcls,
           void* ptcls,
           float dx,
           float dy,
           float dz,
           float rot_mat_00, float rot_mat_01,
           float rot_mat_10, float rot_mat_11,
           float centerx, float centery)
{
    dim3 blocks( (numPtcls / 128)+1, 1, 1 );
    dim3 threads( 128, 1, 1 );

    moveKernel<<< blocks, threads >>>(
        numPtcls,
        (float4*)ptcls,
        dx, dy, dz,
        rot_mat_00, rot_mat_01,
        rot_mat_10, rot_mat_11,
        centerx, centery);
}
