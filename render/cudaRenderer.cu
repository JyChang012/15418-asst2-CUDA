#define TILE 64  // side size of tiles, note the image size is guaranteed to be multiple of 64
#define SHARED 1024
#define BLOCK 32  // block size for computeTile
// #define DEBUG  // for debugging!
#define SCAN_BLOCK_DIM 256  // needed by sharedMemExclusiveScan implementation

#include <string>
#include <algorithm>
#include <cstdio>

#define _USE_MATH_DEFINES

#include <math.h>
#include <stdio.h>
#include <vector>


#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"
#include "circleBoxTest.cu_inl"
#include "cuErrCheck.h"  // for debugging
#include "exclusiveScan.cu_inl"

using std::printf;


__host__ __device__
inline uint floordiv(uint a, uint b) {
    return a / b + (a % b != 0);
}

////////////////////////////////////////////////////////////////////////////////////////
// All cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

// This stores the global constants
struct GlobalConstants {

    SceneName sceneName;

    int numberOfCircles;

    float *position;
    float *velocity;
    float *color;
    float *radius;

    int imageWidth;
    int imageHeight;
    float *imageData;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// Read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int cuConstNoiseYPermutationTable[256];
__constant__ int cuConstNoiseXPermutationTable[256];
__constant__ float cuConstNoise1DValueTable[256];

// Color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float cuConstColorRamp[COLOR_MAP_SIZE][3];


// Include parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"


// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height - imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // Write to global memory: As an optimization, this code uses a float4
    // store, which results in more efficient code than if it were coded as
    // four separate float stores.
    *(float4 * )(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // Write to global memory: As an optimization, this code uses a float4
    // store, which results in more efficient code than if it were coded as
    // four separate float stores.
    *(float4 * )(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
// 
// Update positions of fireworks
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = M_PI;
    const float maxDist = 0.25f;

    float *velocity = cuConstRendererParams.velocity;
    float *position = cuConstRendererParams.position;
    float *radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numberOfCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update 
        return;
    }

    // Determine the firework center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i + 1];

    // Update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j + 1] += velocity[index3j + 1] * dt;

    // Firework sparks
    float sx = position[index3j];
    float sy = position[index3j + 1];

    // Compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // Compute distance from fire-work 
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position 
        // Random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi) / NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j + 1] = position[index3i + 1] + y;
        position[index3j + 2] = 0.0f;

        // Travel scaled unit length 
        velocity[index3j] = cosA / 5.0;
        velocity[index3j + 1] = sinA / 5.0;
        velocity[index3j + 2] = 0.0f;
    }
}

// kernelAdvanceHypnosis   
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numberOfCircles)
        return;

    float *radius = cuConstRendererParams.radius;

    float cutOff = 0.5f;
    // Place circle back in center after reaching threshold radisus 
    if (radius[index] > cutOff) {
        radius[index] = 0.02f;
    } else {
        radius[index] += 0.01f;
    }
}


// kernelAdvanceBouncingBalls
// 
// Update the position of the balls
__global__ void kernelAdvanceBouncingBalls() {
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numberOfCircles)
        return;

    float *velocity = cuConstRendererParams.velocity;
    float *position = cuConstRendererParams.position;

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3 + 1];
    float oldPosition = position[index3 + 1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition 
        return;
    }

    if (position[index3 + 1] < 0 && oldVelocity < 0.f) { // bounce ball
        velocity[index3 + 1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3 + 1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3 + 1] += velocity[index3 + 1] * dt;

    if (fabsf(velocity[index3 + 1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(position[index3 + 1] - oldPosition) < epsilon) { // stop ball
        velocity[index3 + 1] = 0.f;
        position[index3 + 1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// Move the snowflake animation forward one time step.  Update circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numberOfCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float *positionPtr = &cuConstRendererParams.position[index3];
    float *velocityPtr = &cuConstRendererParams.velocity[index3];

    // Load from global memory
    float3 position = *((float3 *) positionPtr);
    float3 velocity = *((float3 *) velocityPtr);

    // Hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // Add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // Drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // Update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // Update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // If the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ((position.y + radius < 0.f) ||
        (position.x + radius) < -0.f ||
        (position.x - radius) > 1.f) {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // Restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // Store updated positions and velocities to global memory
    *((float3 *) positionPtr) = position;
    *((float3 *) velocityPtr) = velocity;
}

// shadePixel -- (CUDA device code)
//
// Given a pixel and a circle, determine the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ __inline__ void
shadePixel(float2 pixelCenter, float3 p, float4 *imagePtr, int circleIndex) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];;
    float maxDist = rad * rad;

    // Circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    float3 rgb;
    float alpha;

    // There is a non-zero contribution.  Now compute the shading value

    // Suggestion: This conditional is in the inner loop.  Although it
    // will evaluate the same for all threads, there is overhead in
    // setting up the lane masks, etc., to implement the conditional.  It
    // would be wise to perform this logic outside of the loops in
    // kernelRenderCircles.  (If feeling good about yourself, you
    // could use some specialized template magic).
    if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrt(pixelDist) / rad;
        rgb = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f - p.z);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
        alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    } else {
        // Simple: each circle has an assigned color
        int index3 = 3 * circleIndex;
        rgb = *(float3 * ) & (cuConstRendererParams.color[index3]);
        alpha = .5f;
    }

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read

    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;

    // Global memory write
    *imagePtr = newColor;

    // END SHOULD-BE-ATOMIC REGION
}

// kernelRenderCircles -- (CUDA device code)
//
// Each thread renders a circle.  Since there is no protection to
// ensure order of update or mutual exclusion on the output image, the
// resulting image will be incorrect.
__global__ void kernelRenderCircles() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numberOfCircles)
        return;

    int index3 = 3 * index;

    // Read position and radius
    float3 p = *(float3 * )(&cuConstRendererParams.position[index3]);
    float rad = cuConstRendererParams.radius[index];

    // Compute the bounding box of the circle. The bound is in integer
    // screen coordinates, so it's clamped to the edges of the screen.
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;
    short minX = static_cast<short>(imageWidth * (p.x - rad));
    short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
    short minY = static_cast<short>(imageHeight * (p.y - rad));
    short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

    // A bunch of clamps.  Is there a CUDA built-in for this?
    short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
    short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
    short screenMinY = (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
    short screenMaxY = (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    // For all pixels in the bounding box
    for (int pixelY = screenMinY; pixelY < screenMaxY; pixelY++) {
        float4 *imgPtr = (float4 * )(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + screenMinX)]);
        for (int pixelX = screenMinX; pixelX < screenMaxX; pixelX++) {
            float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                                 invHeight * (static_cast<float>(pixelY) + 0.5f));
            shadePixel(pixelCenterNorm, p, imgPtr, index);
            imgPtr++;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////
// additional kernel
/*
 * output shape: X by Y by # of circles
 */


__device__ __inline__ void
shadePixelCustom(float2 const pixelCenter, float3 const p, float4& pixel, float const rad, float3 const* colorPtr) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float maxDist = rad * rad;

    // Circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    float3 rgb;
    float alpha;

    // There is a non-zero contribution.  Now compute the shading value

    // Suggestion: This conditional is in the inner loop.  Although it
    // will evaluate the same for all threads, there is overhead in
    // setting up the lane masks, etc., to implement the conditional.  It
    // would be wise to perform this logic outside of the loops in
    // kernelRenderCircles.  (If feeling good about yourself, you
    // could use some specialized template magic).
    if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrt(pixelDist) / rad;
        rgb = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f - p.z);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
        alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    } else {
        // Simple: each circle has an assigned color
        rgb = *colorPtr;
        alpha = .5f;
    }

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION

    pixel.x *= oneMinusAlpha;
    pixel.y *= oneMinusAlpha;
    pixel.z *= oneMinusAlpha;

    pixel.x += alpha * rgb.x;
    pixel.y += alpha * rgb.y;
    pixel.z += alpha * rgb.z;
    pixel.w += alpha;

    // END SHOULD-BE-ATOMIC REGION
}

__global__
void checkCirclesInTile(uint *output, int X, int Y) {
    int const x = threadIdx.z + blockIdx.z * blockDim.z;  // width
    int const y = threadIdx.y + blockIdx.y * blockDim.y;  // height
    int const circle_idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (x >= X or y >= Y or circle_idx >= cuConstRendererParams.numberOfCircles) return;

    int const min_x = x * TILE;
    int const min_y = y * TILE;
    int const max_x = min(min_x + TILE - 1, cuConstRendererParams.imageWidth - 1);
    int const max_y = min(min_y + TILE - 1, cuConstRendererParams.imageHeight - 1);

    float *position = cuConstRendererParams.position + (circle_idx * 3);
    float radius = cuConstRendererParams.radius[circle_idx];

    // TODO: save this to global
    float const invWidth = 1.f / cuConstRendererParams.imageWidth;
    float const invHeight = 1.f / cuConstRendererParams.imageHeight;
    // float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
    //                                      invHeight * (static_cast<float>(pixelY) + 0.5f));
    float const min_x_norm = invWidth * (static_cast<float>(min_x) + 0.5f);
    float const min_y_norm = invHeight * (static_cast<float>(min_y) + 0.5f);
    float const max_x_norm = invWidth * (static_cast<float>(max_x) + 0.5f);
    float const max_y_norm = invHeight * (static_cast<float>(max_y) + 0.5f);

    output[
            circle_idx
            + x * cuConstRendererParams.numberOfCircles
            + y * cuConstRendererParams.numberOfCircles * X
    ] = circleInBox(position[0], position[1], radius, min_x_norm, max_x_norm, max_y_norm, min_y_norm);
}


__global__
void scanLastDimShared(uint *output, int X, int Y) {
    int const x = blockIdx.x;
    int const y = blockIdx.y;  // BlockDim should be (N, 1, 1)
    int const linearIdx = threadIdx.x;
    int const numberOfCircles = cuConstRendererParams.numberOfCircles;
    output += (x + y * X) * numberOfCircles;

    __shared__ uint prefixSumInput[SCAN_BLOCK_DIM];
    __shared__ uint prefixSumOutput[SCAN_BLOCK_DIM];
    __shared__ uint prefixSumScratch[2 * SCAN_BLOCK_DIM];

    __shared__ uint prev;
    if (linearIdx == 0) {
        prev = 0;
        prefixSumInput[SCAN_BLOCK_DIM - 1] = 0;
    }
    uint const segSize = SCAN_BLOCK_DIM - 1;
    for (int i = 0; i < numberOfCircles; i += segSize) {
        int const ed = min(numberOfCircles, i + segSize);
        // collaborative fetch
        if (linearIdx + i < ed) {
            prefixSumInput[linearIdx] = output[i + linearIdx] + (linearIdx == 0 ? prev : 0);
        }

        __syncthreads();

        sharedMemExclusiveScan(linearIdx, prefixSumInput, prefixSumOutput, prefixSumScratch, SCAN_BLOCK_DIM);

        __syncthreads();

        if (linearIdx == SCAN_BLOCK_DIM - 1) {
            prev = prefixSumOutput[SCAN_BLOCK_DIM - 1];
        }

        __syncthreads();

        if (linearIdx + i < ed) {
            output[linearIdx + i] = prefixSumOutput[linearIdx + 1];
        }

    }
    /*
    if (linearIdx == 0) {
        printf("Block (%d, %d):", blockIdx.x, blockIdx.y);
        for (int i = 0; i < min(2048, numberOfCircles); i++) {
            printf(" %d", output[i]);
        }
        printf("\n");
    }
     */
}

__global__
void scanLastDim(uint *output, int X, int Y) {
    /*
    int const x = threadIdx.x + blockIdx.x * blockDim.x;  // width
    int const y = threadIdx.y + blockIdx.y * blockDim.y;  // height

    if (x >= X or y >= Y) return;

    int const start = x * cuConstRendererParams.numberOfCircles
                      + y * TILE * cuConstRendererParams.numberOfCircles;
    */
    // one thread per block!
    uint const x = blockIdx.x;  // width
    uint const y = blockIdx.y;  // height

    uint *start = output + (x * cuConstRendererParams.numberOfCircles
                           + y * cuConstRendererParams.numberOfCircles * X);

    thrust::inclusive_scan(thrust::device,
                           start,
                           start + cuConstRendererParams.numberOfCircles,
                           start);
}

/*
 * input and output shape Y by X by # of circles
 */
__global__
void place2startKernel(uint const *input, uint *output, int X, int Y) {
    int const x = threadIdx.z + blockIdx.z * blockDim.z;  // width
    int const y = threadIdx.y + blockIdx.y * blockDim.y;  // height
    int const pos = threadIdx.x + blockIdx.x * blockDim.x;

    int const numberOfCircles = cuConstRendererParams.numberOfCircles;

    if (x >= X or y >= Y or pos >= numberOfCircles) return;

    uint const offset = x * numberOfCircles + y * numberOfCircles * X;
    input += offset;
    output += offset;

    uint const v = input[pos];
    if (pos == 0 and v == 1 or pos > 0 and v != input[pos - 1]) {
        output[v - 1] = pos;
    }
}


__global__
void computeTile(uint const *relevantCircles, uint const *isInTile, int X, int Y) {
    int const x = threadIdx.x + blockIdx.x * blockDim.x;  // width
    int const y = threadIdx.y + blockIdx.y * blockDim.y;  // height

    short const imageWidth = cuConstRendererParams.imageWidth;
    short const imageHeight = cuConstRendererParams.imageHeight;
    int const numberOfCircles = cuConstRendererParams.numberOfCircles;


    uint const tx = x / TILE, ty = y / TILE;
    uint const offset = tx * numberOfCircles + ty * numberOfCircles * X;
    relevantCircles += offset;
    isInTile += offset;

    // TODO: move this to constant memory!
    float const invWidth = 1.f / imageWidth;
    float const invHeight = 1.f / imageHeight;

    // int const pixelOffset = 4 * (x + y * imageWidth);
    float2 const pixelCenterNorm = make_float2(invWidth * (static_cast<float>(x) + 0.5f),
                                               invHeight * (static_cast<float>(y) + 0.5f));

    float4 *const imgPtr = (float4 *)(cuConstRendererParams.imageData + 4 * (y * imageWidth + x));
    float4 pixel;
    if (x < imageWidth and y < imageHeight) {
        pixel = *imgPtr;
        // for debug
        // printf("block (%d, %d), relevantCircles (%d, %d)\n", blockIdx.x, blockIdx.y, tx, ty);
    }

    /*
     * copy them to shared memory
     * float *position;
     * float *color;
     * float *radius;
     */
    // note: we assume that TILE size is an integer multiple of block size
    uint const relevantSize = isInTile[numberOfCircles - 1];  // size of relevant circles
    uint const linearIdx = threadIdx.x + blockDim.x * threadIdx.y;
    // int const linearIdx =  threadIdx.y * blockDim.y + threadIdx.x;  // linear idx that form a warp!
    uint const totalThreads = blockDim.x * blockDim.y;

    uint const moveAmount = floordiv(SHARED, totalThreads);

    __shared__ float tileRad[BLOCK * BLOCK];

    __shared__ float tilePos[3 * BLOCK * BLOCK];
    float3* position3 = (float3*) cuConstRendererParams.position;
    float3* tilePos3 = (float3*) tilePos;

    __shared__ float tileCol[3 * BLOCK * BLOCK];
    float3* color3 = (float3*) cuConstRendererParams.color;
    float3 *tileCol3 = (float3*) tileCol;

    for (uint i = 0; i < relevantSize; i += SHARED) {
        uint const ed = min(relevantSize, i + SHARED);

        // fetch from global memory
        if (linearIdx + i < ed) {
            uint const circleIdx = relevantCircles[linearIdx + i];
            tileRad[linearIdx] = cuConstRendererParams.radius[circleIdx];
            tilePos3[linearIdx] = position3[circleIdx];
            if (cuConstRendererParams.sceneName != SNOWFLAKES
                and cuConstRendererParams.sceneName != SNOWFLAKES_SINGLE_FRAME) {
                tileCol3[linearIdx] =  color3[circleIdx];
            }
        }
        /*
        for (int from_i = i + linearIdx, to_i = linearIdx; from_i < ed; from_i += totalThreads, to_i += totalThreads) {
            int const circleIdx = relevantCircles[from_i];
            tileRad[to_i] = cuConstRendererParams.radius[circleIdx];
            tilePos3[to_i] = position3[circleIdx];
            if (cuConstRendererParams.sceneName != SNOWFLAKES
                and cuConstRendererParams.sceneName != SNOWFLAKES_SINGLE_FRAME) {
                tileCol3[to_i] = color3[circleIdx];
            }

        }
        */
        /*
        int cpBg = i + linearIdx * moveAmount;
        int cpEd = min(ed, cpBg + moveAmount);
        for (int j = cpBg; j < cpEd; j++) {
            int const circleIdx = relevantCircles[j];
            tileRad[j - i] = cuConstRendererParams.radius[circleIdx];
            tilePos3[j - i] = position3[circleIdx];
            if (cuConstRendererParams.sceneName != SNOWFLAKES
                and cuConstRendererParams.sceneName != SNOWFLAKES_SINGLE_FRAME) {
                tileCol3[j - i] = color3[circleIdx];
            }
        }
        */
        __syncthreads();

        if (x < imageWidth and y < imageHeight) {
            for (uint j = i; j < ed; j++) {
                shadePixelCustom(pixelCenterNorm,
                                 tilePos3[j - i],
                                 pixel,
                                 tileRad[j - i],
                                 tileCol3 + (j - i));
            }
        }
        __syncthreads();
    };
    if (x >= imageWidth || y >= imageHeight) return;

    // write back to data
    *imgPtr = pixel;
}


////////////////////////////////////////////////////////////////////////////////////////


CudaRenderer::CudaRenderer() {
    image = NULL;

    numberOfCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete[] position;
        delete[] velocity;
        delete[] color;
        delete[] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
    }

    if (isInTile) cudaFree(isInTile);
    if (relevantCircles) cudaFree(relevantCircles);
}

const Image *
CudaRenderer::getImage() {

    // Need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numberOfCircles, position, velocity, color, radius);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    bool isFastGPU = false;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;
        if (name.compare("GeForce RTX 2080") == 0) {
            isFastGPU = true;
        }

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    if (!isFastGPU) {
        printf("WARNING: "
               "You're not running on a fast GPU, please consider using "
               "NVIDIA RTX 2080.\n");
        printf("---------------------------------------------------------\n");
    }

    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numberOfCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numberOfCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numberOfCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numberOfCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numberOfCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numberOfCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numberOfCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numberOfCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numberOfCircles = numberOfCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // Also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int *permX;
    int *permY;
    float *value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // Copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
            {1.f, 1.f,  1.f},
            {1.f, 1.f,  1.f},
            {.8f, .9f,  1.f},
            {.8f, .9f,  1.f},
            {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

    // init buffer for storing intermediate results
    int const TW = floordiv(image->width, TILE);  // X
    int const TH = floordiv(image->height, TILE);  // Y
    cudaCheckError(cudaMalloc(&isInTile, sizeof(uint) * TH * TW * numberOfCircles));
    cudaCheckError(cudaMalloc(&relevantCircles, sizeof(uint) * TH * TW * numberOfCircles));

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
            (image->width + blockDim.x - 1) / blockDim.x,
            (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
    // 256 threads per block is a healthy number
    dim3 blockDim(256, 1);
    dim3 gridDim((numberOfCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) {
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>();
    }
    cudaDeviceSynchronize();
}


/*
 * X: width, stride = 1
 * Y: height (major), stride = width
 */
void
CudaRenderer::render() {
    // W by H tiles, each has size TILE by TILE pixels
    uint const width = image->width;
    uint const height = image->height;
    uint const TW = floordiv(width, TILE);  // X
    uint const TH = floordiv(height, TILE);  // Y

    // for each tile, check if each circle is within the tile, store a boolean
    uint const cBlock = 256, wBlock = 1, hBlock = 1;
    dim3 gridDim(
            floordiv(numberOfCircles, cBlock),
            floordiv(TH, hBlock),
            floordiv(TW, wBlock)
    );
    dim3 blockDim(cBlock, hBlock, wBlock); // 256 threads per block
    checkCirclesInTile<<<gridDim, blockDim>>>(isInTile, TW, TH);

    // exclusive scan the output array
    // scanLastDim<<<dim3(TW, TH), 1>>>(isInTile, TW, TH);
    scanLastDimShared<<<dim3(TW, TH), dim3(SCAN_BLOCK_DIM)>>>(isInTile, TW, TH);

    // place indices of one to start
    place2startKernel<<<gridDim, blockDim>>>(isInTile, relevantCircles, TW, TH);

    // launch kernel!
    dim3 kernelGridDim(floordiv(width, BLOCK), floordiv(height, BLOCK));
    dim3 kernelBlockDim(BLOCK, BLOCK);
    computeTile<<<kernelGridDim, kernelBlockDim>>>(relevantCircles, isInTile, TW, TH);

    // cudaCheckError(cudaDeviceSynchronize());

}
