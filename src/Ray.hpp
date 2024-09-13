#pragma once

#include <cuda_runtime.h>

struct Ray {
    float3 origin;
    float3 direction;
	float3 direction_inv;
    float3 color;
    float illumination;

    bool terminated = false;

    // Constructor
    __device__ Ray(float3 o, float3 d)
        : origin(o), direction(d), illumination(0.0f) {

		direction_inv = make_float3(1.0f / d.x, 1.0f / d.y, 1.0f / d.z);
        color = make_float3(1.0, 1.0, 1.0);
    }
};


