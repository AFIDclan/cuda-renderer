#pragma once

#include <cuda_runtime.h>
#include "utils.hpp"
#include "Ray.hpp"
#include <cfloat>

struct TriangleHitInfo {
    float u, v, w;
	float3 hit_point;
	float3 normal;
	float2 uv;
    float dist;
	bool did_hit;
};

struct TrianglePrimitive {
    float3 vertices[3];
    float3 normal;
	float2 uv_coords[3];


    // Constructor
    __host__ __device__ TrianglePrimitive(float3 a, float3 b, float3 c) {
        vertices[0] = a;
        vertices[1] = b;
        vertices[2] = c;

        float3 v0 = vertices[1] - vertices[0];
        float3 v1 = vertices[2] - vertices[0];
        normal = normalize(cross(v0, v1));
    }


    __host__ __device__ TrianglePrimitive(float3 a, float3 b, float3 c, float3 normal)
        : normal(normal) {
        vertices[0] = a;
        vertices[1] = b;
        vertices[2] = c;

        float3 v0 = vertices[2] - vertices[0];
        float3 v1 = vertices[1] - vertices[0];

    }

    __host__ __device__ TrianglePrimitive(float3 a, float3 b, float3 c, float3 normal, float2 uv_a, float2 uv_b, float2 uv_c)
        : normal(normal) {
        vertices[0] = a;
        vertices[1] = b;
        vertices[2] = c;

        float3 v0 = vertices[2] - vertices[0];
        float3 v1 = vertices[1] - vertices[0];


        uv_coords[0] = uv_a;
		uv_coords[1] = uv_b;
		uv_coords[2] = uv_c;
    }

    __host__ __device__ TrianglePrimitive() : normal(make_float3(0.0f, 0.0f, 0.0f)) {
        vertices[0] = make_float3(0.0f, 0.0f, 0.0f);
        vertices[1] = make_float3(0.0f, 0.0f, 0.0f);
        vertices[2] = make_float3(0.0f, 0.0f, 0.0f);

        float3 v0 = vertices[2] - vertices[0];
        float3 v1 = vertices[1] - vertices[0];

    }

    __host__ __device__ float3 ray_intersect(const Ray& ray) {

        float denom = dot(ray.direction, normal);

        if (abs(denom) < 1e-6) {
            return make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
        }

        float t = dot(vertices[0] - ray.origin, normal) / denom;

        if (t < 0.0f) {
            return make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
        }

        float3 point = ray.origin + t * ray.direction;

        return point;
    }

    __host__ __device__ float3 center() const {
        return (vertices[0] + vertices[1] + vertices[2]) / 3.0f;
    }

    // https://github.com/SebLague/Ray-Tracing/blob/main/Assets/Scripts/Shaders/RayTracer.shader
    __host__ __device__ TriangleHitInfo ray_hit(const Ray& ray)
    {
		float3 edgeAB = vertices[1] - vertices[0];
		float3 edgeAC = vertices[2] - vertices[0];

		float3 ao = ray.origin - vertices[0];
		float3 dao = cross(ray.direction, ao);

		float det = -dot(ray.direction, normal);
		float inv_det = 1.0f / det;

		float dist = dot(ao, normal) * inv_det;

		float u = dot(dao, edgeAC) * inv_det;
		float v = dot(dao, edgeAB) * inv_det;
		float w = 1.0f - u - v;

		TriangleHitInfo hit_info;
		hit_info.u = u;
		hit_info.v = v;
		hit_info.w = w;
        hit_info.did_hit = det >= 1e-8 && dist >= 0 && u >= 0 && v >= 0 && w >= 0;
		hit_info.dist = dist;
		hit_info.hit_point = ray.origin + dist * ray.direction;

        // Interpolate the UV coordinates
        float2 uv0 = uv_coords[0];
        float2 uv1 = uv_coords[1];
        float2 uv2 = uv_coords[2];

        float2 interpolated_uv = (w * uv0) + (v * uv1) + (u * uv2);

		hit_info.uv = interpolated_uv;

        // for interpolating the normal
        //hit_info.normal = normalize(tri.normA * w + tri.normB * u + tri.normC * v);

		return hit_info;

    }


    __host__ __device__ float2 point_inside(const float3& point) const {


        float3 v0 = vertices[2] - vertices[0];
        float3 v1 = vertices[1] - vertices[0];
        float3 v2 = point - vertices[0];

        float dot00 = dot(v0, v0);
        float dot01 = dot(v0, v1);
        float dot02 = dot(v0, v2);
        float dot11 = dot(v1, v1);
        float dot12 = dot(v1, v2);

        float invDenom = 1.0f / (dot00 * dot11 - dot01 * dot01);
        float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
        float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

        // If point is inside the triangle, interpolate UV coordinates
        if ((u >= 0.0f) && (v >= 0.0f) && (u + v <= 1.0f)) {

            // Calculate barycentric weight for the third vertex
            float w = 1.0f - u - v;

            // Interpolate the UV coordinates
            float2 uv0 = uv_coords[0];
            float2 uv1 = uv_coords[1];
            float2 uv2 = uv_coords[2];

            float2 interpolated_uv = (w * uv0) + (v * uv1) + (u * uv2);

            return interpolated_uv;
        }

        return make_float2(FLT_MAX, FLT_MAX);
    }

};
