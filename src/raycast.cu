#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "transforms.hpp"
#include "Ray.hpp"
#include "TrianglePrimitive.hpp"
#include "MeshPrimitive.h"
#include "MeshInstance.hpp"
#include "Material.hpp"

struct HitInfo
{
    float min = FLT_MAX;
    TrianglePrimitive triangle;
    float3 location;
    float3 normal;
    Material material;
    float2 uv;
};


__device__ HitInfo& cast_ray(Ray& ray, MeshInstance* mesh_instances, int num_mesh_instances, d_MeshPrimitive* meshes, Material* materials, bool lighting_pass = false, float light_distance = FLT_MAX)
{

    HitInfo hit_info;

    for (int mesh_idx = 0; mesh_idx < num_mesh_instances; mesh_idx++) {

        MeshInstance mesh_instance = mesh_instances[mesh_idx];
        d_MeshPrimitive mesh = meshes[mesh_instance.mesh_index];

        // Express the ray direction in mesh coordinates
        float3 r_direction = apply_euler(mesh_instance.rotation, ray.direction);

        r_direction.x *= mesh_instance.inv_scale.x;
        r_direction.y *= mesh_instance.inv_scale.y;
        r_direction.z *= mesh_instance.inv_scale.z;

        // Express the ray origin in mesh coordinates
        float3 r_origin = apply_lre(mesh_instance.pose, ray.origin);

        // Eg. Scale of 2 --> multiply the origin times 0.5 and make the object appear 2x the size
        r_origin.x *= mesh_instance.inv_scale.x;
        r_origin.y *= mesh_instance.inv_scale.y;
        r_origin.z *= mesh_instance.inv_scale.z;

        Ray r_ray = Ray(
            r_origin,
            r_direction
        );


        int stack[32];
        int stack_index = 0;

        // Start with the root node
        stack[stack_index++] = 0;

        while (stack_index > 0) {
            int node_index = stack[--stack_index];
            d_BVHTree current_bvh = mesh.bvh_top[node_index];

            // We are assuming this ray intersects with the bounding box of the node since it was pushed onto the stack

            if (current_bvh.count_triangles < 0) {
                // If the node has children, push them onto the stack

                float dist_a = mesh.bvh_top[current_bvh.start_index].ray_intersects(r_ray);
                float dist_b = mesh.bvh_top[current_bvh.start_index + 1].ray_intersects(r_ray);

                if (dist_a < dist_b) {
                    if (dist_b < hit_info.min) stack[stack_index++] = current_bvh.start_index + 1;
                    if (dist_a < hit_info.min) stack[stack_index++] = current_bvh.start_index;
                }
                else {
                    if (dist_a < hit_info.min) stack[stack_index++] = current_bvh.start_index;
                    if (dist_b < hit_info.min) stack[stack_index++] = current_bvh.start_index + 1;
                }


            }
            else {
                // Leaf node: check for intersections with triangles
                for (int i = 0; i < current_bvh.count_triangles; i++) {
                    int index = current_bvh.start_index + i;

                    // Positive means the ray is facing the same direction as the normal and we hit the back of the triangle
                    float same_dir = dot(r_ray.direction, mesh.triangles[index].normal);

					if (same_dir > 0) continue;


                    float3 intersection = mesh.triangles[index].ray_intersect(r_ray);

                    // If the intersection is at FLT_MAX, the ray did not intersect with the triangle
                    if (intersection.x == FLT_MAX)
                        continue;

                    float2 uv = mesh.triangles[index].point_inside(intersection);


                    if (uv.x != FLT_MAX) {

                        // Express the location in world coordinates
                        //float3 intersection = r_ray.origin + r_ray.direction * dist;

                        hit_info.location.x = intersection.x * mesh_instance.scale.x;
                        hit_info.location.y = intersection.y * mesh_instance.scale.y;
                        hit_info.location.z = intersection.z * mesh_instance.scale.z;

                        hit_info.location = apply_lre(mesh_instance.inv_pose, hit_info.location);

                        float distance = magnitude(hit_info.location - ray.origin);

                        if (hit_info.min == FLT_MAX || distance < hit_info.min) {
                            hit_info.min = distance;
                            hit_info.triangle = mesh.triangles[index];


                            // Express normal in world coordinates
                            hit_info.normal = apply_euler(mesh_instance.inv_rotation, hit_info.triangle.normal);

                            hit_info.normal.x *= mesh_instance.scale.x;
                            hit_info.normal.y *= mesh_instance.scale.y;
                            hit_info.normal.z *= mesh_instance.scale.z;

                            // Scaling the direction can un-normalize it
                            hit_info.normal = normalize(hit_info.normal);

                            hit_info.uv = uv;

                            hit_info.material = materials[mesh_instance.material_index];

                        }
                    }
                }
            }
        }
    }

    return hit_info;
}


// Simple CUDA kernel to invert image colors
__global__ void render(uchar3* img, int width, int height, size_t pitch, const lre camera_pose, cv::cuda::PtrStepSz<float> projection_map_x, cv::cuda::PtrStepSz<float> projection_map_y, const lre inv_camera_pose, MeshInstance* mesh_instances, int num_mesh_instances, d_MeshPrimitive* meshes, Material* materials) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }


    uchar3* row = (uchar3*)((char*)img + y * pitch);

    float3 origin = make_float3(camera_pose.x, camera_pose.y, camera_pose.z);

    float x_projected = projection_map_x(y, x);
    float y_projected = projection_map_y(y, x);

	float3 direction = make_float3(x_projected, y_projected, 1.0f);


	float a = dot(normalize(direction), make_float3(0, 0, 1));

	if (a < 0.43)
	{
		row[x].x = 0;
		row[x].y = 0;
		row[x].z = 0;
		return;
	}


    // Rotate by 90 deg to make y forward (world space)
    direction = make_float3(direction.x, direction.z, -direction.y);

    // Apply the camera's pose to the direction
    direction = apply_euler(make_float3(inv_camera_pose.yaw, inv_camera_pose.pitch, inv_camera_pose.roll), direction);

    //Camera Ray direction in world space
    direction = normalize(direction);


    float3 accumulated_color = make_float3(0.0, 0.0, 0.0);

    Ray ray = Ray(
        origin,
        direction
    );

	HitInfo hit_info = cast_ray(ray, mesh_instances, num_mesh_instances, meshes, materials, false, 0.0);

    // Hit nothing. Return the sky color
    if (hit_info.min == FLT_MAX) {

        // Pale blue sky
        row[x].x = (1.0 * 255);
        row[x].y = (0.8 * 255);
        row[x].z = (0.6 * 255);

        return;
    }

    // Move the ray to the hit location
    ray.origin = hit_info.location;

    // Apply the color of the triangle to the ray


    if (hit_info.material.texture_width > 0)
    {
        int tex_x = (hit_info.uv.x * (float)hit_info.material.texture_width);
        int tex_y = ((1.0 - hit_info.uv.y) * (float)hit_info.material.texture_height);

        tex_x = fmaxf(tex_x % hit_info.material.texture_width, 0);
        tex_y = fmaxf(tex_y % hit_info.material.texture_height, 0);

        uchar3* tex_row = (uchar3*)((char*)hit_info.material.texture + tex_y * hit_info.material.texture_pitch);

        uchar3 tex_color = tex_row[tex_x];

        // 1/255 = 0.0039215
        ray.color.x *= tex_color.x * 0.0039215f;
        ray.color.y *= tex_color.y * 0.0039215f;
        ray.color.z *= tex_color.z * 0.0039215f;
    }
    else {
        ray.color.x *= hit_info.material.albedo.x;
        ray.color.y *= hit_info.material.albedo.y;
        ray.color.z *= hit_info.material.albedo.z;
    }

    ray.illumination = 1.0;

    ray.illumination = fminf(1.0, ray.illumination);
    ray.illumination = fmaxf(0.4, ray.illumination);

    row[x].x = (ray.illumination * ray.color.x * 255);
    row[x].y = (ray.illumination * ray.color.y * 255);
    row[x].z = (ray.illumination * ray.color.z * 255);


}
