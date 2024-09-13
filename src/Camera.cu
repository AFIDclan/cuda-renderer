#include "Camera.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>


Camera::Camera(int width, int height, float3x3 K, float4 D) : width(width), height(height), K(K), D(D)
{
	// Define CUDA kernel launch configuration
	block_size = dim3(16, 20);
	grid_size = dim3((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

	K_inv = invert_intrinsic(K);

	pose = lre();

	cv::Matx33f K_cv(K.m[0][0], K.m[0][1], K.m[0][2], K.m[1][0], K.m[1][1], K.m[1][2], K.m[2][0], K.m[2][1], K.m[2][2]);
	cv::Matx41f D_cv(D.x, D.y, D.z, D.w);

	cv::Mat map_x(height, width, CV_32FC1);
	cv::Mat map_y(height, width, CV_32FC1);

	std::vector<cv::Point2f> distorted_points;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			cv::Point2f distorted(j, i);

			distorted_points.push_back(distorted);
		}
	}

	cv::fisheye::undistortPoints(distorted_points, distorted_points, K_cv, D_cv);

	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			map_x.at<float>(j, i) = distorted_points[j * width + i].x;
			map_y.at<float>(j, i) = distorted_points[j * width + i].y;
		}
	}

	projection_map_x.upload(map_x);
	projection_map_y.upload(map_y);

}

void Camera::render_scene(Scene& scene, uchar3* img_ptr, size_t pitch, bool synchronize)
{

	lre inv_pose = invert_lre(pose);

	render << <grid_size, block_size >> > (
		img_ptr,
		width,
		height,
		pitch,
		pose,
		projection_map_x,
		projection_map_y,
		inv_pose,
		scene.d_mesh_instances,
		scene.num_mesh_instances,
		scene.d_meshes,
		scene.d_materials
		);

	if (synchronize)
		cudaDeviceSynchronize();

}
