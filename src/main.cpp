#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <cuda_runtime.h>
#include <iostream>

#include "utils.hpp"
#include "Ray.hpp"
#include "TrianglePrimitive.hpp"
#include "MeshPrimitive.h"
#include "MeshInstance.hpp"
#include "Material.hpp"
#include "Scene.h"
#include <curand_kernel.h>
#include "OBJLoader.hpp"
#include "Camera.h"

struct MouseParams
{
    int last_x;
    int last_y;
    bool has_last = false;

    bool is_down = false;

    lre* pose;
};

void display_image(uchar3* d_img, int width, int height, size_t pitch, double fps, MouseParams& mouse_state)
{
    // Wrap the CUDA memory in an OpenCV GpuMat
    cv::cuda::GpuMat img_gpu(height, width, CV_8UC3, d_img, pitch);

    // Download the processed image back to host memory
    cv::Mat img_cpu;
    img_gpu.download(img_cpu);

    // Convert FPS to string and overlay it on the image
    std::string fps_text = "FPS: " + std::to_string(fps);
    cv::putText(img_cpu, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);


     //Display the image using OpenCV
     cv::imshow("Image", img_cpu);

     //Capture key pressed
     int key = cv::waitKey(1);

     if (key == 'w')
     {
	 	lre inv_camera_pose = invert_lre(*mouse_state.pose);

	 	float3 forward = make_float3(0, 0.1, 0);
       
	 	float3 new_pos = apply_lre(inv_camera_pose, forward);

	 	mouse_state.pose->x = new_pos.x;
	 	mouse_state.pose->y = new_pos.y;
	 	mouse_state.pose->z = new_pos.z;
     }
	 else if (key == 's')
	 {
   
         lre inv_camera_pose = invert_lre(*mouse_state.pose);

         float3 forward = make_float3(0, -0.1, 0);

         float3 new_pos = apply_lre(inv_camera_pose, forward);

         mouse_state.pose->x = new_pos.x;
         mouse_state.pose->y = new_pos.y;
         mouse_state.pose->z = new_pos.z;
        
	 }
	 else if (key == 'a')
	 {
        
         lre inv_camera_pose = invert_lre(*mouse_state.pose);

         float3 forward = make_float3(-0.1, 0.0, 0);

         float3 new_pos = apply_lre(inv_camera_pose, forward);

         mouse_state.pose->x = new_pos.x;
         mouse_state.pose->y = new_pos.y;
         mouse_state.pose->z = new_pos.z;
        
     }
     else if (key == 'd')
     {

         lre inv_camera_pose = invert_lre(*mouse_state.pose);

         float3 forward = make_float3(0.1, 0.0, 0);

         float3 new_pos = apply_lre(inv_camera_pose, forward);

         mouse_state.pose->x = new_pos.x;
         mouse_state.pose->y = new_pos.y;
         mouse_state.pose->z = new_pos.z;

     }

     //If the key pressed is 'q', then exit the loop
     if (key == 'q') {
         exit(0);
     }
}


void on_mouse(int event, int x, int y, int, void* param)
{
    // Cast the param back to the correct type
    MouseParams* mouse_state = static_cast<MouseParams*>(param);
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        mouse_state->is_down = true;
    } else if (event == cv::EVENT_LBUTTONUP)
    {
        mouse_state->is_down = false;
    } else if (event == cv::EVENT_MOUSEMOVE)
    {
        
        if (mouse_state->has_last && mouse_state->is_down)
        {
            int dx = x - mouse_state->last_x;
            int dy = y - mouse_state->last_y;

            mouse_state->pose->yaw += dx * 0.001;
            mouse_state->pose->pitch += dy * -0.001;
        }

        mouse_state->last_x = x;
        mouse_state->last_y = y;
        mouse_state->has_last = true;
    }
}

int main() {



    // Image dimensions

    double fps = 0.0;

    int64 start_time = 0;
    int64 end_time = 0;

    int width = 1920;
    int height = 1080;

	float4 D = make_float4(0.016233999489849514, -0.013875757716177956, 0.03264329940126211, -0.019561619947134234);

    float3x3 K = {
        862.097835972576, 0.0, 998.1702383680802,
        0,     862.1368447300727, 569.6759403225842,
        0,     0,   1
    };

	Camera camera = Camera(width, height, K, D);

    camera.pose.x = -1;
    camera.pose.y = -4;
    camera.pose.z = 2;

	Scene scene = Scene();

    Material* mat = new Material();
    mat->upload_texture("556_downstairs.png");

    scene.add_material(*mat);

    MeshPrimitive* mesh = OBJLoader::load("556_downstairs.obj");
    scene.add_mesh(*mesh);

    mesh->bvh_top.print_stats();

    MeshInstance* mesh_instance = new MeshInstance(0, 0);

    scene.add_mesh_instance(*mesh_instance);

    scene.upload_to_device();


    // Allocate CUDA memory for the image
    uchar3* d_img;
    size_t pitch;
    cudaMallocPitch(&d_img, &pitch, width * sizeof(uchar3), height);
    // Allocate CUDA memory for the image
    uchar3* d_img2;
    size_t pitch2;
    cudaMallocPitch(&d_img2, &pitch2, width * sizeof(uchar3), height);

  

	float angle = 0.0f;

    MouseParams mouse_state;
    mouse_state.pose = &camera.pose;

     cv::namedWindow("Image");
     cv::setMouseCallback("Image", on_mouse, &mouse_state);

    // Loop while program is running
	 while (true)
     {
        // Start measuring time

        start_time = cv::getTickCount();

		camera.render_scene(scene, d_img, pitch);
		camera.render_scene(scene, d_img2, pitch);
        cudaDeviceSynchronize();

        auto err = cudaGetLastError();

		if (err != cudaSuccess)
		{
			std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
		}

        // End measuring time
        end_time = cv::getTickCount();
        double time_taken = (end_time - start_time) / cv::getTickFrequency();
        fps = 1.0 / time_taken;

		display_image(d_img, width, height, pitch, fps, mouse_state);
    }

    // Free CUDA memory
    cudaFree(d_img);

    return 0;
}
