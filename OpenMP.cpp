#include<iostream>
#include<fstream>
#include<chrono>
#include<stdio.h>
#include<string>
#include<stdlib.h>
#include<vector>
#include<omp.h>
#include<cstdlib>

#define _USE_MATH_DEFINES
#include<math.h>

#include "lodepng.h"

using namespace std::chrono;


/*
* We solve the 2D heat equation d/dt T(x, y, t) = kappa/(C rho) (d^2/dx^2 + d^2/dy^2) T(x, y, t) with the method of finite differences on a grid.
* Specifically, the following numerical approximations are made:
* - Forward-difference approximation for the time derivative:
*   d/dt T(x, y, t) ~ (T(x, y, t + dt) - T(x, y, t))/dt
* - Central-difference approximation for the space derivative:
*   (d^2/dx^2 + d^2/dy^2) T(x, y, t) ~ (T(x + dx, y, t) - 2T(x, y, t) + T(x - dx, y, t))/(dx)^2 + (T(x, y + dy, t) - 2T(x, y, t) + T(x, y - dy, t))/(dy)^2
* Boundary conditions are given on the (spatial) boundary of the grid and will be fixed in time.
* Define x = i dx, y = j dy, t = n dt. By inserting the above approximations into the PDE we get the following rule for updating T:
* T_{i,j}^{n+1} = T_{i,j}^n + dt * kappa/(C rho) * ((T^n_{i-1,j} - 2T^n_{i,j} + T^n_{i+1,j})/(dx)^2 + (T^n_{i,j-1} - 2T^n_{i,j} + T^n_{i,j+1})/(dy)^2)
*/

int getIndex(int i, int j, int cols){
	return i * cols + j;
}

int main(void) {
	const int threads=8;
	omp_set_num_threads(threads);

	// Measuring the time.
	auto start = high_resolution_clock::now();
	// Defining/Initializing variables.
	// size_xy: numbers of subdivisions of grid in x and y direction (size_x = size_y for easier display of images). n_steps: number of time steps.
	// eta = kappa/(C rho): Physical parameter.
	// dx, dy, dt: step sizes. We simulate a square grid of side length 1. Time step dt is dictated by numerical stability.
	// T_old, T_new: old and updated temperatures.
	const int size_xy = 2048;
	const int n_steps = 10001;
	const float eta = 1.0;
	const float dx = 1.0 / size_xy;
	const float dt = (dx * dx * dx * dx) / (2 * eta * (dx * dx + dx * dx));
	const float prefactor = dt * eta/((dx * dx));
	std::vector<unsigned char> image(size_xy * size_xy * 4);
	int i, j;
	unsigned int k;
	
	float* T_old = new float [size_xy * size_xy];
	float* T_new = new float [size_xy * size_xy];
	
	memset(T_old, 0.0, size_xy * size_xy * 4);

	printf("  Number of processors available = %d\n", omp_get_num_procs());
	printf("  Number of threads =              %d\n", omp_get_max_threads());

	// Enforce interesting fixed boundary condition: 
	// Temperature on the axes y = 0 and y = 1 vary with sin^2 and cos^2 respectively.

	int k_left, k_right, k_top, k_bottom;
	for(i = 0; i < size_xy; i++){
		k_left = i * size_xy;
		k_right = i * size_xy + size_xy - 1;
		k_top = i;
		k_bottom = (size_xy - 1)*size_xy + i;
		T_old[k_left] = cos(i * M_PI / float(size_xy)) * cos(i * M_PI / float(size_xy));
		T_old[k_right] = sin(i * M_PI / float(size_xy)) * sin(i * M_PI / float(size_xy));
		T_old[k_top] = 1 - i/float(size_xy);
		T_old[k_bottom] = 1 - i/float(size_xy);
	}

	memcpy(T_new,T_old,size_xy * size_xy * 4);

	// Solving the problem.
	for (int n = 0; n < n_steps; n++) {
		// Evolution step:

#pragma omp parallel for private(i, j)
		for (i = 1; i < size_xy - 1; i++){ 
			for (j = 1; j < size_xy - 1; j++){
				int k = i * size_xy + j; 
				int kim1 = (i - 1) * size_xy + j; 
				int kip1 = (i + 1) * size_xy + j;
				int kjm1 = i * size_xy + j - 1;
				int kjp1 = i * size_xy + j + 1;
				T_new[k] = T_old[k] + prefactor*(T_old[kim1] + T_old[kip1] + T_old[kjm1] + T_old[kjp1] - 4.0 * T_old[k]);
			}
		}


		// Updating T_old to be T_new.
		float* temp_pointer = T_old;
		T_old = T_new;
		T_new = temp_pointer;
		
		// Writing to png using lodepng (https://lodev.org/lodepng/)
		if (n % 10000 == 0) {
			char filename[100] = "";
			sprintf(filename, "pics/heat_%05d.png", n);
#pragma omp parallel for private(k)
			for (k = 0; k < size_xy * size_xy; k++) {
				image[k * 4] = static_cast<unsigned char>(T_new[k] * 255);
				image[k * 4 + 1] = 0;
				image[k * 4 + 2] = 0;
				image[k * 4 + 3] = 255;
			}
			unsigned error = lodepng::encode(filename, image, size_xy, size_xy);
			if (error) std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
		}
	}

	auto stop = high_resolution_clock::now();
	auto runtime = duration_cast<seconds>(stop - start);
	std::cout << "Total runtime: " << runtime.count() << std::endl;
	
	return 0;
}
