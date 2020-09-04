#pragma once
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/remove.h>
#include <curand.h>
#include <GL/glew.h>
#include <SDL_opengl.h>
#include <cuda_gl_interop.h>

struct pos {

	GLint x, y, z;

};

struct col {

	GLubyte r, g, b, a;

};

struct phy {

	GLdouble s;

	GLdouble d;

	GLdouble x;

	GLdouble y;

	GLdouble z;

};

struct ver {

	__device__ void operator=  (ver a) {
		this->p.x = a.p.x;
		this->p.y = a.p.y;
		this->p.z = a.p.z;
		this->p.d = a.p.d;
		this->p.s = a.p.s;
		this->c.r = a.c.r;
		this->c.g = a.c.g;
		this->c.b = a.c.b;
		this->c.a = a.c.a;
	}

	phy p;

	col c;

};