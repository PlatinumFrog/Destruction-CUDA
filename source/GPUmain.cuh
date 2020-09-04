#pragma once
#include "GPUtypes.cuh"

#define BUFFER_OFFSET(i) ((char *)NULL + (i))

class GPU {

public:

	static GLuint width;

	static GLuint height;

	static GLint camerax;

	static GLint cameray;

	static int nParticlesC;

	static int nParticlesN;

	static GLuint vboidD;

	static GLuint vboidP;

	static GLuint vboidC;

	static GLuint vaoid;

	static GLuint vrshaderid;

	static GLuint frshaderid;

	static GLuint lkshaderid;

	static cudaGraphicsResource *CGRp;

	static cudaGraphicsResource *CGRc;

	static const char* shaders[2];

	static thrust::device_vector<GLdouble> depthB;

	static thrust::device_vector<uint16_t> dim;

	static thrust::device_vector<float> in;

	static thrust::device_vector<ver> verts;

	static float TO_RAD_h(float x);

	static float TO_DEG_h(float x);

	static void init(int w, int h);

	static void input(uint8_t mousel, uint8_t mouser, int16_t mousex, int16_t mousey);

	static void compute();

	static void render();

	static void GPUmain();

	static void free();

};
