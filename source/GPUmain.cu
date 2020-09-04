#include "GPUmain.cuh"

__device__ float DIST(float xA, float yA, float xB, float yB) {
	return sqrt((float)((float)(xA - xB) * (float)(xA - xB)) + ((float)(yA - yB) * (float)(yA - yB)));
}

__device__ float DIR(float xA, float yA, float xB, float yB) {

	return atan2f(yA - yB, xA - xB);
}

__device__ float TO_RAD(float x) {
	const double pi = 3.141592653589793238;
	return (float)((double)x * (double)(pi / (double)180.0));
}

__device__ float TO_DEG(float x) {
	const double pi = 3.141592653589793238;
	return (float)((double)x * (double)((double)180.0 / pi));
}

float GPU::TO_RAD_h(float x) {
	const double pi = 3.141592653589793238;
	return (float)((double)x * (double)(pi / (double)180.0));
}

float GPU::TO_DEG_h(float x) {
	const double pi = 3.141592653589793238;
	return (float)((double)x * (double)((double)180.0 / pi));
}

__device__ float NORMALIZE(float AA, float AB, float n, float BA, float BB) {
	
	float Amin = ((abs(AA) > abs(AB)) ? abs(AA) : abs(AB)), Amax = ((abs(AA) > abs(AB)) ? abs(AA): abs(AB));
	float Bmin = ((abs(BA) > abs(BB)) ? abs(BA) : abs(BB)), Bmax = ((abs(BA) > abs(BB)) ? abs(BA) : abs(BB));
	float num;
	num = (abs(n) > Amax) ? Amax : abs(n);
	num = (abs(n) < Amin) ? Amin : abs(n);

	if ((Amin == Amax) || (Bmin == Bmax)) {
		return 0;
	}
	else if (num == Amin) {
		return Bmin;
	}
	else if (num == Amax) {
		return Bmax;
	}
	return Bmin + (((num - Amin) / (Amax - Amin)) * (Bmax - Bmin));
}

__device__ double NORMALIZE(double AA, double AB, double n, double BA, double BB) {

	double Amin = ((abs(AA) > abs(AB)) ? abs(AA) : abs(AB));
	double Amax = ((abs(AA) > abs(AB)) ? abs(AA) : abs(AB));
	double Bmin = ((abs(BA) > abs(BB)) ? abs(BA) : abs(BB));
	double Bmax = ((abs(BA) > abs(BB)) ? abs(BA) : abs(BB));
	double num;
	num = (abs(n) > Amax) ? Amax : abs(n);
	num = (abs(n) < Amin) ? Amin : abs(n);
	return (n > 0) ? (Bmin + (((num - Amin) / (Amax - Amin)) * (Bmax - Bmin))):(Bmin + (((num - Amin) / (Amax - Amin)) * (Bmax - Bmin)));
}

//GRADIANT: color A, color B, double n (0.0 = color A, 1.0 = color B)

__device__ col GRADIENT(col A, col B, double n) {
	
	return col{
		(GLubyte)NORMALIZE(0.0,1.0,n,(double)A.r,(double)B.r),
		(GLubyte)NORMALIZE(0.0,1.0,n,(double)A.g,(double)B.g), 
		(GLubyte)NORMALIZE(0.0,1.0,n,(double)A.b,(double)B.b), 
		(GLubyte)NORMALIZE(0.0,1.0,n,(double)A.a,(double)B.a) 
	};
}

__device__ col SHADE(col A, double n) {
	if (n < 0.5) {
		return GRADIENT(col{ 0,0,0,255 }, A, n * 2.0);
	} else 
	if (n > 0.5){
		return GRADIENT(A, col{ 255,255,255,255 }, (n * 2.0) - 1.0);
	}
	return A;
}

__global__ void uploadVerts(ver *ve, pos *po, col *co, float *d) {
	int id = threadIdx.x + (blockDim.x * blockIdx.x);
	ver v = ve[id];
	po[id].x = (GLint)v.p.x;
	po[id].y = (GLint)v.p.y;
	po[id].z = (GLint)v.p.z;
	co[id].r = (GLint)v.c.r;
	co[id].g = (GLint)v.c.g;
	co[id].b = (GLint)v.c.b;
	co[id].a = (GLint)v.c.a;
}

__global__ void genGrid(ver *v) {
	int i = threadIdx.x + (blockDim.x * blockIdx.x);
	GLdouble x = (double)(i % 1920);
	GLdouble y = (double)i / 1920.0;

	v[i].p.x = x;
	v[i].p.y = y;
	v[i].p.z = 0.0;

	v[i].c.r = 255;
	v[i].c.g = 127;
	v[i].c.b = 0;
	v[i].c.a = 255;
}

__global__ void physics(ver *pix, float *in, uint16_t *d) {

	int i = threadIdx.x + (blockDim.x * blockIdx.x);
	ver v = pix[i];
	
	if (in[0]) {

		v.p.s += 0.00001 * DIST(in[2], in[3], v.p.x, v.p.y);
		

	}
	
	v.p.d = TO_DEG(DIR(in[2], in[3], v.p.x, v.p.y)) - 90.0;

	v.p.s -= (v.p.s / 2000);

	v.p.x += (cos(TO_RAD(v.p.d)) * v.p.s);
	v.p.y += (sin(TO_RAD(v.p.d)) * v.p.s);

	v.c.r = (GLubyte)(127.5 * (1.0 + (cos(5.0 * (v.p.s + ((2.0 / 3.0) * 3.141592653589793238))))));
	v.c.g = (GLubyte)(127.5 * (1.0 + (cos(5.0 * (v.p.s - ((2.0 / 3.0) * 3.141592653589793238))))));
	v.c.b = (GLubyte)(127.5 * (1.0 + (cos(5.0 * v.p.s))));

	v.p.x = (v.p.x <= 0) ? v.p.x + d[0] : v.p.x;
	v.p.x = (v.p.x >= d[0]) ? v.p.x - d[0] : v.p.x;
	v.p.y = (v.p.y <= 0) ? v.p.y + d[1] : v.p.y;
	v.p.y = (v.p.y >= d[1]) ? v.p.y - d[1] : v.p.y;

	pix[i] = v;

}

GLuint GPU::width;

GLuint GPU::height;

GLint GPU::camerax;

GLint GPU::cameray;

int GPU::nParticlesC;

int GPU::nParticlesN;

GLuint GPU::vboidP;

GLuint GPU::vboidC;

GLuint GPU::vaoid;

GLuint GPU::vrshaderid;

GLuint GPU::frshaderid;

GLuint GPU::lkshaderid;

cudaGraphicsResource *GPU::CGRp;

cudaGraphicsResource *GPU::CGRc;

const char* GPU::shaders[2] = {
R"(
#version 460

layout(location = 0) in ivec3 vertex_position;
layout(location = 1) in ivec4 vertex_colour;

layout(location = 7) uniform mat4 prj_matrix;

out vec4 colour;

out vec4 position;

void main() {
    colour = vec4(vertex_colour) / 255.0;
    gl_Position = prj_matrix * vec4(vertex_position, 1.0);;
}
)"
,
R"(
#version 460

in vec4 colour;

out vec4 frag_colour;

void main() {
   frag_colour = colour;
}
)"
};

thrust::device_vector<uint16_t> GPU::dim;

thrust::device_vector<float> GPU::in;

//collection of vertices to be simulated and rendered
thrust::device_vector<ver> GPU::verts;

void GPU::init(int w, int h)
{
	width = w;
	height = h;
	nParticlesC = w * h;
	dim.resize(2);
	dim[0] = w;
	dim[1] = h;
	in.resize(10, 0);
	verts.resize(nParticlesC, ver{ phy{0,0,0,0,0}, col{255,0,0,255} });
	genGrid<<<nParticlesC/1024,1024>>>(thrust::raw_pointer_cast(&verts[0]));
	cudaDeviceSynchronize();

	vrshaderid = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vrshaderid, 1, &shaders[0], NULL);
	glCompileShader(vrshaderid);
	GLint success;
	GLchar infoLog[512];
	glGetShaderiv(vrshaderid, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(vrshaderid, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
	}
	frshaderid = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(frshaderid, 1, &shaders[1], NULL);
	glCompileShader(frshaderid);
	glGetShaderiv(frshaderid, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(frshaderid, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
	}

	lkshaderid = glCreateProgram();
	glAttachShader(lkshaderid, vrshaderid);
	glAttachShader(lkshaderid, frshaderid);
	glLinkProgram(lkshaderid);
	glGetProgramiv(lkshaderid, GL_LINK_STATUS, &success);
	if (!success) {
		glGetProgramInfoLog(lkshaderid, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
	}
	glDeleteShader(vrshaderid);
	glDeleteShader(frshaderid);

	glGenVertexArrays(1, &vaoid);
	glGenBuffers(1,&vboidP);
	glGenBuffers(1, &vboidC);
	
	glBindVertexArray(vaoid);

	glBindBuffer(GL_ARRAY_BUFFER, vboidP);
	glBufferData(GL_ARRAY_BUFFER, nParticlesC * sizeof(pos), 0, GL_DYNAMIC_DRAW);
	glVertexAttribIPointer(0, 3, GL_INT, sizeof(pos), NULL);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindBuffer(GL_ARRAY_BUFFER, vboidC);
	glBufferData(GL_ARRAY_BUFFER, nParticlesC * sizeof(col), 0, GL_DYNAMIC_DRAW);
	glVertexAttribIPointer(1, 4, GL_UNSIGNED_BYTE, sizeof(col), NULL);
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	cudaGraphicsGLRegisterBuffer(&CGRp, vboidP, cudaGraphicsMapFlagsWriteDiscard);
	cudaGraphicsGLRegisterBuffer(&CGRc, vboidC, cudaGraphicsMapFlagsWriteDiscard);

	glBindVertexArray(0);
	
}

void GPU::input(uint8_t mousel, uint8_t mouser, int16_t mousex, int16_t mousey)
{
															//buttons:
	in[0] = (float)mousel;									//ml = 0
	in[1] = (float)mouser;									//mr = 1
	in[6] = in[2];											//mpx
	in[7] = in[3];											//mpy
	in[2] = (float)mousex;									//mx
	in[3] = (float)mousey;									//my
	in[4] = in[2] - in[6];									//mdx
	in[5] = in[3] - in[7];									//mdy

	
	//in[0] = (buttons[8] == 0) ? 0.0f : 1.0f;				//w = 2
	//in[0] = (buttons[9] == 0) ? 0.0f : 1.0f;				//s = 3
	//in[0] = (buttons[10] == 0) ? 0.0f : 1.0f;				//a = 4
	//in[0] = (buttons[12] == 0) ? 0.0f : 1.0f;				//d = 5
	//in[0] = (buttons[13] == 0) ? 0.0f : 1.0f;				//q = 6
	//in[0] = (buttons[14] == 0) ? 0.0f : 1.0f;				//e = 7
	//in[0] = (buttons[15] == 0) ? 0.0f : 1.0f;				//0 = 8
	//in[0] = (buttons[16] == 0) ? 0.0f : 1.0f;				//1 = 9
	//in[0] = (buttons[17] == 0) ? 0.0f : 1.0f;				//2 = 10
	//in[0] = (buttons[18] == 0) ? 0.0f : 1.0f;				//3 = 11
	//in[0] = (buttons[19] == 0) ? 0.0f : 1.0f;				//4 = 12
	//in[0] = (buttons[20] == 0) ? 0.0f : 1.0f;				//5 = 13
	//in[0] = (buttons[21] == 0) ? 0.0f : 1.0f;				//6 = 14
	//in[0] = (buttons[22] == 0) ? 0.0f : 1.0f;				//7 = 15
	//in[0] = (buttons[23] == 0) ? 0.0f : 1.0f;				//8 = 16
	//in[0] = (buttons[24] == 0) ? 0.0f : 1.0f;				//9 = 17

	
	
}

void GPU::compute()
{
	
	physics<<<nParticlesC/1024, 1024>>>(thrust::raw_pointer_cast(&verts[0]), thrust::raw_pointer_cast(&in[0]), thrust::raw_pointer_cast(&dim[0]));
	cudaDeviceSynchronize();
}

void GPU::render()
{

	pos *posi;
	col *cols;

	size_t sizep;
	size_t sizec;

	cudaGraphicsMapResources(1, &CGRp, 0);
	cudaGraphicsMapResources(1, &CGRc, 0);

	cudaGraphicsResourceGetMappedPointer((void**)&posi, &sizep, CGRp);
	cudaGraphicsResourceGetMappedPointer((void**)&cols, &sizec, CGRc);

	uploadVerts<<<nParticlesC/1024, 1024>>>(thrust::raw_pointer_cast(&verts[0]), posi, cols, thrust::raw_pointer_cast(&in[0]));
	cudaDeviceSynchronize();

	cudaGraphicsUnmapResources(1, &CGRp, 0);
	cudaGraphicsUnmapResources(1, &CGRc, 0);

	glBindVertexArray(vaoid);

	glDrawArrays(GL_POINTS,0,nParticlesC);
	
	glBindVertexArray(0);
}

void GPU::GPUmain()
{

	compute();

	render();

}

void GPU::free()
{
	cudaGraphicsUnregisterResource(CGRp);
	cudaGraphicsUnregisterResource(CGRc);
	glDeleteVertexArrays(1,&vaoid);
	glDeleteBuffers(1, &vboidP);
	glDeleteBuffers(1, &vboidC);
	verts.clear();
	thrust::device_vector<ver>().swap(verts);
}