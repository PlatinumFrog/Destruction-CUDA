#pragma once
#include <SDL.h>

#include <GL/glew.h>
#include <SDL_opengl.h>

#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <array>

#include "GPUmain.cuh"

#include <glm.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>

class Window {

public:

	Window(const std::string &n, int w, int h);

	~Window();

	inline bool isClosed() const { return closed; };

	void getEvents();

	void renderFrame();

	static std::fstream log;

	static std::map<std::string, bool> keys;

	static std::array<bool, 2> mouseButtons;

	static std::array<int, 2> mouseLoc;

private:

	bool init();

private:

	int width;

	int height;

	std::string name;

	bool closed = false;

	SDL_Window *window = nullptr;

	SDL_GLContext glcontext;

	const Uint8 *keyState;

};