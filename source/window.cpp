#include "window.h"

std::fstream Window::log;

std::map<std::string, bool> Window::keys;

std::array<bool, 2> Window::mouseButtons;

std::array<int, 2> Window::mouseLoc;

Window::Window(const std::string &n, int w, int h) :
	name(n), width(w), height(h)
{
	//open log
	log.open("Debug.log", std::fstream::out | std::ofstream::trunc);

	//set window to closed to exit while loop if initialization failed.
	closed = !init();
}

Window::~Window()
{

	GPU::free();

	//destroy log
	log.close();

	//remove opengl from window
	SDL_GL_DeleteContext(glcontext);

	//destroy window
	SDL_DestroyWindow(window);

	//quit SDL
	SDL_Quit();

}

bool Window::init()
{
	//initialize SDL
	if (SDL_Init(SDL_INIT_EVERYTHING) != 0) {

		log << "Failed to initialize SDL!\n";
		return false;

	}

	//set window atributes
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 6);

	SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);

	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);



	//create window
	window = SDL_CreateWindow(
		name.c_str(),
		SDL_WINDOWPOS_CENTERED,
		SDL_WINDOWPOS_CENTERED,
		width,
		height,
		SDL_WINDOW_OPENGL

	);

	//create opengl context in the window
	glcontext = SDL_GL_CreateContext(window);

	SDL_GL_SetSwapInterval(1);

	//check if the window was created
	if (window == nullptr) {

		log << "Failed to create window!\n";
		return false;

	}

	//turn on experimental features
	glewExperimental = GL_TRUE;

	//initiallize glew
	if (glewInit() != GLEW_OK) {

		log << "Failed to Init GLEW";

		return false;

	}
	
	

	//set drawing parameters
	glViewport(0, 0, width, height);

	glPointSize(1);
	glEnable(GL_BLEND);                                // Allow Transparency
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  // how transparency acts
	
	GPU::init(width, height);

	

	return true;
}

void Window::getEvents()
{
	//refresh events
	SDL_PumpEvents();

	//get events
	SDL_Event event;

	//get keyboard state
	keyState = SDL_GetKeyboardState(NULL);

	//check for quit
	if (SDL_PollEvent(&event)) {

		switch (event.type) {

		case SDL_QUIT:

			closed = true;
			break;

		default:
			break;
		}
	}

	//set mouse x and y
	SDL_GetMouseState(&mouseLoc[0], &mouseLoc[1]);

	//check for mouse press
	mouseButtons[0] = (SDL_GetMouseState(NULL, NULL) & SDL_BUTTON(SDL_BUTTON_LEFT)) ? true : false;
	mouseButtons[1] = (SDL_GetMouseState(NULL, NULL) & SDL_BUTTON(SDL_BUTTON_RIGHT)) ? true : false;

	//mouseLoc[1] = height - mouseLoc[1];

	//check for keypress
	keys["w"] = (keyState[SDL_SCANCODE_W]) ? true : false;
	keys["s"] = (keyState[SDL_SCANCODE_S]) ? true : false;
	keys["a"] = (keyState[SDL_SCANCODE_A]) ? true : false;
	keys["d"] = (keyState[SDL_SCANCODE_D]) ? true : false;
	keys["q"] = (keyState[SDL_SCANCODE_Q]) ? true : false;
	keys["e"] = (keyState[SDL_SCANCODE_E]) ? true : false;

	keys["1"] = (keyState[SDL_SCANCODE_1]) ? true : false;
	keys["2"] = (keyState[SDL_SCANCODE_2]) ? true : false;
	keys["3"] = (keyState[SDL_SCANCODE_3]) ? true : false;
	keys["4"] = (keyState[SDL_SCANCODE_4]) ? true : false;
	keys["5"] = (keyState[SDL_SCANCODE_5]) ? true : false;
	keys["6"] = (keyState[SDL_SCANCODE_6]) ? true : false;
	keys["7"] = (keyState[SDL_SCANCODE_7]) ? true : false;
	keys["8"] = (keyState[SDL_SCANCODE_8]) ? true : false;
	keys["9"] = (keyState[SDL_SCANCODE_9]) ? true : false;
	keys["0"] = (keyState[SDL_SCANCODE_0]) ? true : false;

}

void Window::renderFrame()
{

	GPU::input(mouseButtons[0],mouseButtons[1], mouseLoc[0], height - mouseLoc[1]);

	GPU::compute();

	glClearColor(0, 0, 0, 0); // we clear the screen with black (else, frames would overlay...)
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the buffer

	glUseProgram(GPU::lkshaderid);

	glm::mat4 prj = glm::ortho(0.0f, (float)width, 0.0f, (float)height, -255.0f, 0.0f);
	//glm::mat4 prj = glm::(0.0f, (float)width, 0.0f, (float)height, -255.0f, 0.0f);
	glUniformMatrix4fv(7, 1, GL_FALSE, glm::value_ptr(prj));

	GPU::render();

	SDL_GL_SwapWindow(window); //swap buffers
}