#include "window.h"

int main(int argc, char *argv[]) {

	Window window("Destruction", 1920, 1080);

	while (!window.isClosed()) {

		window.getEvents();
		window.renderFrame();
		Window::log.flush();

	}

	return 0;
}