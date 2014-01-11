FILES = driver.c glrasterizer.c glr.c glrglfw.c
CFLAGS = -g -Wall -std=c99

# Linux defaults
LDFLAGS = -lGLEW -lm
GLFWFLAGS = -lglfw
OPENGLFLAGS = -lGL -lGLU

# find out which OS we are on
UNAME := $(shell uname -s)

# OS X overrides
ifeq ($(UNAME),Darwin)
	GLFWFLAGS = -L /usr/local/Cellar/glfw3/3.0.3/lib/ -lglfw3 -framework IOKit -framework CoreVideo
	OPENGLFLAGS = -framework Cocoa -framework OpenGL
endif

driver:
	python shaders.py
	gcc $(CFLAGS) $(FILES) $(OPENGLFLAGS) $(GLFWFLAGS) $(LDFLAGS) -o driver.exe

clean:
	rm -f *.exe

