driver:
	python shaders.py
	gcc -g -Wall -std=c99 driver.c glrasterizer.c glr.c glrglfw.c -lGLU -lGL -lglfw -lGLEW -lm -o driver.exe

clean:
	rm -f *.exe
