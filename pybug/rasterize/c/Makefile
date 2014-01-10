driver:
	python shaders.py
	gcc -g -Wall -std=c99 driver.c glrasterizer.c glr.c glrglfw.c -lm -lGLU -lGL -lglfw -lGLEW -o driver.exe

clean:
	rm -f *.exe
