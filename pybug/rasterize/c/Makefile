driver:
	python shaders.py
	gcc -g -Wall -std=c99 driver.c glrasterizer.c glrglut.c glr.c -lm -lGLU -lGL -lglut -lGLEW -o driver.exe

clean:
	rm -f *.exe
