driver:
	python shaders.py
	gcc -g -Wall -std=c99 driver.c glrasterizer.c glrglut.c glr.c -L/usr/local/lib/ -lm -lGLU -lGL -lglfw -lGLEW -o driver.exe

clean:
	rm -f *.exe
