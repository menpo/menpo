execute:
	python setup.py build_ext --inplace

clean:
	find . -name "*.so" -delete
