include $(shell rospack find mk)/cmake.mk

test:
	nosetests -w test/
