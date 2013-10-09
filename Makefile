include $(shell rospack find mk)/cmake.mk

test:
	nosetests -w test/

doc:
	sphinx-build -b html docs/ docs/_build
