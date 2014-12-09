include $(shell rospack find mk)/cmake.mk

test:
	NOSETEST_FLAG=1 nosetests -w test/

doc:
	sphinx-build -b html docs/ docs/_build
