include $(shell rospack find mk)/cmake.mk

test:
	nosetests --with-coverage --cover-html --cover-package analysislib -w test/

doc:
	sphinx-build -b html docs/ docs/_build
