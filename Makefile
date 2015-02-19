include $(shell rospack find mk)/cmake.mk

test:
	NOSETEST_FLAG=1 nosetests -w test/

test-parallel:
	NOSETEST_FLAG=1 nosetests -s --processes=4 --process-restartworker --process-timeout=1000 -w test/


doc:
	sphinx-build -b html docs/ docs/_build
