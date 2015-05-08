include $(shell rospack find mk)/cmake.mk

test:
	NOSETEST_FLAG=1 nosetests -w test/

test-parallel:
	NOSETEST_FLAG=1 nosetests --processes=6 --process-restartworker --process-timeout=1000 -w test/

test-cover:
	rm -rf ./cover
	NOSETEST_FLAG=1 nosetests --with-coverage --cover-erase --cover-package=analysislib --cover-html -I 'test_scripts\.py' -w test/

doc:
	sphinx-build -b html docs/ docs/_build
