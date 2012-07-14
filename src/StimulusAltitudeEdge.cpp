/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- */
#include "vros_display/stimulus_interface.h"

#include "Poco/ClassLibrary.h"

#include <iostream>

#include <osg/MatrixTransform>
#include <osg/ShapeDrawable>
#include <osg/Material>
#include <osg/CullFace>
#include <osg/Light>
#include <osg/LightSource>
#include <osg/LightModel>

#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgDB/FileUtils>

#include <boost/filesystem.hpp>

#include <jansson.h>

std::string join_path(std::string a,std::string b);
void LoadShaderSource( osg::Shader* shader, const std::string& fileName );
std::string join_path(std::string a,std::string b) {
	// roughly inspired by Python's os.path.join
	char pathsep = '/'; // TODO: FIXME: not OK on Windows.
	if (a.at(a.size()-1)==pathsep) {
		return a+b;
	} else {
		return a+std::string("/")+b;
	}
}

// load source from a file.
void LoadShaderSource( osg::Shader* shader, const std::string& fileName )
{
    std::string fqFileName = osgDB::findDataFile(fileName);
    if( fqFileName.length() != 0 )
    {
        shader->loadShaderSourceFromFile( fqFileName.c_str() );
    }
    else
    {
		std::stringstream ss;
		ss << "File \"" << fileName << "\" not found.";
		throw std::ios_base::failure(ss.str());
    }
}


class StimulusAltitudeEdge: public StimulusInterface
{
public:
	StimulusAltitudeEdge();

	std::string name() const { return "StimulusAltitudeEdge"; }
	void post_init(std::string config_data_dir);

	osg::ref_ptr<osg::Group> get_3d_world() {return _group; }

	std::vector<std::string> get_topic_names() const;
	void receive_json_message(const std::string& topic_name, const std::string& json_message);
	std::string get_message_type(const std::string& topic_name) const;

private:
	osg::ref_ptr<osg::Group> _group;
	float _edge_height;
	osg::Uniform* edge_height_uniform;
};


StimulusAltitudeEdge::StimulusAltitudeEdge() : _edge_height(0.5) {
}

void StimulusAltitudeEdge::post_init(std::string config_data_dir) {

	osg::ref_ptr<osg::Geode> geode = new osg::Geode;
	{
		osg::ref_ptr<osg::TessellationHints> hints = new osg::TessellationHints;
		hints->setDetailRatio(2.0f);
		osg::ref_ptr<osg::ShapeDrawable> shape;

		osg::ref_ptr<osg::Shape> cyl = new osg::Cylinder(
												  osg::Vec3(0.0f, 0.0f, 0.0f), // center
												  0.5, //radius
												  1.0); //height
		shape = new osg::ShapeDrawable( cyl, hints.get());
		geode->addDrawable(shape.get());

		osg::StateSet* state = geode->getOrCreateStateSet();
		state->setMode(GL_LIGHTING,osg::StateAttribute::OFF);

		osg::Program* AltitudeProgram;
		osg::Shader*  AltitudeVertObj;
		osg::Shader*  AltitudeFragObj;

		AltitudeProgram = new osg::Program;
		AltitudeProgram->setName( "altitude" );
		AltitudeVertObj = new osg::Shader( osg::Shader::VERTEX );
		AltitudeFragObj = new osg::Shader( osg::Shader::FRAGMENT );
		AltitudeProgram->addShader( AltitudeFragObj );
		AltitudeProgram->addShader( AltitudeVertObj );

		std::string shader_dir = join_path( config_data_dir, "src/shaders" );
		LoadShaderSource( AltitudeVertObj, join_path(shader_dir,"altitude.vert" ));
		LoadShaderSource( AltitudeFragObj, join_path(shader_dir, "altitude.frag" ));

		state->setAttributeAndModes(AltitudeProgram, osg::StateAttribute::ON);
		edge_height_uniform = new osg::Uniform( osg::Uniform::FLOAT, "edge_height" );
		state->addUniform( edge_height_uniform );
	}

  _group = new osg::Group;
  _group->addChild(geode);
  _group->setName("StimulusAltitudeEdge._group");
}

std::vector<std::string> StimulusAltitudeEdge::get_topic_names() const {
	std::vector<std::string> result;
	result.push_back("edge_height");
	return result;
}

void StimulusAltitudeEdge::receive_json_message(const std::string& topic_name,
                                                const std::string& json_message) {
	json_t *root;
	json_error_t error;

	root = json_loads(json_message.c_str(), 0, &error);
	if(!root) {
		throw std::runtime_error("error in json");
	}

	json_t *data_json = json_object_get(root, "data");
	if (data_json==NULL) {
		throw std::runtime_error("key not in JSON");
	}
	if(!json_is_number(data_json)){
		throw std::runtime_error("error in json");
	}
	_edge_height = json_number_value( data_json );
	edge_height_uniform->set(_edge_height);

}

std::string StimulusAltitudeEdge::get_message_type(const std::string& topic_name) const {
	std::string result;

	if (topic_name=="edge_height") {
		result = "std_msgs/Float32";
	} else {
		throw std::runtime_error("unknown topic name");
	}
	return result;
}

POCO_BEGIN_MANIFEST(StimulusInterface)
POCO_EXPORT_CLASS(StimulusAltitudeEdge)
POCO_END_MANIFEST

void pocoInitializeLibrary()
{
}

void pocoUninitializeLibrary()
{
}
