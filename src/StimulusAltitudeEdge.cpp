/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */
#include "flyvr/StimulusInterface.hpp"

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

#include <jansson.h>

class StimulusAltitudeEdge: public StimulusInterface
{
public:
    StimulusAltitudeEdge();

    std::string name() const { return "StimulusAltitudeEdge"; }
    void post_init();

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

void StimulusAltitudeEdge::post_init() {
    osg::ref_ptr<osg::Node> drawn_geometry_node = load_osg_file("StimulusAltitudeEdge.osg");
    {
        osg::StateSet* state = drawn_geometry_node->getOrCreateStateSet();
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

        load_shader_source( AltitudeVertObj, "altitude.vert" );
        load_shader_source( AltitudeFragObj, "altitude.frag" );

        state->setAttributeAndModes(AltitudeProgram, osg::StateAttribute::ON);
        edge_height_uniform = new osg::Uniform( osg::Uniform::FLOAT, "edge_height" );
        state->addUniform( edge_height_uniform );
    }

  _group = new osg::Group;
  _group->addChild(drawn_geometry_node);
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
