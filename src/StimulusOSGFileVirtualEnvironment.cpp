/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */
#include "vros_display/stimulus_interface.h"

#include "Poco/ClassLibrary.h"

#include <iostream>

#include <osg/MatrixTransform>
#include <osg/TextureCubeMap>
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

std::string join_path(std::string a,std::string b) {
    // roughly inspired by Python's os.path.join
    char pathsep = '/'; // TODO: FIXME: not OK on Windows.
    if (a.at(a.size()-1)==pathsep) {
        return a+b;
    } else {
        return a+std::string("/")+b;
    }
}

class StimulusOSGFileVirtualEnvironment: public StimulusInterface
{
public:
std::string name() const {
    return "StimulusOSGFileVirtualEnvironment";
}

void _load_stimulus_filename( std::string osg_filename ) {


    if (!top) {
        std::cerr << "top node not defined!?" << std::endl;
        return;
    }

    // don't show the old switching node.
    top->removeChild(switch_node);

    // (rely on C++ to delete the old switching node).

    // (create a new switching node.
    switch_node = new osg::MatrixTransform;

    // now load it with new contents
    std::cerr << "reading .osg file: " << osg_filename << std::endl;
    osg::Node* tmp = osgDB::readNodeFile(osg_filename);
    if (tmp!=NULL) {
        switch_node->addChild( tmp );
    } else {
        throw std::runtime_error("could not read figure model");
    }

    top->addChild(switch_node);
}

void _load_skybox_basename( std::string basename ) {

    if (!top) {
        std::cerr << "top node not defined!?" << std::endl;
        return;
    }

    if (skybox_node.valid()) {
        top->removeChild(skybox_node);
        skybox_node = NULL; // dereference the previous node
    }

    if (basename=="<none>") {
        return;
    }

    osg::ref_ptr<osg::MatrixTransform> mt = new osg::MatrixTransform;

    if (basename=="<default>") {
        add_default_skybox( mt );
    } else {
        std::string extension = ".png";
        try {
            add_skybox(mt, basename, extension);
        } catch (std::runtime_error) {
            extension = ".jpg";
            add_skybox(mt, basename, extension);
        }

    }

    skybox_node = mt;
    top->addChild(skybox_node);
}


virtual void post_init(std::string config_data_dir) {
  top = new osg::MatrixTransform; top->addDescription("virtual world top node");

  switch_node = new osg::MatrixTransform;
  top->addChild(switch_node);

  std::string osg_filename = join_path(config_data_dir,"osgfile.osg");
  _load_stimulus_filename( osg_filename );
  _virtual_world = top;

}

osg::Vec4 get_clear_color() const {
    return osg::Vec4(0.5, 0.5, 0.5, 1.0); // mid gray
}

void resized(int width,int height) {
}

osg::ref_ptr<osg::Group> get_3d_world() {
    return _virtual_world;
}

osg::ref_ptr<osg::Group> get_2d_hud() {
    return 0;
}

std::vector<std::string> get_topic_names() const {
    std::vector<std::string> result;
    result.push_back("stimulus_filename");
    result.push_back("skybox_basename");
    return result;
}

void receive_json_message(const std::string& topic_name, const std::string& json_message) {
    json_t *root;
    json_error_t error;

    root = json_loads(json_message.c_str(), 0, &error);
    if(!root) {
        fprintf(stderr, "error: in %s(%d) on json line %d: %s\n", __FILE__, __LINE__, error.line, error.text);
        throw std::runtime_error("error in json");
    }

    if (topic_name=="stimulus_filename") {

        json_t *data_json;

        data_json = json_object_get(root, "data");
        if(!json_is_string(data_json)){
            fprintf(stderr, "error: in %s(%d): expected string\n", __FILE__, __LINE__);
            throw std::runtime_error("error in json");
        }
        std::string stimulus_filename = json_string_value( data_json );
        std::cerr << "loading filename: " << stimulus_filename << std::endl;
        _load_stimulus_filename( stimulus_filename );
    } else if (topic_name=="skybox_basename") {

        json_t *data_json;

        data_json = json_object_get(root, "data");
        if(!json_is_string(data_json)){
            fprintf(stderr, "error: in %s(%d): expected string\n", __FILE__, __LINE__);
            throw std::runtime_error("error in json");
        }
        std::string skybox_basename = json_string_value( data_json );
        std::cerr << "loading skybox filename: " << skybox_basename << std::endl;
        _load_skybox_basename( skybox_basename );
    } else {
        fprintf(stderr, "error: in %s(%d): unknown topic\n", __FILE__, __LINE__);
        throw std::runtime_error("error in json");
    }
}

std::string get_message_type(const std::string& topic_name) const {
    std::string result;

    if (topic_name=="stimulus_filename") {
        result = "std_msgs/String";
    } else if (topic_name=="skybox_basename") {
        result = "std_msgs/String";
    } else {
        throw std::runtime_error("unknown topic name");
    }
    return result;

}

private:
    osg::ref_ptr<osg::Group> _virtual_world;
    osg::ref_ptr<osg::MatrixTransform> top;
    osg::ref_ptr<osg::MatrixTransform> switch_node;
    osg::ref_ptr<osg::MatrixTransform> skybox_node;
};


POCO_BEGIN_MANIFEST(StimulusInterface)
POCO_EXPORT_CLASS(StimulusOSGFileVirtualEnvironment)
POCO_END_MANIFEST

void pocoInitializeLibrary()
{
}

void pocoUninitializeLibrary()
{
}
