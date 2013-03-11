/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */
#include "flyvr/StimulusInterface.hpp"
#include "flyvr/flyvr_assert.h"

#include "json2osg.hpp"

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

#include <jansson.h>

// this function from http://stackoverflow.com/a/8098080
std::string string_format(const std::string &fmt, ...) {
    int size=100;
    std::string str;
    va_list ap;
    while (1) {
        str.resize(size);
        va_start(ap, fmt);
        int n = vsnprintf((char *)str.c_str(), size, fmt.c_str(), ap);
        va_end(ap);
        if (n > -1 && n < size) {
            str.resize(n);
            return str;
        }
        if (n > -1)
            size=n+1;
        else
            size*=2;
    }
}


std::string join_path(std::string a,std::string b) {
    // roughly inspired by Python's os.path.join
    char pathsep = '/'; // TODO: FIXME: not OK on Windows.
    if (a.at(a.size()-1)==pathsep) {
        return a+b;
    } else {
        return a+std::string("/")+b;
    }
}

class StimulusOSGFile: public StimulusInterface
{
public:
std::string name() const {
    return "StimulusOSGFile";
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
    switch_node = new osg::PositionAttitudeTransform;
    _update_pat();

    // now load it with new contents
    std::cerr << "reading .osg file: " << osg_filename << std::endl;

    osg::Node* tmp = load_osg_file(osg_filename);
    if (tmp!=NULL) {
        switch_node->addChild( tmp );
    } else {
        throw std::runtime_error(string_format("File %s not found",osg_filename.c_str()));
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

void _update_pat() {
    flyvr_assert(switch_node.valid());
    switch_node->setPosition( model_position );
    switch_node->setAttitude( model_attitude );
}

virtual void post_init(bool slave) {
  top = new osg::MatrixTransform; top->addDescription("virtual world top node");

  // when we first start, don't load any model, but create a node that is later deleted.
  switch_node = new osg::PositionAttitudeTransform;
  _update_pat();
  top->addChild(switch_node);

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
    result.push_back("model_pose");
    return result;
}

void receive_json_message(const std::string& topic_name, const std::string& json_message) {
    json_t *root;
    json_error_t error;

    root = json_loads(json_message.c_str(), 0, &error);
    if(!root) {
        throw std::runtime_error(string_format(
           "error: in %s(%d) on json line %d: %s\n", __FILE__, __LINE__,
           error.line, error.text));
    }

    if (topic_name=="stimulus_filename") {
        std::string stimulus_filename = parse_string(root);
        std::cerr << "loading filename: " << stimulus_filename << std::endl;
        _load_stimulus_filename( stimulus_filename );
    } else if (topic_name=="skybox_basename") {
        std::string skybox_basename = parse_string(root);
        std::cerr << "loading skybox filename: " << skybox_basename << std::endl;
        _load_skybox_basename( skybox_basename );
    } else if (topic_name=="model_pose") {
        json_t *data_json;

        data_json = json_object_get(root, "position");
        model_position = parse_vec3(data_json);

        data_json = json_object_get(root, "orientation");
        model_attitude = parse_quat(data_json);
        _update_pat();
    } else {
        throw std::runtime_error( string_format( "error: in %s(%d): unknown topic\n", __FILE__, __LINE__));
    }
}

std::string get_message_type(const std::string& topic_name) const {
    std::string result;

    if (topic_name=="stimulus_filename") {
        result = "std_msgs/String";
    } else if (topic_name=="skybox_basename") {
        result = "std_msgs/String";
    } else if (topic_name=="model_pose") {
        result = "geometry_msgs/Pose";
    } else {
        throw std::runtime_error(string_format("unknown topic name: %s",topic_name.c_str()));
    }
    return result;

}

private:
    osg::ref_ptr<osg::Group> _virtual_world;
    osg::ref_ptr<osg::MatrixTransform> top;
    osg::ref_ptr<osg::PositionAttitudeTransform> switch_node;
    osg::Vec3 model_position;
    osg::Quat model_attitude;
    osg::ref_ptr<osg::MatrixTransform> skybox_node;
};


POCO_BEGIN_MANIFEST(StimulusInterface)
POCO_EXPORT_CLASS(StimulusOSGFile)
POCO_END_MANIFEST

void pocoInitializeLibrary()
{
}

void pocoUninitializeLibrary()
{
}
