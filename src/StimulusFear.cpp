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
#include <osg/Texture2D>

#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgDB/FileUtils>

#define SOLID_COLOR 0

class StimulusFear: public StimulusInterface
{
public:
std::string name() const {
	return "StimulusFear";
}

virtual void post_init(bool slave) {
    _L = 4;
    _bL = _L * 2;
	_virtual_world = create_virtual_world();
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
    result.push_back("fear_orientation_deg");
    result.push_back("fear_floor_a_height");
    result.push_back("fear_floor_b_height");
    result.push_back("fear_floor_a_image");
    result.push_back("fear_floor_b_image");
	return result;
}

void receive_json_message(const std::string& topic_name, const std::string& json_message) {
    float f;
    json_t *root;
    json_error_t error;

    root = json_loads(json_message.c_str(), 0, &error);
    flyvr_assert(root != NULL);

    /* most messages are float types */
    if ((topic_name=="fear_orientation_deg") || 
        (topic_name=="fear_floor_a_height")  || 
        (topic_name=="fear_floor_b_height")) {
        f = parse_float(root);
    }

    if (topic_name=="fear_orientation_deg") {
        osg::Quat q;
        q.makeRotate(osg::DegreesToRadians(f),osg::Vec3(0,0,1)); 
        _virtual_world->setAttitude(q);
    } else if (topic_name=="fear_floor_b_height") {
        _plane_z_b->setCenter(osg::Vec3(0.0f, 0.0f, f));
        _shape_b->dirtyDisplayList();
        _shape_b->dirtyBound();
    } else if (topic_name=="fear_floor_a_height") {
        _plane_z_a_xminus->setCenter(osg::Vec3(_L/-2.0, 0.0f, f));
        _shape_a->dirtyDisplayList();
        _shape_a->dirtyBound();
    } else if (topic_name=="fear_floor_a_image") {
        std::string s = parse_string(root);
        _texture_a->setImage(load_image_file(s));
    } else if (topic_name=="fear_floor_b_image") {
        std::string s = parse_string(root);
        _texture_b->setImage(load_image_file(s));
    }


    json_decref(root);
}

std::string get_message_type(const std::string& topic_name) const {
    std::string result;

    if ((topic_name=="fear_orientation_deg") || 
        (topic_name=="fear_floor_a_height")  || 
        (topic_name=="fear_floor_b_height")) {
        result = "std_msgs/Float32";
    } else if ((topic_name=="fear_floor_a_image") || (topic_name=="fear_floor_b_image")) {
        result = "std_msgs/String";
    } else {
        throw std::runtime_error("unknown topic name " + topic_name);
    }
    return result;
}

void init_texture(osg::ref_ptr<osg::Texture2D>& texture, std::string s) {
    texture->setDataVariance(osg::Object::DYNAMIC);
    texture->setFilter(osg::Texture::MIN_FILTER, osg::Texture::NEAREST);
    texture->setFilter(osg::Texture::MAG_FILTER, osg::Texture::NEAREST);
    texture->setWrap(osg::Texture::WRAP_S, osg::Texture::REPEAT);
    texture->setWrap(osg::Texture::WRAP_T, osg::Texture::REPEAT);
    texture->setImage(load_image_file(s));
}

osg::ref_ptr<osg::PositionAttitudeTransform> create_virtual_world()
{
  osg::ref_ptr<osg::PositionAttitudeTransform> myroot = new osg::PositionAttitudeTransform;
  myroot->addDescription("virtual world root node");
  //disable all lighting in the scene so objects always appear the same color.
  myroot->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

  osg::ref_ptr<osg::Geode> geode = new osg::Geode;
  myroot->addChild(geode.get());

  float l = 0.005f;

  _plane_z_b = new osg::Box(osg::Vec3(0.0f, 0.0f, -0.4f),_bL,_bL,l);
  _shape_b = new osg::ShapeDrawable(_plane_z_b);
#if SOLID_COLOR
  _shape_b->setColor(osg::Vec4(1.0f, 0.0f, 0.0f, 1.0f));
#else
  _texture_b = new osg::Texture2D();
  init_texture(_texture_b, "checkerboard128.png");
  osg::StateSet *ssb = _shape_b->getOrCreateStateSet();
  ssb->setMode(GL_BLEND, osg::StateAttribute::ON);
  ssb->setTextureAttributeAndModes(0, _texture_b, osg::StateAttribute::ON);
#endif
  geode->addDrawable(_shape_b.get());

  _plane_z_a_xminus = new osg::Box(osg::Vec3(_L/-2.0, 0.0f, -0.1f),_L,_L,l);
  _shape_a = new osg::ShapeDrawable(_plane_z_a_xminus);
#if SOLID_COLOR
  _shape_a->setColor(osg::Vec4(0.0f, 0.0f, 1.0f, 1.0f));
#else
  _texture_a = new osg::Texture2D();
  init_texture(_texture_a, "checkerboard64.png");
  osg::StateSet *ssa = _shape_a->getOrCreateStateSet();
  ssa->setMode(GL_BLEND, osg::StateAttribute::ON);
  ssa->setTextureAttributeAndModes(0, _texture_a, osg::StateAttribute::ON);
#endif
  geode->addDrawable(_shape_a.get());

  return myroot;
}



private:
    float                                           _L,_bL;
    osg::ref_ptr<osg::Geode> _geode;
    osg::ref_ptr<osg::PositionAttitudeTransform>    _virtual_world;
    osg::ref_ptr<osg::Box>                          _plane_z_b;
    osg::ref_ptr<osg::ShapeDrawable>                _shape_b;
    osg::ref_ptr<osg::Texture2D>                    _texture_b;
    osg::ref_ptr<osg::Box>                          _plane_z_a_xminus;
    osg::ref_ptr<osg::ShapeDrawable>                _shape_a;
    osg::ref_ptr<osg::Texture2D>                    _texture_a;

};


POCO_BEGIN_MANIFEST(StimulusInterface)
POCO_EXPORT_CLASS(StimulusFear)
POCO_END_MANIFEST

void pocoInitializeLibrary()
{
}

void pocoUninitializeLibrary()
{
}
