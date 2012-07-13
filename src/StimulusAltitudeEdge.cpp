/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- */
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
#include <osg/Texture2D>

#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgDB/FileUtils>

#include <boost/filesystem.hpp>

#include <jansson.h>

#define  grating_tex_width  1
#define  grating_tex_height 1024

class StimulusAltitudeEdge: public StimulusInterface
{
public:
	StimulusAltitudeEdge();

	std::string name() const { return "StimulusAltitudeEdge"; }
	void post_init(std::string config_data_dir);

	osg::ref_ptr<osg::Group> get_3d_world() {return _group; }

	void update(const double& t,
				const osg::Vec3& position,
				const osg::Quat& orientation);
	std::vector<std::string> get_topic_names() const;
	void receive_json_message(const std::string& topic_name, const std::string& json_message);
	std::string get_message_type(const std::string& topic_name) const;

private:
	void _set_texture_data();
	osg::ref_ptr<osg::Group> _group;
	float _edge_fraction;
	osg::ref_ptr<osg::Texture2D> left_texture;
};


StimulusAltitudeEdge::StimulusAltitudeEdge() : _edge_fraction(0.5) {
    left_texture = new osg::Texture2D();
	left_texture->setDataVariance(osg::Object::DYNAMIC);
	osg::ref_ptr<osg::Geode> left_geode = new osg::Geode;
	{
		osg::ref_ptr<osg::TessellationHints> hints = new osg::TessellationHints;
		hints->setDetailRatio(2.0f);
		osg::ref_ptr<osg::ShapeDrawable> shape;

		osg::ref_ptr<osg::Shape> cyl = new osg::Cylinder(
												  osg::Vec3(0.0f, 0.0f, 0.0f), // center
												  0.5, //radius
												  1.0); //height
		//cyl->setRotation();
		shape = new osg::ShapeDrawable( cyl, hints.get());
		shape->setColor(osg::Vec4(0.5f, 0.5f, 0.7f, 1.0f));
		left_geode->addDrawable(shape.get());

		osg::StateSet* state = left_geode->getOrCreateStateSet();
		state->setTextureAttributeAndModes(0, left_texture, osg::StateAttribute::ON);
		state->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
	}

  _group = new osg::Group;
  _group->addChild(left_geode);
  _group->setName("StimulusAltitudeEdge._group");

}

void StimulusAltitudeEdge::post_init(std::string config_data_dir) {
	std::cout << "StimulusAltitudeEdge: created" << std::endl;
	osgDB::writeNodeFile(*_group.get(), "StimulusAltitudeEdge.osg");
}

void StimulusAltitudeEdge::_set_texture_data() {
	osg::Image* image = new osg::Image;

	image->allocateImage(grating_tex_width,grating_tex_height,1,
						 GL_LUMINANCE,GL_UNSIGNED_BYTE);
	unsigned char* ptr = image->data(0,0,0);

	for (int i=0;i<grating_tex_height;i++) {
		float frac = (float)i/(float)grating_tex_height;
		if (frac < _edge_fraction) {
			ptr[i] = 0;
		} else {
			ptr[i] = 255;
		}
	}
	left_texture->setImage(image);
}

void StimulusAltitudeEdge::update(const double& t,
                                  const osg::Vec3& position,
                                  const osg::Quat& orientation) {
	_set_texture_data();
}

std::vector<std::string> StimulusAltitudeEdge::get_topic_names() const {
	std::vector<std::string> result;
	result.push_back("edge_fraction");
	return result;
}

void StimulusAltitudeEdge::receive_json_message(const std::string& topic_name,
                                                const std::string& json_message) {
	json_t *root;
	json_error_t error;

	root = json_loads(json_message.c_str(), 0, &error);
	if(!root) {
		fprintf(stderr, "error: in %s(%d) on json line %d: %s\n", __FILE__, __LINE__, error.line, error.text);
		throw std::runtime_error("error in json");
	}

	json_t *data_json = json_object_get(root, "edge_fraction");
	if(!json_is_number(data_json)){
		fprintf(stderr, "error: in %s(%d): expected number\n", __FILE__, __LINE__);
		throw std::runtime_error("error in json");
	}
	_edge_fraction = json_number_value( data_json );

}

std::string StimulusAltitudeEdge::get_message_type(const std::string& topic_name) const {
	std::string result;

	if (topic_name=="edge_fraction") {
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
