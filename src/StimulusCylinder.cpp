/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- */
#include "vros_display/stimulus_interface.h"
#include "vros_display/vros_assert.h"

#include "json2osg.hpp"

#include "Poco/ClassLibrary.h"

#include <iostream>

#include <osg/MatrixTransform>
#include <osg/TextureCubeMap>
#include <osg/Texture2D>
#include <osg/ShapeDrawable>
#include <osg/Material>
#include <osg/CullFace>
#include <osg/Light>
#include <osg/LightSource>
#include <osg/LightModel>

#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgDB/FileUtils>


class StimulusCylinder: public StimulusInterface
{
public:
	const static float DEFAULT_RADIUS = 0.5f;
	const static float DEFAULT_HEIGHT = 1.0f;	

std::string name() const {
	return "StimulusCylinder";
}

virtual void post_init(void) {
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
    result.push_back("cylinder_radius");
    result.push_back("cylinder_height");
    result.push_back("cylinder_image");
    result.push_back("model_pose");
    return result;
}

void receive_json_message(const std::string& topic_name, const std::string& json_message) {
    json_t *root;
    json_error_t error;

    root = json_loads(json_message.c_str(), 0, &error);
    vros_assert(root != NULL);

    if (topic_name=="cylinder_radius") {
		set_cylinder_radius(parse_float(root));
    } else if (topic_name=="cylinder_height") {
		set_cylinder_height(parse_float(root));
    } else if (topic_name=="cylinder_image") {
		set_cylinder_image(parse_string(root));
    } else if (topic_name=="model_pose") {
        json_t *data_json;
		osg::Vec3 position;
        data_json = json_object_get(root, "position");
		position = parse_vec3(data_json);
		set_cylinder_position(position.x(),position.y(),position.z());
    } else {
        throw std::runtime_error("unknown topic name");
    }
}

std::string get_message_type(const std::string& topic_name) const {
    std::string result;

    if (topic_name=="cylinder_radius") {
        result = "std_msgs/Float32";
    } else if (topic_name=="cylinder_height") {
        result = "std_msgs/Float32";
    } else if (topic_name=="cylinder_image") {
        result = "std_msgs/String";
    } else if (topic_name=="model_pose") {
        result = "geometry_msgs/Pose";
    } else {
        throw std::runtime_error("unknown topic name");
    }
    return result;
}

osg::ref_ptr<osg::Group> create_virtual_world() {
	osg::ref_ptr<osg::MatrixTransform> myroot = new osg::MatrixTransform; myroot->addDescription("virtual world root node");
	//add_default_skybox(myroot);

	// Create a geometry transform node enabling use cut-and-pasted
	// geometry from OSG example and have it on the XY plane with Z
	// pointing up.
	osg::ref_ptr<osg::MatrixTransform> geom_transform_node = new osg::MatrixTransform;
	geom_transform_node->setMatrix(osg::Matrix::rotate(osg::DegreesToRadians(90.0),1.0,0.0,0.0));

	osg::ref_ptr<osg::Geode> geode = new osg::Geode;
	osg::ref_ptr<osg::MatrixTransform> cylinder_transform = new osg::MatrixTransform;
	//rotate the cylinder to orientate up like the flycave
	cylinder_transform->setMatrix(osg::Matrix::rotate(osg::DegreesToRadians(90.0),1.0,0.0,0.0));
	cylinder_transform->addChild(geode.get());
	geom_transform_node->addChild(cylinder_transform.get());

	osg::ref_ptr<osg::TessellationHints> hints = new osg::TessellationHints;
	hints->setDetailRatio(2.0f);

	_cylinder = new osg::Cylinder();
	_shape = new osg::ShapeDrawable(_cylinder, hints.get());
	set_cylinder_position(0.0,0.0,0.0);
	set_cylinder_radius(DEFAULT_RADIUS);
	set_cylinder_height(DEFAULT_HEIGHT);

	//shape->setColor(osg::Vec4(1.0f, 0.0f, 0.0f, 1.0f));
	//shape->getOrCreateStateSet()->setMode(GL_BLEND, osg::StateAttribute::ON);
	geode->addDrawable(_shape.get());

	_texture = new osg::Texture2D;
	_texture->setDataVariance(osg::Object::DYNAMIC);
	_texture->setFilter(osg::Texture::MIN_FILTER, osg::Texture::LINEAR_MIPMAP_LINEAR);
	_texture->setFilter(osg::Texture::MAG_FILTER, osg::Texture::LINEAR);
	_texture->setWrap(osg::Texture::WRAP_S, osg::Texture::CLAMP);
	_texture->setWrap(osg::Texture::WRAP_T, osg::Texture::CLAMP);
	set_cylinder_image("stars2x800r.png");

	osg::ref_ptr<osg::Material> material = new osg::Material;
	material->setEmission(osg::Material::FRONT, osg::Vec4(0.8, 0.8, 0.8, 1.0));
	//material->setColorMode(osg::Material::DIFFUSE);
	//material->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0, 0, 0, 1));
	//material->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(1, 1, 1, 1));
	//material->setShininess(osg::Material::FRONT_AND_BACK, 64.0f);

	osg::StateSet *sphereStateSet = _shape->getOrCreateStateSet();
	sphereStateSet->setAttribute(material);
	sphereStateSet->setTextureAttributeAndModes(0, _texture, osg::StateAttribute::ON);

	//  {
	//	  osg::ref_ptr<osg::Light> _light = new osg::Light;
	//	  _light->setLightNum(0);
	//	  _light->setAmbient(osg::Vec4(0.00f,0.0f,0.00f,1.0f));
	//	  _light->setDiffuse(osg::Vec4(0.8f,0.8f,0.8f,1.0f));
	//	  _light->setSpecular(osg::Vec4(1.0f,1.0f,1.0f,1.0f));
	//	  _light->setPosition(osg::Vec4(2.0, 2.0, 5.0, 1.0));

	//	  geom_transform_node->getOrCreateStateSet()->setAssociatedModes(_light.get(),osg::StateAttribute::ON);

	//	  // enable lighting by default.
	//	  geom_transform_node->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::ON);

	//  }
	myroot->addChild(geom_transform_node);

	return myroot;
}

private:
	osg::ref_ptr<osg::Group> 			_virtual_world;
    osg::ref_ptr<osg::Cylinder> 		_cylinder;
	osg::ref_ptr<osg::Texture2D> 		_texture;
	osg::ref_ptr<osg::ShapeDrawable> 	_shape;

void set_cylinder_position(float x, float y, float z) {
	_cylinder->setCenter(osg::Vec3(x,y,z));
	_shape->dirtyDisplayList();
	_shape->dirtyBound(); 
}

void set_cylinder_radius(float r) {
	_cylinder->setRadius(r);
	_shape->dirtyDisplayList();
	_shape->dirtyBound(); 
}

void set_cylinder_height(float h) {
	_cylinder->setHeight(h);
	_shape->dirtyDisplayList();
	_shape->dirtyBound(); 
}

void set_cylinder_image(std::string s) {
	osg::ref_ptr<osg::Image> image = load_image_file(s);
	_texture->setImage(image);
}

};


POCO_BEGIN_MANIFEST(StimulusInterface)
POCO_EXPORT_CLASS(StimulusCylinder)
POCO_END_MANIFEST

void pocoInitializeLibrary()
{
}

void pocoUninitializeLibrary()
{
}
