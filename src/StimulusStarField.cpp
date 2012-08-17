/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */
#include "vros_display/stimulus_interface.h"
#include "vros_display/vros_assert.h"

#include "Poco/ClassLibrary.h"

#include <iostream>

#include <osg/Point>
#include <osg/PointSprite>
#include <osg/MatrixTransform>
#include <osg/ShapeDrawable>
#include <osg/Material>
#include <osg/CullFace>
#include <osg/Light>
#include <osg/LightSource>
#include <osg/LightModel>
#include <osg/Texture2D>
#include <osg/BlendFunc>

#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgDB/FileUtils>

#include <boost/filesystem.hpp>

#include <jansson.h>

class MyGeometryCallback :
    public osg::Drawable::UpdateCallback,
    public osg::Drawable::AttributeFunctor {
public:
    MyGeometryCallback() :  _firstCall(true) {}

    virtual void update(osg::NodeVisitor* nv,osg::Drawable* drawable) {
        const osg::FrameStamp* fs = nv->getFrameStamp();
        double simulationTime = fs->getSimulationTime();
        if (_firstCall) {
            _firstCall = false;
            _startTime = simulationTime;
        }

        _time = simulationTime-_startTime;

        drawable->accept(*this);
        drawable->dirtyBound();
        osg::Geometry* this_geom = dynamic_cast<osg::Geometry*>(drawable);
        if (this_geom!=NULL) {
            osg::Vec3Array* vertices = dynamic_cast<osg::Vec3Array*>(this_geom->getVertexArray());
            vertices->at(0).set( sin(_time), cos(_time), 0.0);
        }
    }
private:
    double _time, _startTime;
    bool _firstCall;
};

class StimulusStarField: public StimulusInterface
{
public:
    StimulusStarField();

    std::string name() const { return "StimulusStarField"; }
    void post_init();

    osg::ref_ptr<osg::Group> get_3d_world() {return _group; }

    virtual osg::Vec4 get_clear_color() const;

    std::vector<std::string> get_topic_names() const;
    void receive_json_message(const std::string& topic_name, const std::string& json_message);
    std::string get_message_type(const std::string& topic_name) const;

    void setVelocity(double x, double y, double z);

private:
    osg::ref_ptr<osg::Group> _group;
    osg::Vec3 _starfield_velocity;
};

StimulusStarField::StimulusStarField() {
    setVelocity( 0.0, 0.0, 0.0);
}

void StimulusStarField::post_init() {
    osg::ref_ptr<osg::Geode> geode = new osg::Geode();
    osg::Texture2D* tex = new osg::Texture2D;
    {
        osg::Image* im = osgDB::readImageFile(get_plugin_data_path("star.png"));
        vros_assert(im!=NULL);
        tex->setImage(im);
    }

    {
        osg::ref_ptr<osg::Geometry> this_geom = new osg::Geometry;

        osg::Vec3Array* vertices = new osg::Vec3Array;
        for (int i=0;i<1000;i++) {
            vertices->push_back( osg::Vec3( fmod(i,10.0), fmod(i/10.0,10.0), fmod(i/100.0,10.0) ) );
        }
        this_geom->setVertexArray(vertices);
        this_geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POINTS,0,vertices->size()));
        this_geom->setDataVariance( osg::Object::DYNAMIC );
        this_geom->setSupportsDisplayList(false);
        this_geom->setUpdateCallback(new MyGeometryCallback());
        geode->addDrawable(this_geom.get());
    }

    {
        osg::StateSet* state = geode->getOrCreateStateSet();
        state->setMode(GL_LIGHTING,osg::StateAttribute::OFF);

        osg::BlendFunc *fn = new osg::BlendFunc();
        state->setAttributeAndModes(fn, osg::StateAttribute::ON);

        osg::PointSprite *sprite = new osg::PointSprite();
        state->setTextureAttributeAndModes(0, sprite, osg::StateAttribute::ON);

        osg::Point* _point = new osg::Point;
        _point->setSize(20.0f);
        state->setAttribute(_point);

        state->setRenderingHint( osg::StateSet::TRANSPARENT_BIN );

        osg::Program* Program;
        osg::Shader*  VertObj;
        osg::Shader*  FragObj;

        Program = new osg::Program;
        Program->setName( "starfield" );
        VertObj = new osg::Shader( osg::Shader::VERTEX );
        FragObj = new osg::Shader( osg::Shader::FRAGMENT );
        Program->addShader( FragObj );
        Program->addShader( VertObj );

        load_shader_source( VertObj, "starfield.vert" );
        load_shader_source( FragObj, "starfield.frag" );

        state->setAttributeAndModes(Program, osg::StateAttribute::ON);

        osg::Uniform* sampler = new osg::Uniform( osg::Uniform::SAMPLER_2D,
                                                  "star_tex" );
        state->addUniform( sampler );
        state->setTextureAttributeAndModes(0,tex,osg::StateAttribute::ON);
    }

    osg::ref_ptr<osg::Group> root = new osg::Group;
    root->addChild(geode);
    _group = root;
    _group->setName("StimulusStarField._group");
}

osg::Vec4 StimulusStarField::get_clear_color() const {
    return osg::Vec4(0.0, 0.0, 0.0, 0.0); // transparent black
}

std::vector<std::string> StimulusStarField::get_topic_names() const {
    std::vector<std::string> result;
    result.push_back("velocity");
    return result;
}

void StimulusStarField::receive_json_message(const std::string& topic_name,
                                             const std::string& json_message) {
    json_t *root;
    json_error_t error;
    double x,y,z;

    root = json_loads(json_message.c_str(), 0, &error);
    vros_assert(root != NULL);

    json_t *data_json;

    data_json = json_object_get(root, "x");
    vros_assert(data_json != NULL);
    vros_assert(json_is_number(data_json));
    x = json_number_value( data_json );

    data_json = json_object_get(root, "y");
    vros_assert(data_json != NULL);
    vros_assert(json_is_number(data_json));
    y = json_number_value( data_json );

    data_json = json_object_get(root, "z");
    vros_assert(data_json != NULL);
    vros_assert(json_is_number(data_json));
    z = json_number_value( data_json );

    setVelocity(x,y,z);
}

void StimulusStarField::setVelocity(double x, double y, double z) {
    _starfield_velocity = osg::Vec3(x,y,z);
}

std::string StimulusStarField::get_message_type(const std::string& topic_name) const {
    std::string result;

    if (topic_name=="velocity") {
        result = "geometry_msgs/Vector3";
    } else {
        throw std::runtime_error("unknown topic name");
    }
    return result;
}

POCO_BEGIN_MANIFEST(StimulusInterface)
POCO_EXPORT_CLASS(StimulusStarField)
POCO_END_MANIFEST

void pocoInitializeLibrary()
{
}

void pocoUninitializeLibrary()
{
}
