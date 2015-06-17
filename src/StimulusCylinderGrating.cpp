/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */
#include "flyvr/StimulusInterface.hpp"
#include "flyvr/flyvr_assert.h"

#include "json2osg.hpp"

#include "Poco/ClassLibrary.h"

#include <iostream>
#include <stdio.h>
#include <string.h>

#include <osg/MatrixTransform>
#include <osg/ShapeDrawable>
#include <osg/Geode>

#include <OpenThreads/ScopedLock>

#define NUM_GRATINGS 1

#include <math.h>
const static double D2R = M_PI/180.0;

typedef struct
{
    bool  reset_phase_position;
    float phase_position;
    float phase_velocity;
    float wavelength;
    float contrast;
    float orientation;
} GratingParams;

GratingParams parse_grating_info(const json_t * const root) {
    GratingParams result;

    json_t *data_json;

    data_json = json_object_get(root, "reset_phase_position");
    flyvr_assert(data_json != NULL);
    flyvr_assert(json_is_boolean(data_json));
    result.reset_phase_position = json_is_true( data_json );

    data_json = json_object_get(root, "phase_position");
    flyvr_assert(data_json != NULL);
    flyvr_assert(json_is_real(data_json));
    result.phase_position = json_real_value( data_json );

    data_json = json_object_get(root, "phase_velocity");
    flyvr_assert(data_json != NULL);
    flyvr_assert(json_is_real(data_json));
    result.phase_velocity = json_real_value( data_json );

    data_json = json_object_get(root, "wavelength");
    flyvr_assert(data_json != NULL);
    flyvr_assert(json_is_real(data_json));
    result.wavelength = json_real_value( data_json );

    data_json = json_object_get(root, "contrast");
    flyvr_assert(data_json != NULL);
    flyvr_assert(json_is_real(data_json));
    result.contrast = json_real_value( data_json );

    data_json = json_object_get(root, "orientation");
    flyvr_assert(data_json != NULL);
    flyvr_assert(json_is_real(data_json));
    result.orientation = json_real_value( data_json );

    return result;
}

typedef struct
{
    OpenThreads::Mutex                  mutex;
    bool                                dirty;
    GratingParams                       params;
    osg::ref_ptr<osg::Uniform>          u_phase_position;
    osg::ref_ptr<osg::Uniform>          u_wavelength;
    osg::ref_ptr<osg::Uniform>          u_contrast;
    osg::ref_ptr<osg::Uniform>          u_orientation;
} GratingType;

typedef struct
{
    osg::ref_ptr<osg::Cylinder>         cylinder;
    osg::ref_ptr<osg::ShapeDrawable>    shape;
    osg::ref_ptr<osg::Geode>            geode;

    osg::ref_ptr<osg::StateSet>         state_set;

    osg::ref_ptr<osg::Program>          program;
    osg::ref_ptr<osg::Shader>           vertex_shader;
    osg::ref_ptr<osg::Shader>           fragment_shader;

    GratingType gratings[NUM_GRATINGS];
} CylInfo;

typedef struct
{
    osg::ref_ptr<osg::Box>              box;
    osg::ref_ptr<osg::ShapeDrawable>    shape;
    osg::ref_ptr<osg::Geode>            geode;

    osg::ref_ptr<osg::StateSet>         state_set;

    osg::ref_ptr<osg::Program>          program;
    osg::ref_ptr<osg::Shader>           vertex_shader;
    osg::ref_ptr<osg::Shader>           fragment_shader;

    GratingType gratings[NUM_GRATINGS];
} SquareInfo;

class StimulusCylinderGrating: public StimulusInterface
{
public:
    StimulusCylinderGrating();

    std::vector<std::string> get_topic_names() const;
    void receive_json_message(const std::string& topic_name, const std::string& json_message);
    std::string get_message_type(const std::string& topic_name) const;
    void update( const double& time, const osg::Vec3& observer_position, const osg::Quat& observer_orientation );

    std::string name() const { return "StimulusCylinderGrating"; }
    osg::ref_ptr<osg::Group> get_3d_world() {return _virtual_world; }

private:
    osg::ref_ptr<osg::PositionAttitudeTransform>            _virtual_world;
    double                              _t0;
    CylInfo _cyl;
    SquareInfo _square;
    int _geometry_type;
    osg::Geode*                         _current_world;
    bool _lock_z;


    osg::ref_ptr<osg::PositionAttitudeTransform> create_virtual_world();
    void post_init(bool);
    void init_cyl(CylInfo&);
    void init_square(SquareInfo&);
    void set_grating_info( int i, GratingParams& new_values);
    void set_geometry_type( int value );
};

StimulusCylinderGrating::StimulusCylinderGrating() :
    _t0(-1)
{
    _virtual_world = create_virtual_world();
}

std::vector<std::string> StimulusCylinderGrating::get_topic_names() const
{
    std::vector<std::string> result;
	result.push_back("grating_info");
	result.push_back("grating_geometry_type");
    result.push_back("grating_lock_z");

    return result;
}

void StimulusCylinderGrating::set_geometry_type( int value ) {

    GratingParams orig_params = (_geometry_type == 0) ? _cyl.gratings[0].params : _square.gratings[0].params;

    _geometry_type = value;
    if (_geometry_type == 0) {
        std::cout << "using cylinder " << std::endl;

        _cyl.gratings[0].params = orig_params; // carry over old parameters

        // Cylinder
        _virtual_world->removeChild( _current_world );

        _current_world = _cyl.geode.get();
        _virtual_world->addChild( _current_world );
    } else if (_geometry_type == 1) {

        std::cout << "using square" << std::endl;

        _square.gratings[0].params = orig_params; // carry over old parameters

        // Square
        _virtual_world->removeChild( _current_world );

        _current_world = _square.geode.get();
        _virtual_world->addChild( _current_world );
    } else {
        throw std::runtime_error("unknown geometry type");
    }

    set_grating_info(0, orig_params); // set the dirty flag, if nothing else...

}

void StimulusCylinderGrating::receive_json_message(const std::string& topic_name, const std::string& json_message)
{
    json_t *root;
    json_error_t error;

    root = json_loads(json_message.c_str(), 0, &error);
    flyvr_assert(root != NULL);

    if (topic_name=="grating_info") {
        GratingParams new_values;

        new_values = parse_grating_info(root);
        set_grating_info(0,new_values);

    } else if (topic_name=="grating_geometry_type") {

        int geometry_type = parse_int(root);
        set_geometry_type( geometry_type );
    } else if (topic_name=="grating_lock_z") {
        _lock_z = (bool)parse_bool(root);
        if (!_lock_z) {
            _virtual_world->setPosition( osg::Vec3( 0.0, 0.0, 0.0 ));
        }
    } else {
        throw std::runtime_error("unknown topic name");
    }

    json_decref(root);
}

std::string StimulusCylinderGrating::get_message_type(const std::string& topic_name) const
{
    std::string result;
    if (topic_name=="grating_info") {
        result = "strawlab_freeflight_experiments/CylinderGratingInfo";
    } else if (topic_name=="grating_geometry_type") {
        result = "std_msgs/Int32";
    } else if (topic_name=="grating_lock_z") {
        result = "std_msgs/Bool";
    } else {
        throw std::runtime_error("unknown topic name");
    }
    return result;
}

void StimulusCylinderGrating::update( const double& time, const osg::Vec3& observer_position, const osg::Quat& observer_orientation )
{
    double dt=0.0;
    if (_t0 > 0) {
        // update by dt seconds
        dt = time-_t0;
        _t0 = time;
    } else {
        // first iteration
        _t0 = time;
    }

    for (int i=0; i<NUM_GRATINGS; i++) {
        GratingType &grating = (_geometry_type == 0) ? _cyl.gratings[i] : _square.gratings[i];

        {
            OpenThreads::ScopedLock<OpenThreads::Mutex> lock(grating.mutex);

            if (grating.dirty) {
                grating.dirty = false; // reset dirty flag

                grating.u_phase_position->set(grating.params.phase_position);
                grating.u_wavelength->set(grating.params.wavelength);
                grating.u_contrast->set(grating.params.contrast);
                grating.u_orientation->set(grating.params.orientation);
            }

            if (grating.params.phase_velocity != 0) {
                grating.params.phase_position += dt*grating.params.phase_velocity;
                grating.u_phase_position->set(grating.params.phase_position);
            }
        }

    }

    if (_lock_z) {
        _virtual_world->setPosition( osg::Vec3(0.0, 0.0, observer_position[2]) );
    }
}

void StimulusCylinderGrating::init_cyl(CylInfo& cyl) {

    osg::ref_ptr<osg::TessellationHints> hints = new osg::TessellationHints;
    hints->setDetailRatio(2.0f);
    hints->setCreateTop(false);
    hints->setCreateBottom(false);

    osg::Vec3 center = osg::Vec3(0.0f,0.0f,0.0f);
    float radius = 1.0f;
    float height = 3.0;

    cyl.cylinder = new osg::Cylinder(center,radius,height);
    cyl.shape = new osg::ShapeDrawable(cyl.cylinder, hints.get());

    const char* phase_position_name;
    const char* wavelength_name;
    const char* contrast_name;
    const char* orientation_name;

    for (int i=0; i<NUM_GRATINGS; i++) {
        GratingParams new_values;
        new_values.phase_position = 0.0;
        new_values.phase_velocity = 360*D2R;
        new_values.wavelength = 20*D2R;
        new_values.contrast = 1.0;
        new_values.orientation = 0;

        if (i==0) {
            phase_position_name="phase_position0";
            wavelength_name="wavelength0";
            contrast_name="contrast0";
            orientation_name="orientation0";
        } else {
            flyvr_assert(false);
        }

        cyl.gratings[i].u_phase_position = new osg::Uniform( osg::Uniform::FLOAT, phase_position_name );
        cyl.gratings[i].u_wavelength = new osg::Uniform( osg::Uniform::FLOAT, wavelength_name );
        cyl.gratings[i].u_contrast = new osg::Uniform( osg::Uniform::FLOAT, contrast_name );
        cyl.gratings[i].u_orientation = new osg::Uniform( osg::Uniform::FLOAT, orientation_name );

        set_grating_info( i, new_values);
    }

    cyl.geode = new osg::Geode;
    cyl.geode->addDrawable(cyl.shape.get());

    cyl.program = new osg::Program;
    cyl.program->setName( "cylinder_shader" );

    cyl.vertex_shader = new osg::Shader( osg::Shader::VERTEX );
    cyl.fragment_shader = new osg::Shader( osg::Shader::FRAGMENT );

    cyl.program->addShader(cyl.vertex_shader);
    cyl.program->addShader(cyl.fragment_shader);

    load_shader_source( cyl.vertex_shader, "grating_blended.vert" );
    load_shader_source( cyl.fragment_shader, "grating_blended.frag" );

    cyl.state_set = cyl.shape->getOrCreateStateSet();

    cyl.state_set->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    cyl.state_set->setMode(GL_BLEND, osg::StateAttribute::ON);

    cyl.state_set->setAttributeAndModes( cyl.program, osg::StateAttribute::ON);

    for (int i=0;i<NUM_GRATINGS;i++) {
        cyl.state_set->addUniform( cyl.gratings[i].u_phase_position );
        cyl.state_set->addUniform( cyl.gratings[i].u_wavelength );
        cyl.state_set->addUniform( cyl.gratings[i].u_contrast );
        cyl.state_set->addUniform( cyl.gratings[i].u_orientation );
    }

}

void StimulusCylinderGrating::init_square(SquareInfo& square) {

    osg::Vec3 center = osg::Vec3(0.0f, 0.0f, -0.016f);
    float lengthX = 10.0f;
    float lengthY = lengthX;
    float lengthZ = 1e-6;

    square.box = new osg::Box(center,lengthX,lengthY,lengthZ);
    square.shape = new osg::ShapeDrawable(square.box); //, hints.get());

    const char* phase_position_name;
    const char* wavelength_name;
    const char* contrast_name;
    const char* orientation_name;

    for (int i=0; i<NUM_GRATINGS; i++) {
        GratingParams new_values;
        new_values.phase_position = 0.0;
        new_values.phase_velocity = 2.0*M_PI;
        new_values.wavelength = 0.038;
        new_values.contrast = 10000.0;
        new_values.orientation = 0;

        if (i==0) {
            phase_position_name="phase_position0";
            wavelength_name="wavelength0";
            contrast_name="contrast0";
            orientation_name="orientation0";
        } else {
            flyvr_assert(false);
        }

        square.gratings[i].u_phase_position = new osg::Uniform( osg::Uniform::FLOAT, phase_position_name );
        square.gratings[i].u_wavelength = new osg::Uniform( osg::Uniform::FLOAT, wavelength_name );
        square.gratings[i].u_contrast = new osg::Uniform( osg::Uniform::FLOAT, contrast_name );
        square.gratings[i].u_orientation = new osg::Uniform( osg::Uniform::FLOAT, orientation_name );

        set_grating_info( i, new_values);
    }

    square.geode = new osg::Geode;
    square.geode->addDrawable(square.shape.get());

    square.program = new osg::Program;
    square.program->setName( "box_shader" );

    square.vertex_shader = new osg::Shader( osg::Shader::VERTEX );
    square.fragment_shader = new osg::Shader( osg::Shader::FRAGMENT );

    square.program->addShader(square.vertex_shader);
    square.program->addShader(square.fragment_shader);

    load_shader_source( square.vertex_shader, "box_grating_blended.vert" );
    load_shader_source( square.fragment_shader, "box_grating_blended.frag" );

    square.state_set = square.shape->getOrCreateStateSet();

    square.state_set->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    square.state_set->setMode(GL_BLEND, osg::StateAttribute::ON);

    square.state_set->setAttributeAndModes( square.program, osg::StateAttribute::ON);

    for (int i=0;i<NUM_GRATINGS;i++) {
        square.state_set->addUniform( square.gratings[i].u_phase_position );
        square.state_set->addUniform( square.gratings[i].u_wavelength );
        square.state_set->addUniform( square.gratings[i].u_contrast );
        square.state_set->addUniform( square.gratings[i].u_orientation );
    }

}

osg::ref_ptr<osg::PositionAttitudeTransform> StimulusCylinderGrating::create_virtual_world() {
    osg::ref_ptr<osg::PositionAttitudeTransform> myroot = new osg::PositionAttitudeTransform;
    myroot->addDescription("virtual world root node");
    return myroot;
}

void StimulusCylinderGrating::post_init(bool slave)
{
    init_cyl(_cyl);
    init_square(_square);

    _geometry_type = 0;
    _current_world = _cyl.geode.get();
    _virtual_world->addChild( _current_world );
}

void StimulusCylinderGrating::set_grating_info(int i, GratingParams &new_values) {
    GratingType &grating = (_geometry_type == 0) ? _cyl.gratings[0] : _square.gratings[0];
    {
        OpenThreads::ScopedLock<OpenThreads::Mutex> lock(grating.mutex);

        float orig_position = grating.params.phase_position;
        grating.params = new_values;
        if (!new_values.reset_phase_position) {
            grating.params.phase_position = orig_position;
        }
        grating.dirty = true;
    }
}

POCO_BEGIN_MANIFEST(StimulusInterface)
POCO_EXPORT_CLASS(StimulusCylinderGrating)
POCO_END_MANIFEST

void pocoInitializeLibrary()
{
}

void pocoUninitializeLibrary()
{
}
