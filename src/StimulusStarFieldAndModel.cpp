/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */
#include "flyvr/StimulusInterface.hpp"
#include "flyvr/flyvr_assert.h"

#include "json2osg.hpp"

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

#include <osgParticle/Particle>
#include <osgParticle/ParticleSystem>
#include <osgParticle/ParticleSystemUpdater>
#include <osgParticle/ModularEmitter>
#include <osgParticle/ModularProgram>

#include <osgParticle/Operator>
#include <osgParticle/SinkOperator>

#include <osgParticle/Shooter>

#include <osgParticle/Placer>
#include <osgParticle/BoxPlacer>

#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgDB/FileUtils>

#include <jansson.h>

// -----------------------------------------------------------
//
// Geez, this seems way more complicated than necessary. Maybe we
// should just write our own simple particle system instead of writing
// custom plugin classes for OSG's particle system code.
//
// -----------------------------------------------------------

// -----------------------------------------------------------
// class VelocityOperator - update velocity of existing particles
// -----------------------------------------------------------

class VelocityOperator : public osgParticle::Operator {
public:
    VelocityOperator() : Operator() {_velocity.set(1.0f, 1.0f, 1.0f);}
    VelocityOperator( const VelocityOperator& copy, const osg::CopyOp& copyop = osg::CopyOp::SHALLOW_COPY )
    :   Operator(copy, copyop), _velocity(copy._velocity)
    {}
    META_Object( osgParticle, VelocityOperator );

    void setVelocity( float x, float y, float z ) { _velocity.set(x, y, z); }
    void setVelocity( const osg::Vec3& velocity ) { _velocity = velocity; }
    /// Apply the acceleration to a particle. Do not call this method manually.
    inline void operate( osgParticle::Particle* P, double dt ) {P->setVelocity( _velocity );}

protected:
    virtual ~VelocityOperator() {}
    VelocityOperator& operator=( const VelocityOperator& ) { return *this; }

    osg::Vec3 _velocity;
};

// -----------------------------------------------------------
// class ConstantShooter - set velocity of new particles
// -----------------------------------------------------------

class ConstantShooter: public osgParticle::Shooter {
public:
    inline ConstantShooter() : Shooter(), _velocity(0.0f, 0.0f, 0.0f) {}
    inline ConstantShooter(const ConstantShooter& copy, const osg::CopyOp& copyop = osg::CopyOp::SHALLOW_COPY) : Shooter(copy, copyop), _velocity(copy._velocity) {}
    META_Object(osgParticle, ConstantShooter);

    /// Set the range of possible values for initial rotational speed of particles.
    inline void setVelocity(const osg::Vec3& v) {_velocity=v;}

    /// Shoot a particle. Do not call this method manually.
    inline void shoot(osgParticle::Particle* P) const { P->setVelocity(_velocity); }

protected:
    virtual ~ConstantShooter() {}
    ConstantShooter& operator=(const ConstantShooter&) { return *this; }

private:
    osg::Vec3 _velocity;
};

// -----------------------------------------------------------
// class StimulusStarFieldAndModel
// -----------------------------------------------------------

class StimulusStarFieldAndModel: public StimulusInterface
{
public:
    StimulusStarFieldAndModel();

    std::string name() const { return "StimulusStarFieldAndModel"; }
    void post_init(bool slave);

    void createStarfieldEffect( osgParticle::ModularEmitter* emitter, osgParticle::ModularProgram* program );

    osg::ref_ptr<osg::Group> get_3d_world() {return _group; }

    virtual osg::Vec4 get_clear_color() const;

    std::vector<std::string> get_topic_names() const;
    void receive_json_message(const std::string& topic_name, const std::string& json_message);
    std::string get_message_type(const std::string& topic_name) const;

    void _load_stimulus_filename( std::string osg_filename );
    void _update_pat();

private:
    osg::ref_ptr<osg::Group> _group;
    osg::ref_ptr<osg::PositionAttitudeTransform> switch_node;
    osg::Vec3 model_position;
    osg::Quat model_attitude;
    osg::Vec3 _starfield_velocity;
    osg::ref_ptr<ConstantShooter> _shooter;
    osg::ref_ptr<osgParticle::Placer> _placer;
    osg::ref_ptr<VelocityOperator> _vel_operator;
    osg::ref_ptr<osgParticle::ParticleSystem> _ps;

    void set_star_velocity(float x, float y, float z);
    void set_star_size(float v);
};

StimulusStarFieldAndModel::StimulusStarFieldAndModel() {
    _shooter = new ConstantShooter;
    _placer = new osgParticle::BoxPlacer;
    _vel_operator = new VelocityOperator;

    _group = new osg::Group;
    switch_node = new osg::PositionAttitudeTransform;
    _update_pat();
    _group->addChild(switch_node);

}

void StimulusStarFieldAndModel::_update_pat() {
    flyvr_assert(switch_node.valid());
    switch_node->setPosition( model_position );
    switch_node->setAttitude( model_attitude );
}

void StimulusStarFieldAndModel::_load_stimulus_filename( std::string osg_filename ) {

    if (!_group) {
        std::cerr << "_group node not defined!?" << std::endl;
        return;
    }

    // don't show the old switching node.
    _group->removeChild(switch_node);

    // (rely on C++ to delete the old switching node).

    // (create a new switching node.
    switch_node = new osg::PositionAttitudeTransform;
    _update_pat();

    // now load it with new contents
    osg::Node* tmp = osgDB::readNodeFile(osg_filename);
    flyvr_assert(tmp!=NULL);
    switch_node->addChild( tmp );
    _group->addChild(switch_node);
}

void StimulusStarFieldAndModel::post_init(bool slave) {
    std::string osg_filename = get_plugin_data_path("post.osg");
    _load_stimulus_filename( osg_filename );

    // this is based on the OSG example osgparticleshader.cpp

    _ps = new osgParticle::ParticleSystem;
    _ps->getDefaultParticleTemplate().setLifeTime( 5.0f );
    _ps->getDefaultParticleTemplate().setShape( osgParticle::Particle::POINT );
    _ps->setVisibilityDistance( -1.0f );

    std::string textureFile = get_plugin_data_path("blackstar.png");
    _ps->setDefaultAttributesUsingShaders( textureFile, false, 0 );

    osg::StateSet* stateset = _ps->getOrCreateStateSet();
    stateset->setAttribute( new osg::Point(5.0f) ); // pointSize
    stateset->setTextureAttributeAndModes( 0, new osg::PointSprite, osg::StateAttribute::ON );

    osg::ref_ptr<osgParticle::ModularEmitter> emitter = new osgParticle::ModularEmitter;
    emitter->setParticleSystem( _ps.get() );

    osg::ref_ptr<osgParticle::ModularProgram> program = new osgParticle::ModularProgram;
    program->setParticleSystem( _ps.get() );
    program->addOperator( _vel_operator.get() );

    createStarfieldEffect( emitter.get(), program.get() );

    osg::ref_ptr<osg::MatrixTransform> parent = new osg::MatrixTransform;
    parent->addChild( emitter.get() );
    parent->addChild( program.get() );

    osg::ref_ptr<osgParticle::ParticleSystemUpdater> updater = new osgParticle::ParticleSystemUpdater;

    osg::ref_ptr<osg::Group> root = _group;
    root->addChild( parent.get() );
    root->addChild( updater.get() );

    updater->addParticleSystem( _ps.get() );

    osg::ref_ptr<osg::Geode> geode = new osg::Geode;
    geode->addDrawable( _ps.get() );
    root->addChild( geode.get() );

    _group->setName("StimulusStarFieldAndModel._group");

    set_star_velocity( 0.0, 0.0, 0.0);
    set_star_size( 5.0 );
}

void StimulusStarFieldAndModel::createStarfieldEffect( osgParticle::ModularEmitter* emitter, osgParticle::ModularProgram* program ){
    // Emit specific number of particles every frame
    osg::ref_ptr<osgParticle::RandomRateCounter> rrc = new osgParticle::RandomRateCounter;
    rrc->setRateRange( 500, 2000 );

    // Kill particles going inside/outside of specified domains.
    osg::ref_ptr<osgParticle::SinkOperator> sink = new osgParticle::SinkOperator;
    sink->setSinkStrategy( osgParticle::SinkOperator::SINK_OUTSIDE );
    sink->addSphereDomain( osg::Vec3(), 20.0f );

    emitter->setCounter( rrc.get() );
    emitter->setShooter( _shooter.get() );
    emitter->setPlacer( _placer.get() );

    program->addOperator( sink.get() );
}

osg::Vec4 StimulusStarFieldAndModel::get_clear_color() const {
    return osg::Vec4(1,1,1,1); // white
}

std::vector<std::string> StimulusStarFieldAndModel::get_topic_names() const {
    std::vector<std::string> result;
    result.push_back("star_velocity");
    result.push_back("star_size");
    result.push_back("model_pose");
    return result;
}

void StimulusStarFieldAndModel::set_star_velocity(float x, float y, float z) {
    _starfield_velocity = osg::Vec3(x,y,z);
    _shooter->setVelocity( _starfield_velocity );
    _vel_operator->setVelocity( _starfield_velocity );
}

void StimulusStarFieldAndModel::set_star_size(float v) {
    osg::StateSet* stateset = _ps->getOrCreateStateSet();
    stateset->setAttribute( new osg::Point(v) );
}

void StimulusStarFieldAndModel::receive_json_message(const std::string& topic_name,
                                             const std::string& json_message) {

    json_t *root;
    json_error_t error;

    root = json_loads(json_message.c_str(), 0, &error);
    flyvr_assert(root != NULL);

    if (topic_name=="star_velocity") {
        osg::Vec3 vel = parse_vec3(root);
        set_star_velocity(vel[0],vel[1],vel[2]);
    } else if (topic_name=="star_size") {
        set_star_size(parse_float(root));
    } else if (topic_name=="model_pose") {
        json_t *data_json;

        data_json = json_object_get(root, "position");
        model_position = parse_vec3(data_json);

        data_json = json_object_get(root, "orientation");
        model_attitude = parse_quat(data_json);
        _update_pat();
    } else {
        throw std::runtime_error("unknown topic name");
    }

    json_decref(root);
}

std::string StimulusStarFieldAndModel::get_message_type(const std::string& topic_name) const {
    std::string result;

    if (topic_name=="star_velocity") {
        result = "geometry_msgs/Vector3";
    } else if (topic_name=="star_size") {
        result = "std_msgs/Float32";
    } else if (topic_name=="model_pose") {
        result = "geometry_msgs/Pose";
    } else {
        throw std::runtime_error("unknown topic name");
    }
    return result;
}

POCO_BEGIN_MANIFEST(StimulusInterface)
POCO_EXPORT_CLASS(StimulusStarFieldAndModel)
POCO_END_MANIFEST

void pocoInitializeLibrary()
{
}

void pocoUninitializeLibrary()
{
}
