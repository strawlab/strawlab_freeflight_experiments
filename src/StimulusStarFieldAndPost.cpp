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

#include <boost/filesystem.hpp>

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
// class StimulusStarFieldAndPost
// -----------------------------------------------------------

class StimulusStarFieldAndPost: public StimulusInterface
{
public:
    StimulusStarFieldAndPost();

    std::string name() const { return "StimulusStarFieldAndPost"; }
    void post_init();

    void createStarfieldEffect( osgParticle::ModularEmitter* emitter, osgParticle::ModularProgram* program );

    osg::ref_ptr<osg::Group> get_3d_world() {return _group; }

    virtual osg::Vec4 get_clear_color() const;

    std::vector<std::string> get_topic_names() const;
    void receive_json_message(const std::string& topic_name, const std::string& json_message);
    std::string get_message_type(const std::string& topic_name) const;

    void setVelocity(double x, double y, double z);

private:
    osg::ref_ptr<osg::Group> _group;
    osg::Vec3 _starfield_velocity;
    osg::ref_ptr<ConstantShooter> _shooter;
    osg::ref_ptr<osgParticle::Placer> _placer;
    osg::ref_ptr<VelocityOperator> _vel_operator;
};

StimulusStarFieldAndPost::StimulusStarFieldAndPost() {
    _shooter = new ConstantShooter;
    _placer = new osgParticle::BoxPlacer;
    _vel_operator = new VelocityOperator;

    setVelocity( 0.0, 0.0, 0.0);
}

void StimulusStarFieldAndPost::post_init() {
    // this is based on the OSG example osgparticleshader.cpp

    osg::ref_ptr<osgParticle::ParticleSystem> ps = new osgParticle::ParticleSystem;
    ps->getDefaultParticleTemplate().setLifeTime( 5.0f );
    ps->getDefaultParticleTemplate().setShape( osgParticle::Particle::POINT );
    ps->setVisibilityDistance( -1.0f );

    std::string textureFile = get_plugin_data_path("blackstar.png");
    ps->setDefaultAttributesUsingShaders( textureFile, false, 0 );

    osg::StateSet* stateset = ps->getOrCreateStateSet();
    stateset->setAttribute( new osg::Point(5.0f) ); // pointSize
    stateset->setTextureAttributeAndModes( 0, new osg::PointSprite, osg::StateAttribute::ON );

    osg::ref_ptr<osgParticle::ModularEmitter> emitter = new osgParticle::ModularEmitter;
    emitter->setParticleSystem( ps.get() );

    osg::ref_ptr<osgParticle::ModularProgram> program = new osgParticle::ModularProgram;
    program->setParticleSystem( ps.get() );
    program->addOperator( _vel_operator.get() );

    createStarfieldEffect( emitter.get(), program.get() );

    osg::ref_ptr<osg::MatrixTransform> parent = new osg::MatrixTransform;
    parent->addChild( emitter.get() );
    parent->addChild( program.get() );

    osg::ref_ptr<osgParticle::ParticleSystemUpdater> updater = new osgParticle::ParticleSystemUpdater;

    osg::ref_ptr<osg::Group> root = new osg::Group;
    root->addChild( parent.get() );
    root->addChild( updater.get() );

    updater->addParticleSystem( ps.get() );

    osg::ref_ptr<osg::Geode> geode = new osg::Geode;
    geode->addDrawable( ps.get() );
    root->addChild( geode.get() );

    _group = root;
    _group->setName("StimulusStarFieldAndPost._group");
}

void StimulusStarFieldAndPost::createStarfieldEffect( osgParticle::ModularEmitter* emitter, osgParticle::ModularProgram* program ){
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

osg::Vec4 StimulusStarFieldAndPost::get_clear_color() const {
    return osg::Vec4(1,1,1,1); // white
}

std::vector<std::string> StimulusStarFieldAndPost::get_topic_names() const {
    std::vector<std::string> result;
    result.push_back("velocity");
    return result;
}

void StimulusStarFieldAndPost::receive_json_message(const std::string& topic_name,
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

void StimulusStarFieldAndPost::setVelocity(double x, double y, double z) {
    _starfield_velocity = osg::Vec3(x,y,z);
    _shooter->setVelocity( _starfield_velocity );
    _vel_operator->setVelocity( _starfield_velocity );
}

std::string StimulusStarFieldAndPost::get_message_type(const std::string& topic_name) const {
    std::string result;

    if (topic_name=="velocity") {
        result = "geometry_msgs/Vector3";
    } else {
        throw std::runtime_error("unknown topic name");
    }
    return result;
}

POCO_BEGIN_MANIFEST(StimulusInterface)
POCO_EXPORT_CLASS(StimulusStarFieldAndPost)
POCO_END_MANIFEST

void pocoInitializeLibrary()
{
}

void pocoUninitializeLibrary()
{
}
