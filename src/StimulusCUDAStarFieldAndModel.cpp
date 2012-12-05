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
#include <osg/AlphaFunc>
#include <osg/PolygonMode>

#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgDB/FileUtils>

#include <osgCompute/Computation>
#include <osgCuda/Buffer>
#include <osgCuda/Geometry>
#include <osgCuda/Program>
#include <osgCudaStats/Stats>
#include <osgCudaInit/Init>

#include <boost/filesystem.hpp>

#include <jansson.h>

//////////////////
// COMPUTATIONS //
//////////////////
extern "C" void move(
                     unsigned int numPtcls,
                     void* ptcls,
                     float velx, float vely, float velz,
                     float etime );

class MovePtcls : public osgCompute::Computation
{
public:
    MovePtcls() {}

    virtual void setFrameStamp( const osg::FrameStamp* fs) { _fs=fs; }
    virtual void launch()
    {
        if( !_ptcls.valid() )
            return;

        if( !_timer.valid() )
        {
            _timer = new osgCuda::Timer;
            _timer->setName( "MovePtcls");
        }

        float time = (float)_fs->getSimulationTime();
        if( _firstFrame )
        {
            _lastTime = time;
            _firstFrame = false;
        }
        float elapsedtime = static_cast<float>(time - _lastTime);
        _lastTime = time;

        _timer->start();

        move(
            _ptcls->getNumElements(),
            _ptcls->map( osgCompute::MAP_DEVICE_TARGET ),
            _vel[0], _vel[1], _vel[2],
            elapsedtime  );

        _timer->stop();
    }

    virtual void setVelocity( const osg::Vec3& v ) {
        _vel = v;
    }

    virtual void acceptResource( osgCompute::Resource& resource )
    {
        if( resource.isIdentifiedBy("PARTICLE BUFFER" ) )
            _ptcls = dynamic_cast<osgCompute::Memory*>( &resource );
    }

    bool isFirstFrame() {
        return _firstFrame;
    }

private:
    osg::ref_ptr<osgCuda::Timer>        _timer;
    double                              _lastTime;
    bool						        _firstFrame;
    osg::ref_ptr<const osg::FrameStamp>       _fs;
    osg::ref_ptr<osgCompute::Memory>    _ptcls;
    osg::Vec3 _vel;
};


extern "C" void emit(
                     unsigned int numPtcls,
                     void* ptcls,
                     void* seeds,
                     unsigned int seedIdx,
                     osg::Vec3f bbmin,
                     osg::Vec3f bbmax );

class EmitPtcls : public osgCompute::Computation
{
public:
    EmitPtcls( osg::Vec3f min, osg::Vec3f max ) : _min(min), _max(max) {}

    virtual void launch()
    {
        if( !_ptcls.valid() )
            return;

        if( !_seeds.valid() )
        {
            _seeds = new osgCuda::Buffer;
            _seeds->setElementSize( sizeof(float) );
            _seeds->setName( "Seeds" );
            _seeds->setDimension(0,_ptcls->getNumElements());

            float* seedsData = (float*)_seeds->map(osgCompute::MAP_HOST_TARGET);
            for( unsigned int s=0; s<_ptcls->getNumElements(); ++s )
                seedsData[s] = ( float(rand()) / RAND_MAX );
        }

        if( !_timer.valid() )
        {
            _timer = new osgCuda::Timer;
            _timer->setName( "EmitPtcls");
        }

        _timer->start();

        emit(
            _ptcls->getNumElements(),
            _ptcls->map( osgCompute::MAP_DEVICE_TARGET ),
            _seeds->map( osgCompute::MAP_DEVICE_SOURCE ),
            (unsigned int)(rand()),
            _min,
            _max  );

        _timer->stop();
    }

    virtual void acceptResource( osgCompute::Resource& resource )
    {
        if( resource.isIdentifiedBy("PARTICLE BUFFER" ) )
            _ptcls = dynamic_cast<osgCompute::Memory*>( &resource );
    }

private:
    osg::Vec3f                                        _max;
    osg::Vec3f                                        _min;

    osg::ref_ptr<osgCuda::Timer>                      _timer;
    osg::ref_ptr<osgCompute::Memory>                  _ptcls;
    osg::ref_ptr<osgCompute::Memory>                  _seeds;
};



////////////////////////
// PARTICLE OPERATION //
////////////////////////
class particleDataType : public osg::Referenced
{
public:
    particleDataType(osg::Node*n, osg::ref_ptr<osgCompute::Memory> ptcls, osg::Vec3f bbmin, osg::Vec3f bbmax );

    osg::ref_ptr<osgCompute::Computation> _emit;
    osg::ref_ptr<MovePtcls> _move;
};

particleDataType::particleDataType(osg::Node* n, osg::ref_ptr<osgCompute::Memory> ptcls, osg::Vec3f bbmin, osg::Vec3f bbmax )
{
    //    setKeep( true );

    _move = new MovePtcls();
    _move->acceptResource( *ptcls );

    _emit = new EmitPtcls(bbmin,bbmax);
    _emit->acceptResource( *ptcls );
}

class particleNodeCallback : public osg::NodeCallback
{
public:
   virtual void operator()(osg::Node* node, osg::NodeVisitor* nv)
   {
      osg::ref_ptr<particleDataType> particleData =
         dynamic_cast<particleDataType*> (node->getUserData() );
      if(particleData)
      {
          //particleData->updateTurretRotation();
          //particleData->updateGunElevation();
          if( osgCompute::GLMemory::getContext() == NULL || osgCompute::GLMemory::getContext()->getState() == NULL ) {
              return;
          }

          if ( !particleData->_move->isFirstFrame() ) {
              particleData->_move->setFrameStamp( nv->getFrameStamp() );
          }

          particleData->_emit->launch();
          particleData->_move->launch();

      }
      traverse(node, nv);
   }
};

class ParticleNode : public osg::Group {
public:
    ParticleNode( StimulusInterface& rsrc, osg::Vec3 bbmin_, osg::Vec3 bbmax_, osg::Vec3 color);
    virtual void setVelocity( const osg::Vec3& v );
private:
    particleDataType* _pd;
};

ParticleNode::ParticleNode( StimulusInterface& rsrc, osg::Vec3 bbmin, osg::Vec3 bbmax, osg::Vec3 color){
    //std::string textureFile = get_plugin_data_path("blackstar.png");

    /////////////////////
    // PARTICLE BUFFER //
    /////////////////////
    unsigned int numPtcls = 50000;
    osg::ref_ptr<osgCuda::Geometry> geom = new osgCuda::Geometry;
    geom->setName("Particles");
    geom->addIdentifier( "PARTICLE BUFFER" );
    osg::Vec4Array* coords = new osg::Vec4Array(numPtcls);
    for( unsigned int v=0; v<coords->size(); ++v )
        (*coords)[v].set(-10000,-10000,-10000,0); // large value so they get reset
    geom->setVertexArray(coords);
    geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POINTS,0,coords->size()));

    osg::ref_ptr<osg::Geode> geode = new osg::Geode;
    geode->addDrawable( geom.get() );

    osg::ref_ptr<osg::Program> program = new osg::Program;

    osg::Shader* StarFieldVertObj = new osg::Shader(osg::Shader::VERTEX );
    osg::Shader* StarFieldFragObj = new osg::Shader( osg::Shader::FRAGMENT );
    rsrc.load_shader_source( StarFieldVertObj, "starfield.vert" );
    rsrc.load_shader_source( StarFieldFragObj, "starfield.frag" );

    osg::Vec4 f4 = rsrc.get_clear_color();
    osg::Vec3 fog_color = osg::Vec3( f4[0], f4[1], f4[2] );

    program->addShader( StarFieldVertObj );
    program->addShader( StarFieldFragObj );
    geode->getOrCreateStateSet()->setAttribute(program);
    geode->getOrCreateStateSet()->setMode(GL_VERTEX_PROGRAM_POINT_SIZE, osg::StateAttribute::ON);
    geode->getOrCreateStateSet()->setTextureAttributeAndModes(0, new osg::PointSprite, osg::StateAttribute::ON);
    geode->getOrCreateStateSet()->setAttribute( new osg::AlphaFunc( osg::AlphaFunc::GREATER, 0.1f) );
    geode->getOrCreateStateSet()->setMode( GL_ALPHA_TEST, GL_TRUE );
    geode->getOrCreateStateSet()->addUniform( new osg::Uniform( "pixelsize", 50.0f ) );
    geode->getOrCreateStateSet()->addUniform( new osg::Uniform( "color", color ) );
    geode->getOrCreateStateSet()->addUniform( new osg::Uniform( "fog_color", fog_color));
    geode->setCullingActive( false );

    this->addChild( geode.get() );

    _pd = new particleDataType(this, geom->getMemory(), bbmin, bbmax );
    this->setUserData(_pd);
    osg::ref_ptr<particleNodeCallback> pcb = new particleNodeCallback();
    this->setUpdateCallback(pcb);
}

void ParticleNode::setVelocity( const osg::Vec3& v ) {
    if (_pd->_move) {
        _pd->_move->setVelocity(v);
    }
}

class StimulusCUDAStarFieldAndModel: public StimulusInterface
{
public:
    StimulusCUDAStarFieldAndModel();

    std::string name() const { return "StimulusCUDAStarFieldAndModel"; }
    void post_init();

    osg::ref_ptr<osg::Group> get_3d_world() {return _group; }

    virtual osg::Vec4 get_clear_color() const;

    std::vector<std::string> get_topic_names() const;
    void receive_json_message(const std::string& topic_name, const std::string& json_message);
    std::string get_message_type(const std::string& topic_name) const;

    void setVelocity(double x, double y, double z);
    void _load_stimulus_filename( std::string osg_filename );
    void _update_pat();

private:
    osg::ref_ptr<osg::Group> _group;
    osg::ref_ptr<osg::PositionAttitudeTransform> switch_node;
    osg::Vec3 model_position;
    osg::Quat model_attitude;

    osg::ref_ptr<ParticleNode> pn_black;
    osg::ref_ptr<ParticleNode> pn_white;

    osg::Vec3f bbmin;
    osg::Vec3f bbmax;

};

StimulusCUDAStarFieldAndModel::StimulusCUDAStarFieldAndModel() {
    vros_assert( is_CUDA_available()==true );

    _group = new osg::Group;
    switch_node = new osg::PositionAttitudeTransform;
    _update_pat();
    _group->addChild(switch_node);

    bbmin = osg::Vec3f(-10,-10,-10);
    bbmax = osg::Vec3f(10,10,10);

}

void StimulusCUDAStarFieldAndModel::_update_pat() {
    vros_assert(switch_node.valid());
    switch_node->setPosition( model_position );
    switch_node->setAttitude( model_attitude );
}

void StimulusCUDAStarFieldAndModel::_load_stimulus_filename( std::string osg_filename ) {

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
    vros_assert(tmp!=NULL);
    switch_node->addChild( tmp );
    _group->addChild(switch_node);
}

void StimulusCUDAStarFieldAndModel::post_init() {
    std::string osg_filename = get_plugin_data_path("post.osg");
    _load_stimulus_filename( osg_filename );

    pn_black = new ParticleNode(*this,bbmin,bbmax, osg::Vec3(0,0,0));
    pn_white = new ParticleNode(*this,bbmin,bbmax, osg::Vec3(1,1,1));

    _group->addChild( pn_black.get() );
    _group->addChild( pn_white.get() );
    setVelocity( 0.0, 0.0, 0.0);

    /////////////////////////
    // CREATE BOUNDING BOX //
    /////////////////////////

    // XXX FIXME. This is somehow required to fix opengl state after
    // drawing OSG model.

    osg::Geode* bbox = new osg::Geode;
    osg::ShapeDrawable* sd = new osg::ShapeDrawable(new osg::Box((bbmin + bbmax) * 0.5f,bbmax.x() - bbmin.x(),bbmax.y() - bbmin.y(),bbmax.z() - bbmin.z()),new osg::TessellationHints());
    sd->setColor(get_clear_color());
    bbox->addDrawable(sd);
    bbox->getOrCreateStateSet()->setMode( GL_LIGHTING, osg::StateAttribute::OFF );
    bbox->getOrCreateStateSet()->setAttribute( new osg::PolygonMode(osg::PolygonMode::FRONT_AND_BACK,osg::PolygonMode::LINE));

    _group->addChild( bbox );

    _group->setName("StimulusCUDAStarFieldAndModel._group");
}

osg::Vec4 StimulusCUDAStarFieldAndModel::get_clear_color() const {
    return osg::Vec4(0.5,0.5,0.5,1); // gray
}

std::vector<std::string> StimulusCUDAStarFieldAndModel::get_topic_names() const {
    std::vector<std::string> result;
    result.push_back("velocity");
    result.push_back("model_pose");
    return result;
}

osg::Vec3 parse_vec3(json_t* root) {
    json_t *data_json;
    double x,y,z;

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

    return osg::Vec3(x,y,z);
}

osg::Quat parse_quat(json_t* root) {
    json_t *data_json;
    double x,y,z,w;

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

    data_json = json_object_get(root, "w");
    vros_assert(data_json != NULL);
    vros_assert(json_is_number(data_json));
    w = json_number_value( data_json );

    return osg::Quat(x,y,z,w);
}

void StimulusCUDAStarFieldAndModel::receive_json_message(const std::string& topic_name,
                                             const std::string& json_message) {
    json_t *root;
    json_error_t error;

    root = json_loads(json_message.c_str(), 0, &error);
    vros_assert(root != NULL);

    if (topic_name=="velocity") {
        osg::Vec3 vel = parse_vec3(root);
        setVelocity(vel[0],vel[1],vel[2]);
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
}

void StimulusCUDAStarFieldAndModel::setVelocity(double x, double y, double z) {
    osg::Vec3 v = osg::Vec3(x,y,z);
    if (pn_black) {
        pn_black->setVelocity(v);
    }
    if (pn_white) {
        pn_white->setVelocity(v);
    }
}

std::string StimulusCUDAStarFieldAndModel::get_message_type(const std::string& topic_name) const {
    std::string result;

    if (topic_name=="velocity") {
        result = "geometry_msgs/Vector3";
    } else if (topic_name=="model_pose") {
        result = "geometry_msgs/Pose";
    } else {
        throw std::runtime_error("unknown topic name");
    }
    return result;
}

POCO_BEGIN_MANIFEST(StimulusInterface)
POCO_EXPORT_CLASS(StimulusCUDAStarFieldAndModel)
POCO_END_MANIFEST

void pocoInitializeLibrary()
{
}

void pocoUninitializeLibrary()
{
}
