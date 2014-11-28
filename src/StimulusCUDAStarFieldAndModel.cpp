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

#include <jansson.h>

/*

 See also PtclKernels.cu for the particle kernels this code uses.

 */

//////////////////
// COMPUTATIONS //
//////////////////
extern "C" void move(
                     unsigned int numPtcls,
                     void* ptcls,
                     float dx, float dy, float dz,
                     float rot_mat_00, float rot_mat_01,
                     float rot_mat_10, float rot_mat_11,
                     float centerx, float centery);

class MovePtcls : public osgCompute::Computation
{
public:
    MovePtcls() {}

    virtual void setFrameStamp( const osg::FrameStamp* fs) { _fs=fs; }
    virtual void launch()
    {
        if( !_ptcls.valid() )
            return;

        if( !_fs.valid() )
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

        float rot_mat_00, rot_mat_01, rot_mat_10, rot_mat_11;

        // Rotation is just about Z axis - we don't care about
        // observer Z or complex 3D rotation.
        float centerx = _observer_pos[0];
        float centery = _observer_pos[1];
        osg::Vec3 dpos = _vel*elapsedtime;
        float psi = _rotation_rate*elapsedtime;
        rot_mat_00 = cosf(psi);   rot_mat_01 = -sinf(psi);
        rot_mat_10 = sinf(psi);   rot_mat_11 =  cosf(psi);

        move(
            _ptcls->getNumElements(),
            _ptcls->map( osgCompute::MAP_DEVICE_TARGET ),
            dpos[0], dpos[1], dpos[2],
            rot_mat_00, rot_mat_01,
            rot_mat_10, rot_mat_11,
            centerx, centery);

        _timer->stop();
    }

    virtual void setVelocity( const osg::Vec3& v ) {
        _vel = v;
    }

    virtual void setRotationRate( const double& rate ) {
        _rotation_rate = rate;
    }

    virtual void setObserverPosition( const osg::Vec3& pos ) {
        _observer_pos = pos;
    }

    virtual void acceptResource( osgCompute::Resource& resource )
    {
        if( resource.isIdentifiedBy("PARTICLE BUFFER" ) )
            _ptcls = dynamic_cast<osgCompute::Memory*>( &resource );
    }

private:
    osg::ref_ptr<osgCuda::Timer>        _timer;
    double                              _lastTime;
    bool						        _firstFrame;
    osg::ref_ptr<const osg::FrameStamp>       _fs;
    osg::ref_ptr<osgCompute::Memory>    _ptcls;
    osg::Vec3 _vel;
    double _rotation_rate;
    osg::Vec3 _observer_pos;
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
          if( osgCompute::GLMemory::getContext() == NULL || osgCompute::GLMemory::getContext()->getState() == NULL ) {
              return;
          }

          particleData->_move->setFrameStamp( nv->getFrameStamp() );

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
    virtual void setRotationRate( const double& rate );
    virtual void setObserverPosition( const osg::Vec3& v );
    virtual void setPixelSize( float size );
private:
    particleDataType* _pd;
    osg::ref_ptr<osg::Uniform> _pixelsize;
};

ParticleNode::ParticleNode( StimulusInterface& rsrc, osg::Vec3 bbmin, osg::Vec3 bbmax, osg::Vec3 color){
    /////////////////////
    // PARTICLE BUFFER //
    /////////////////////
    unsigned int numPtcls = 5000;
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

    //osg::Vec4 f4 = rsrc.get_clear_color();
    //osg::Vec3 fog_color = osg::Vec3( f4[0], f4[1], f4[2] );

    program->addShader( StarFieldVertObj );
    program->addShader( StarFieldFragObj );
    geode->getOrCreateStateSet()->setAttribute(program);
    geode->getOrCreateStateSet()->setMode(GL_VERTEX_PROGRAM_POINT_SIZE, osg::StateAttribute::ON);
    geode->getOrCreateStateSet()->setTextureAttributeAndModes(0, new osg::PointSprite, osg::StateAttribute::ON);
    geode->getOrCreateStateSet()->setAttribute( new osg::AlphaFunc( osg::AlphaFunc::GREATER, 0.1f) );
    geode->getOrCreateStateSet()->setMode( GL_ALPHA_TEST, GL_TRUE );
    _pixelsize = new osg::Uniform( "pixelsize", 101.0f );
    geode->getOrCreateStateSet()->addUniform( _pixelsize );
    geode->getOrCreateStateSet()->addUniform( new osg::Uniform( "color", color ) );
    //geode->getOrCreateStateSet()->addUniform( new osg::Uniform( "fog_color", fog_color));
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

void ParticleNode::setRotationRate( const double& rate ) {
    if (_pd->_move) {
        _pd->_move->setRotationRate(rate);
    }
}

void ParticleNode::setObserverPosition( const osg::Vec3& v ) {
    if (_pd->_move) {
        _pd->_move->setObserverPosition(v);
    }
}

void ParticleNode::setPixelSize( float size ) {
    _pixelsize->set(size);
}

class StimulusCUDAStarFieldAndModel: public StimulusInterface
{
public:
    StimulusCUDAStarFieldAndModel();

    std::string name() const { return "StimulusCUDAStarFieldAndModel"; }
    void post_init(bool slave);

    osg::ref_ptr<osg::Group> get_3d_world() {return _group; }

    virtual osg::Vec4 get_clear_color() const;

    void update( const double& time, const osg::Vec3& observer_position, const osg::Quat& observer_orientation );

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

    osg::ref_ptr<ParticleNode> pn_white;

    osg::Vec3f bbmin;
    osg::Vec3f bbmax;

};

StimulusCUDAStarFieldAndModel::StimulusCUDAStarFieldAndModel() {
    flyvr_assert( is_CUDA_available()==true );

    _group = new osg::Group;
    switch_node = new osg::PositionAttitudeTransform;
    _update_pat();
    _group->addChild(switch_node);

    bbmin = osg::Vec3f(-10,-10,-10);
    bbmax = osg::Vec3f(10,10,10);

}

void StimulusCUDAStarFieldAndModel::_update_pat() {
    flyvr_assert(switch_node.valid());
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
    osg::Node* tmp = load_osg_file(osg_filename);
    flyvr_assert(tmp!=NULL);
    switch_node->addChild( tmp );
    _group->addChild(switch_node);
}

void StimulusCUDAStarFieldAndModel::post_init(bool slave) {
    std::string osg_filename = "post.osg";
    _load_stimulus_filename( osg_filename );

    pn_white = new ParticleNode(*this,bbmin,bbmax, osg::Vec3(1,1,1));

    _group->addChild( pn_white.get() );

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
    return osg::Vec4(0.0,0.0,0.0,1); // black
}

void StimulusCUDAStarFieldAndModel::update( const double& time, const osg::Vec3& observer_position, const osg::Quat& observer_orientation ) {
    if (pn_white) {
          pn_white->setObserverPosition(observer_position);
    }
}

std::vector<std::string> StimulusCUDAStarFieldAndModel::get_topic_names() const {
    std::vector<std::string> result;
    result.push_back("star_velocity");
    result.push_back("star_rotation_rate");
    result.push_back("star_size");
    result.push_back("model_pose");
    return result;
}

void StimulusCUDAStarFieldAndModel::receive_json_message(const std::string& topic_name,
                                             const std::string& json_message) {
    json_t *root;
    json_error_t error;

    root = json_loads(json_message.c_str(), 0, &error);

    if (root == NULL) {
        std::ostringstream errstream;
        errstream << "ERROR: could not load JSON message \"" << json_message << "\" to topic \"" << topic_name << "\".";
        std::string errmsg = errstream.str();
        flyvr_assert_msg(false, errmsg.c_str());
    }

    if (topic_name=="star_velocity") {
        osg::Vec3 vel = parse_vec3(root);
        if (pn_white) {
            pn_white->setVelocity(vel);
        }
    } else if (topic_name=="star_rotation_rate") {
        float rate = parse_float(root);
        if (pn_white) {
            pn_white->setRotationRate(rate);
        }
    } else if (topic_name=="star_size") {
        float size = parse_float(root);
        if (pn_white) {
            pn_white->setPixelSize(size);
        }
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

std::string StimulusCUDAStarFieldAndModel::get_message_type(const std::string& topic_name) const {
    std::string result;

    if (topic_name=="star_velocity") {
        result = "geometry_msgs/Vector3";
    } else if (topic_name=="star_rotation_rate") {
        result = "std_msgs/Float32";
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
POCO_EXPORT_CLASS(StimulusCUDAStarFieldAndModel)
POCO_END_MANIFEST

void pocoInitializeLibrary()
{
}

void pocoUninitializeLibrary()
{
}
