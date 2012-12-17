#include <jansson.h>
#include <osg/Vec3>
#include <osg/Quat>

osg::Vec3 parse_vec3(json_t* root);
osg::Quat parse_quat(json_t* root);
