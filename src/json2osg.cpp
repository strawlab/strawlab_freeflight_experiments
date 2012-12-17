#include "json2osg.hpp"
#include "vros_display/vros_assert.h"

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
