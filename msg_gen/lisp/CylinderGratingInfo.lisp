; Auto-generated. Do not edit!


(cl:in-package strawlab_freeflight_experiments-msg)


;//! \htmlinclude CylinderGratingInfo.msg.html

(cl:defclass <CylinderGratingInfo> (roslisp-msg-protocol:ros-message)
  ((reset_phase_position
    :reader reset_phase_position
    :initarg :reset_phase_position
    :type cl:boolean
    :initform cl:nil)
   (phase_position
    :reader phase_position
    :initarg :phase_position
    :type cl:float
    :initform 0.0)
   (phase_velocity
    :reader phase_velocity
    :initarg :phase_velocity
    :type cl:float
    :initform 0.0)
   (wavelength
    :reader wavelength
    :initarg :wavelength
    :type cl:float
    :initform 0.0)
   (contrast
    :reader contrast
    :initarg :contrast
    :type cl:float
    :initform 0.0)
   (orientation
    :reader orientation
    :initarg :orientation
    :type cl:float
    :initform 0.0))
)

(cl:defclass CylinderGratingInfo (<CylinderGratingInfo>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <CylinderGratingInfo>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'CylinderGratingInfo)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name strawlab_freeflight_experiments-msg:<CylinderGratingInfo> is deprecated: use strawlab_freeflight_experiments-msg:CylinderGratingInfo instead.")))

(cl:ensure-generic-function 'reset_phase_position-val :lambda-list '(m))
(cl:defmethod reset_phase_position-val ((m <CylinderGratingInfo>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader strawlab_freeflight_experiments-msg:reset_phase_position-val is deprecated.  Use strawlab_freeflight_experiments-msg:reset_phase_position instead.")
  (reset_phase_position m))

(cl:ensure-generic-function 'phase_position-val :lambda-list '(m))
(cl:defmethod phase_position-val ((m <CylinderGratingInfo>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader strawlab_freeflight_experiments-msg:phase_position-val is deprecated.  Use strawlab_freeflight_experiments-msg:phase_position instead.")
  (phase_position m))

(cl:ensure-generic-function 'phase_velocity-val :lambda-list '(m))
(cl:defmethod phase_velocity-val ((m <CylinderGratingInfo>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader strawlab_freeflight_experiments-msg:phase_velocity-val is deprecated.  Use strawlab_freeflight_experiments-msg:phase_velocity instead.")
  (phase_velocity m))

(cl:ensure-generic-function 'wavelength-val :lambda-list '(m))
(cl:defmethod wavelength-val ((m <CylinderGratingInfo>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader strawlab_freeflight_experiments-msg:wavelength-val is deprecated.  Use strawlab_freeflight_experiments-msg:wavelength instead.")
  (wavelength m))

(cl:ensure-generic-function 'contrast-val :lambda-list '(m))
(cl:defmethod contrast-val ((m <CylinderGratingInfo>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader strawlab_freeflight_experiments-msg:contrast-val is deprecated.  Use strawlab_freeflight_experiments-msg:contrast instead.")
  (contrast m))

(cl:ensure-generic-function 'orientation-val :lambda-list '(m))
(cl:defmethod orientation-val ((m <CylinderGratingInfo>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader strawlab_freeflight_experiments-msg:orientation-val is deprecated.  Use strawlab_freeflight_experiments-msg:orientation instead.")
  (orientation m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <CylinderGratingInfo>) ostream)
  "Serializes a message object of type '<CylinderGratingInfo>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'reset_phase_position) 1 0)) ostream)
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'phase_position))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'phase_velocity))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'wavelength))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'contrast))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'orientation))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <CylinderGratingInfo>) istream)
  "Deserializes a message object of type '<CylinderGratingInfo>"
    (cl:setf (cl:slot-value msg 'reset_phase_position) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'phase_position) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'phase_velocity) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'wavelength) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'contrast) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'orientation) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<CylinderGratingInfo>)))
  "Returns string type for a message object of type '<CylinderGratingInfo>"
  "strawlab_freeflight_experiments/CylinderGratingInfo")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'CylinderGratingInfo)))
  "Returns string type for a message object of type 'CylinderGratingInfo"
  "strawlab_freeflight_experiments/CylinderGratingInfo")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<CylinderGratingInfo>)))
  "Returns md5sum for a message object of type '<CylinderGratingInfo>"
  "33cc2b3577994aec59fd718fd0be13fe")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'CylinderGratingInfo)))
  "Returns md5sum for a message object of type 'CylinderGratingInfo"
  "33cc2b3577994aec59fd718fd0be13fe")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<CylinderGratingInfo>)))
  "Returns full string definition for message of type '<CylinderGratingInfo>"
  (cl:format cl:nil "bool reset_phase_position # True to use 'phase_position', False to ignore~%float32 phase_position    # phase of grating (radians)~%float32 phase_velocity    # velocity of grating phase (radians per second)~%float32 wavelength        # spatial wavelength of grating (radians)~%float32 contrast          # Michelson contrast of grating~%float32 orientation       # orientation of grating (radians)~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'CylinderGratingInfo)))
  "Returns full string definition for message of type 'CylinderGratingInfo"
  (cl:format cl:nil "bool reset_phase_position # True to use 'phase_position', False to ignore~%float32 phase_position    # phase of grating (radians)~%float32 phase_velocity    # velocity of grating phase (radians per second)~%float32 wavelength        # spatial wavelength of grating (radians)~%float32 contrast          # Michelson contrast of grating~%float32 orientation       # orientation of grating (radians)~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <CylinderGratingInfo>))
  (cl:+ 0
     1
     4
     4
     4
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <CylinderGratingInfo>))
  "Converts a ROS message object to a list"
  (cl:list 'CylinderGratingInfo
    (cl:cons ':reset_phase_position (reset_phase_position msg))
    (cl:cons ':phase_position (phase_position msg))
    (cl:cons ':phase_velocity (phase_velocity msg))
    (cl:cons ':wavelength (wavelength msg))
    (cl:cons ':contrast (contrast msg))
    (cl:cons ':orientation (orientation msg))
))
