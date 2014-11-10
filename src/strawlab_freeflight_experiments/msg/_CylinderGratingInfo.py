"""autogenerated by genmsg_py from CylinderGratingInfo.msg. Do not edit."""
import roslib.message
import struct


class CylinderGratingInfo(roslib.message.Message):
  _md5sum = "33cc2b3577994aec59fd718fd0be13fe"
  _type = "strawlab_freeflight_experiments/CylinderGratingInfo"
  _has_header = False #flag to mark the presence of a Header object
  _full_text = """bool reset_phase_position # True to use 'phase_position', False to ignore
float32 phase_position    # phase of grating (radians)
float32 phase_velocity    # velocity of grating phase (radians per second)
float32 wavelength        # spatial wavelength of grating (radians)
float32 contrast          # Michelson contrast of grating
float32 orientation       # orientation of grating (radians)

"""
  __slots__ = ['reset_phase_position','phase_position','phase_velocity','wavelength','contrast','orientation']
  _slot_types = ['bool','float32','float32','float32','float32','float32']

  def __init__(self, *args, **kwds):
    """
    Constructor. Any message fields that are implicitly/explicitly
    set to None will be assigned a default value. The recommend
    use is keyword arguments as this is more robust to future message
    changes.  You cannot mix in-order arguments and keyword arguments.
    
    The available fields are:
       reset_phase_position,phase_position,phase_velocity,wavelength,contrast,orientation
    
    @param args: complete set of field values, in .msg order
    @param kwds: use keyword arguments corresponding to message field names
    to set specific fields. 
    """
    if args or kwds:
      super(CylinderGratingInfo, self).__init__(*args, **kwds)
      #message fields cannot be None, assign default values for those that are
      if self.reset_phase_position is None:
        self.reset_phase_position = False
      if self.phase_position is None:
        self.phase_position = 0.
      if self.phase_velocity is None:
        self.phase_velocity = 0.
      if self.wavelength is None:
        self.wavelength = 0.
      if self.contrast is None:
        self.contrast = 0.
      if self.orientation is None:
        self.orientation = 0.
    else:
      self.reset_phase_position = False
      self.phase_position = 0.
      self.phase_velocity = 0.
      self.wavelength = 0.
      self.contrast = 0.
      self.orientation = 0.

  def _get_types(self):
    """
    internal API method
    """
    return self._slot_types

  def serialize(self, buff):
    """
    serialize message into buffer
    @param buff: buffer
    @type  buff: StringIO
    """
    try:
      _x = self
      buff.write(_struct_B5f.pack(_x.reset_phase_position, _x.phase_position, _x.phase_velocity, _x.wavelength, _x.contrast, _x.orientation))
    except struct.error as se: self._check_types(se)
    except TypeError as te: self._check_types(te)

  def deserialize(self, str):
    """
    unpack serialized message in str into this message instance
    @param str: byte array of serialized message
    @type  str: str
    """
    try:
      end = 0
      _x = self
      start = end
      end += 21
      (_x.reset_phase_position, _x.phase_position, _x.phase_velocity, _x.wavelength, _x.contrast, _x.orientation,) = _struct_B5f.unpack(str[start:end])
      self.reset_phase_position = bool(self.reset_phase_position)
      return self
    except struct.error as e:
      raise roslib.message.DeserializationError(e) #most likely buffer underfill


  def serialize_numpy(self, buff, numpy):
    """
    serialize message with numpy array types into buffer
    @param buff: buffer
    @type  buff: StringIO
    @param numpy: numpy python module
    @type  numpy module
    """
    try:
      _x = self
      buff.write(_struct_B5f.pack(_x.reset_phase_position, _x.phase_position, _x.phase_velocity, _x.wavelength, _x.contrast, _x.orientation))
    except struct.error as se: self._check_types(se)
    except TypeError as te: self._check_types(te)

  def deserialize_numpy(self, str, numpy):
    """
    unpack serialized message in str into this message instance using numpy for array types
    @param str: byte array of serialized message
    @type  str: str
    @param numpy: numpy python module
    @type  numpy: module
    """
    try:
      end = 0
      _x = self
      start = end
      end += 21
      (_x.reset_phase_position, _x.phase_position, _x.phase_velocity, _x.wavelength, _x.contrast, _x.orientation,) = _struct_B5f.unpack(str[start:end])
      self.reset_phase_position = bool(self.reset_phase_position)
      return self
    except struct.error as e:
      raise roslib.message.DeserializationError(e) #most likely buffer underfill

_struct_I = roslib.message.struct_I
_struct_B5f = struct.Struct("<B5f")
