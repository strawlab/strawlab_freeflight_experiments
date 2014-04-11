import serial
import string
import struct

def _crc16(crc, c):
    crc ^= ord(c)
    for i in range(8):
        if crc & 0x1:
            crc = (crc >> 1) ^ 0xA001
        else:
            crc = (crc >> 1)
    return crc

def _printable(c):
    return (ord(c) > 32) and (ord(c) < 127)

def crc16(s):
    crc = 0
    for c in s:
        crc = _crc16(crc,c)
    return crc

def crc16_string(s):
    crc = crc16(s)
    return s + chr(crc & 0xFF) + chr(crc >> 8)

class ReadError(Exception):
    pass

class OptoTuneLensDriver(object):

    def __init__(self, port='/dev/ttyACM0', debug=True):
        self._ser = None
        self._debug = debug

        self._struct_byte = struct.Struct('>H')

        self.connect(port)

    def _send(self, s):
        if self._debug:
            for x in s:
                print '--> %c 0x%X' % (x if _printable(x) else ' ', ord(x))
        self._ser.write(s)

    def _read(self, expect_reply=True):
        if not expect_reply:
            self._ser.flushInput()
            return

        s = self._ser.readline()
        if not s:
            raise Exception("Nothing read")
        for x in s:
            print '<-- %c 0x%X' % (x if _printable(x) else ' ',ord(x))
        #dont return the CRC bytes, they are junk (not implemented)
        return s[:-2]

    def close(self):
        if self._ser and self._ser.isOpen():
            self._ser.close()

    def connect(self, port):
        self.close()

        self._ser = serial.Serial(port=port,
                            timeout=0.2,
                            baudrate=115200)
        self._ser.open()
        self._ser.flushInput()
        self._ser.flushOutput()

    def is_connected(self):
        if self._ser and self._ser.isOpen():
            self._ser.write('Start\r\n')
            c = self._ser.readline()
            return c == 'Ready\r\n'
        return False

    def get_temperature(self):
        self._send(crc16_string('TA'))
        res = self._read()
        if res[0:3] != 'TA\x00':
            raise ReadError()
        return self._struct_byte.unpack(res[3:5])[0] * 0.0625

    def set_current(self, v):
        self._send(crc16_string(
            'A' + self._struct_byte.pack(v)))
        self._read(expect_reply=False)

if __name__ == "__main__":
    #set current channel=a value=1202
    s = '\x41\x04\xb2'
    assert len(s) == 3
    s = crc16_string(s)
    assert len(s) == 5
    for x in s:
        print "0x%X" % ord(x)
    assert s == '\x41\x04\xb2\xd2\xa1'

    o = OptoTuneLensDriver()
    try:
        print "connected", o.is_connected()
        t = o.get_temperature()
        print "temp =", t
        o.set_current(0)
    finally:
        o.close()

    
