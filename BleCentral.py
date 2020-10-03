"""
BLE Central device that will connect to android phone app (periphal)
Periphal must send specific PIDTUNER_UUID

See https://elinux.org/RPi_Bluetooth_LE for info on raspberrypi
"""

from bluepy import btle
#from bluepy.btle import BTLEDisconnectError
import struct
import time
import keyboard

PIDTUNER_UUID = u'c83bb2d6757e47ae8947003fefbeadde'
PID_UUID_STR = 'deadbeef-3f00-4789-ae47-7e75d6b23bc8'#PIDTUNER_UUID.decode('utf-8')
THROTTLE_UUID = btle.UUID('deadbee1-3f00-4789-ae47-7e75d6b23bc8')
KP_UUID = btle.UUID('deadbee2-3f00-4789-ae47-7e75d6b23bc8')
KI_UUID = btle.UUID('deadbee3-3f00-4789-ae47-7e75d6b23bc8')
KD_UUID = btle.UUID('deadbee4-3f00-4789-ae47-7e75d6b23bc8')
START_STOP_UUID = btle.UUID('deadbee5-3f00-4789-ae47-7e75d6b23bc8')
CCD_UUID = btle.UUID('00002902-0000-1000-8000-00805f9b34fb')

print(PID_UUID_STR)
PID_UUID = btle.UUID(PID_UUID_STR)
print(PID_UUID)
print(PID_UUID.binVal)
print(PID_UUID.getCommonName())


pid_characteristic_uuids_list = [THROTTLE_UUID,KP_UUID,KI_UUID,KD_UUID,START_STOP_UUID]
#return true if the ScanEntry (returned from a scan) is the android PID tuner app
def is_device_pidtuner(device):

    scan_data = device.getScanData()
    for scan_data_tuple in scan_data:
        adtype,description,value = scan_data_tuple
        #print(value)
        #print(type(value))
        #print(value[0:32])
        #service ID will only be first 128 bits(16 bytes/32 hex characters)
        #service_id = value[0:32]
        service_id = value
        #print(service_id)
        if( PID_UUID.getCommonName() == value ):
            print("MATCH")
            return True

    return False
    


class ScanDelegate(btle.DefaultDelegate):

    def __init__(self):
        btle.DefaultDelegate.__init__(self)

    def handleDiscovery(self, dev, isNewDev, isNewData):
        if( isNewDev):
            print("Discovered Device ", dev.addr)
            print(dev.getScanData())
            #is_device_pidtuner(dev)

class NotificationDelegate(btle.DefaultDelegate):

    def __init__(self, notificationCallback):
        btle.DefaultDelegate.__init__(self)
        self.notificationCallback = notificationCallback

    def handleNotification(self, cHandle, data):
        #cHandle is handle to gatt characteristic which is sending notification
        #data is bytes value containing char data. struct.unpack
        print("NOTIFIED")
        self.notificationCallback(data)

class BlePidProfile:

    def __init__(self,throttle=15.0, kp=0.6, ki=0.0, kd=0.15, start_stop=0):
        self.scanner = btle.Scanner()#.withDelegate(ScaneDelegate())
        self.connected = False
        self.pid_found = False
        self.throttle_value = throttle
        self.kp_value = kp
        self.ki_value = ki
        self.kd_value = kd
        self.start_stop_value = start_stop  #0 == stop, 1 ==start

    #Converts int byte data transmitted from PID service to float value
    # Input - byte array from PID service
    # Output - float
    def UnpackFloat(self, float_bytes):
        #!i = big endian 4 byte float
        # values sent were multiplied by 1000 since kotlin doesn't support float conversion to bytes
        # now we divide by 1000 to convert back
        value = struct.unpack('!i',float_bytes)[0]
        final_value = value / 1000.0

        return final_value

    def RequestAllData(self):
        self.throttle_value = self.UnpackFloat(self.throttle_characteristic.read())
        self.kp_value = self.UnpackFloat(self.kp_characteristic.read())
        self.ki_value = self.UnpackFloat(self.ki_characteristic.read())
        self.kd_value = self.UnpackFloat(self.kd_characteristic.read())
        self.start_stop_value = struct.unpack('!i',self.start_stop_char.read())[0]
        return (self.throttle_value,self.kp_value,self.ki_value,
                self.kd_value,self.start_stop_value)     



    #Call this whenever a notification is received
    def ReceiveNotification(self, data):
        print("receive notification")
        self.start_stop_value = struct.unpack('!i',data)[0]
        self.throttle_value = self.UnpackFloat(self.throttle_characteristic.read())
        self.kp_value = self.UnpackFloat(self.kp_characteristic.read())
        self.ki_value = self.UnpackFloat(self.ki_characteristic.read())
        self.kd_value = self.UnpackFloat(self.kd_characteristic.read())
        #self.start_stop_value = struct.unpack('!i',self.start_stop_char.read())
       
        #print(self.start_stop_value, struct.unpack('!i',data))


    def WaitForNotification(self,timeout=None):
        return self.periph.waitForNotifications(timeout)

    def RequestStartStopChar(self):
        value = struct.unpack('!i',self.start_stop_char.read())[0]
        return value

    #scan time in seconds
    #connects to PID profile if found
    #returns true if found and connected, false otherwise
    def ScanForPidService(self,time):
        self.devices = self.scanner.scan(time)
        self.pid_found = False
        for device in self.devices:
            if( is_device_pidtuner(device) ):
                print("Connecting to " + device.addr)
                self.periph = btle.Peripheral(device)
                self.notification_delegate =  NotificationDelegate( self.ReceiveNotification )
                self.periph.setDelegate( self.notification_delegate )

                #self.periph.setDelegate( NotificationDelegate( PidTest ) )
                self.pid_found = True
             
        if(self.pid_found==True):
            try:
                print(self.periph)
                #Some sort of bug in bluepy
                #have to call getServices/getCharacterstics before it will get responses from service...
                self.services = self.periph.getServices()
                self.pid_service = self.periph.getServiceByUUID(PID_UUID)
            
                self.throttle_characteristic = self.pid_service.getCharacteristics(THROTTLE_UUID)[0]
                self.kp_characteristic = self.pid_service.getCharacteristics(KP_UUID)[0]
                self.ki_characteristic = self.pid_service.getCharacteristics(KI_UUID)[0]
                self.kd_characteristic = self.pid_service.getCharacteristics(KD_UUID)[0]
                self.start_stop_char = self.pid_service.getCharacteristics(START_STOP_UUID)[0]
                #print(self.start_stop_char)

                #Descriptors not implemented in bluepy library, so no way to driectly get descriptor. Must get char descriptor and assume cccd is +1 from there
                #print( (self.kp_characteristic).read() )
                #print(self.start_stop_char.read())
                #print(struct.unpack('!i',self.start_stop_char.read()))

                self.cccd_handle = self.start_stop_char.getHandle()+1
                self.notification_enable_data = b"\x01\x00"
                #print(self.cccd_handle)
                self.periph.writeCharacteristic(self.cccd_handle,self.notification_enable_data,withResponse=True)
                self.connected = True
                print("Done getting characteristics and notifications enabled") 
                #self.periph.           

            except Exception as inst:
                print(inst)

        return self.connected

def PidTest():
    pid_profile = BlePidProfile()
    if pid_profile.ScanForPidService(5.0):
        print("connected to service")
    if pid_profile.WaitForNotification(4.0):
        print("notified")
    print("sleeping...")
    print(pid_profile.throttle_value)
    print(pid_profile.start_stop_value)
    while(pid_profile.start_stop_value==1):
        print("Checking btle")
        start = time.time()
        #value = pid_profile.RequestStartStopChar()
        value = pid_profile.RequestAllData()
        end = time.time()
        print(value)
        print("Reading Time {}".format(end-start))
        #print(value,pid_profile.start_stop_value) 
        #for i in range(1000):
        #    print("Not Checking btle {}".format(i))
        


    print(pid_profile.throttle_value)
    print(pid_profile.start_stop_value)
    #time.sleep(4)
    exit(0)
        
if __name__ == "__main__":

    PidTest()
    #ScanForPidService(3.0)
    scanner = btle.Scanner().withDelegate(ScanDelegate())
    devices = scanner.scan(5.0)


    periph = None
    pid_found = False
    for device in devices:
        if is_device_pidtuner(device):
            print("Connecting to " + device.addr)
            periph = btle.Peripheral( device )
            periph.setDelegate( NotificationDelegate( PidTest ) )
            print(periph.getServices())
            pid_found = True
            break


    if pid_found == True:

        for service in periph.getServices():
            print("Service Found: {}".format(service.uuid),service.uuid.binVal)
            print(service.uuid.getCommonName())
        #time.sleep(5)

        try:
            pid_service = periph.getServiceByUUID(PID_UUID)
            characteristics = pid_service.getCharacteristics()
            for char in characteristics:
                print(char)

            
            throttle_characteristic = pid_service.getCharacteristics(THROTTLE_UUID)[0]
            kp_characteristic = pid_service.getCharacteristics(KP_UUID)[0]
            ki_characteristic = pid_service.getCharacteristics(KI_UUID)[0]
            kd_characteristic = pid_service.getCharacteristics(KD_UUID)[0]
            start_stop_char = pid_service.getCharacteristics(START_STOP_UUID)[0]

            
            #char_list = pid_service.getCharacteristics(pid_characteristic_uuids_list)
            #print("hello")
            #(throttle_characteristic,kp_characteristic,ki_characteristic,kd_characteristic,start_stop_char) = tuple(char_list)
            #print(char_list)
            print("THROTTLE ",throttle_characteristic.read())
            print(struct.unpack('!i',throttle_characteristic.read()))

            print("kp ",kp_characteristic.read())
            print(struct.unpack('!i',kp_characteristic.read()))

            print("ki ",ki_characteristic.read())
            print(struct.unpack('!i',ki_characteristic.read()))


            for descriptor in pid_service.getDescriptors():
                print(descriptor)
        
            #this is making assumption that last descriptor offered by pid is cccd
            #cccd = pid_service.getDescriptors()[-1]
            #Descriptors not implemented in bluepy library, so no way to driectly get descriptor. Must get char descriptor and assume cccd is +1 from there
            cccd_handle = start_stop_char.getHandle()+1
            notification_enable_data = b"\x01\x00"
            
            periph.writeCharacteristic(cccd_handle,notification_enable_data,withResponse=True)
            print("waiting...")
            if periph.waitForNotifications(4.0):
                print("Notification")
                #continue
    
        except Exception as inst:
            print(inst)
            #print("Error finding {}".format(PID_UUID))

        

        
        #print("waiting for q...")
        #keyboard.wait('q')
        print("Disconnecting...")
        periph.disconnect()

    


