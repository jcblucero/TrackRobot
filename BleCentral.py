"""
BLE Central device that will connect to android phone app (periphal)
Periphal must send specific PIDTUNER_UUID

See https://elinux.org/RPi_Bluetooth_LE for info on raspberrypi
"""

from bluepy import btle
import time

PIDTUNER_UUID = u'c83bb2d6757e47ae8947003fefbeadde'
PID_UUID_STR = 'deadbeef-3f00-4789-ae47-7e75d6b23bc8'#PIDTUNER_UUID.decode('utf-8')
print(PID_UUID_STR)
PID_UUID = btle.UUID(PID_UUID_STR)
print(PID_UUID)
print(PID_UUID.binVal)
print(PID_UUID.getCommonName())

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


        
if __name__ == "__main__":
    scanner = btle.Scanner().withDelegate(ScanDelegate())
    devices = scanner.scan(5.0)


    periph = None
    pid_found = False
    for device in devices:
        if is_device_pidtuner(device):
            print("Connecting to " + device.addr)
            periph = btle.Peripheral( device )
            print(periph.getServices())
            pid_found = True
            break


    if pid_found == True:

        for service in periph.getServices():
            print("Service Found: {}".format(service.uuid),service.uuid.binVal)
            print(service.uuid.getCommonName())
        #time.sleep(5)

        

        print("Disconnecting...")
        periph.disconnect()

    


