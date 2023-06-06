import numpy as np
import serial
from os import path
import sys
import time

def readTimeData(minutes): 
    ser.reset_input_buffer()
    data = []
    sec = []
    fejl = 0
    time_end = time.time() + minutes * 60
    time_start = time.time()
    
    while time.time() < time_end:
        rec = ser.readline()
        raw_data = rec.decode().split(';')
        time_now = time.time() - time_start
        try:
            xyzt = raw_data[:4]
            for i in range(len(xyzt)):
                xyzt[i] = "".join([z for z in xyzt[i] if z.lstrip('-').isdigit() or z=='-' or z==',']) #Filter for only digit values and convert to ints
                xyzt[i] = float(xyzt[i].replace(',', '.'))
        except:
            print("Data was not received correctly - Skipping sample")
            fejl += 1
            continue        
        sec.append(time_now)
        data.append(xyzt)
        print("\r[", str(time.localtime((time.time() - time_start))[3:6]).replace(',',':') , " of ", time.localtime(time_end-time_start)[3:6], " recorded.]", sep="", end="", flush=True) #Print recording progress
    print("\nAntal fejlsamples: ", fejl)
    print("Antal fuldførte samples: ", len(data), "\n")
    return np.hstack((np.array(data), np.array(sec)[:,np.newaxis]))

def exportData(array, fname):
    arr = np.array(array)
    np.savetxt(fname+'.txt', arr, fmt='%f', delimiter='\t') #OBS on folder that it saves in
    print("text file saved as " + fname + ".txt")
    return

#-----------------------MAIN-------------------------

if __name__ == "__main__":
    ser = serial.Serial("COM10", 
                        baudrate = 115200, 
                        parity=serial.PARITY_NONE, 
                        stopbits=serial.STOPBITS_ONE, 
                        bytesize=serial.EIGHTBITS, 
                        timeout=2)
    
#   Change Parameters
    folder = "Dataset_5/ST_out/"
    export_name = "ST_out_085"     # Exported file name (as .txt file)
    MOS = 2        # Minutes Of Sampling
# ---------------------

    ser.reset_input_buffer()
    ser.write(b'\r')
    start = ser.readlines()
    ser.write(b'3x')
    comm = ser.readline().decode()
    ser.write(b'c')
    comm = ser.readline().decode()
    ser.reset_input_buffer()
    if path.exists(folder + export_name + '.txt') == True: #Don't delete important files
            print("File name is already in folder")
            p = str(input("Press 'y' if you want to overwrite: "))
            print(p)
            if p != 'y':
                print("Change variable 'export_name' and try again")
                sys.exit()


    A = readTimeData(MOS)
    exportData(A, folder + export_name)
    
    print("\n--DONE--\n")
    ser.reset_input_buffer()
    ser.write(b's')
    ser.close()