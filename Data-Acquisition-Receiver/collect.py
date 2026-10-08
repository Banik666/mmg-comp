import serial

SECONDARY_BUFF_SZIE = 128

ser = serial.Serial('COM4', 1000000, timeout=1)

# global array


while True:
    # listen for data packet  
    size = 2*2*SECONDARY_BUFF_SZIE
    data = b''
    while len(data) < size:
        chunk = ser.read(size - len(data))
        if not chunk: break
        data += chunk

    # unpack data packet buffer into [2] x [Secondary buff size] array
    data_array = memoryview(data).cast('H', [2, SECONDARY_BUFF_SZIE]).tolist()

    # print(data_array)

    # append data in a txt file
    with open("temp.csv", "a") as f:
        for i in range(SECONDARY_BUFF_SZIE):
            f.write(str(data_array[0][i])+",")
            f.write(str(data_array[1][i])+"\n")
    f.close()

    print("buffer is written")







