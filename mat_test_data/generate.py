import random
files = ["matA.bin", "matB.bin"]

for  file in files:
    with open(file, "wb") as f:
        size = 3200
        for i in range (size):
            for j in range (size):
                f.write(random.randint(1, 5).to_bytes(1, byteorder='little'))
