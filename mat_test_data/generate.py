import random

with open("matB.txt", "w") as f:
    
    size = 3200
    for i in range (size):
        for j in range (size):
            f.write(str(random.randint(1, 5))+ " ")
        f.write("\n")