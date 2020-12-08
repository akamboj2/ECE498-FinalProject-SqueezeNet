f = open("ram.txt",'w')
for i in range(2**11):
    f.write(str(hex(i))[2:]+"\n")
