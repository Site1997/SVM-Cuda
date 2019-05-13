inf=open("a9a","r")
ouf=open("a9a_flat","w")

for l in inf:
    tokens=l.split(' ')
    y=int(tokens[0])
    next=1
    for t in tokens[1:-1]:
        x=int(t.split(':')[0])
        for i in range(next,x):
            ouf.write("0 ")
        ouf.write("1 ")
        next=x+1
    for i in range(next,124):
        ouf.write("0 ")
    ouf.write(str(y))
    ouf.write("\n")

ouf.close()