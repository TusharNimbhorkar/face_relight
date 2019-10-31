import numpy as np
from sintel_io import flow_read


u,v = flow_read('flow_bw_0005.flo')
h =u.shape[0]
w =u.shape[1]

X, Y = np.meshgrid(np.arange(0, w), np.arange(0, h))
stacked = np.dstack((X, Y))
stacked_flo = np.dstack((np.multiply(-1.0,u), np.multiply(-1.0,v)))
add_both = stacked + stacked_flo
rounded_add = np.around(add_both)
# rounded_add.astype(int)
# rounded_add.astype(int)[0,0,:]
print()

X1 = stacked[:, :, 0]
Y1 = stacked[:, :, 1]

X2 = rounded_add[:, :, 0]
Y2 = rounded_add[:, :, 1]

# f = open('abc.txt')
final_out = []
count = w*h

for i in range(h):
    for j in range(w):
        if int(X2[i, j]) >=0  and  int(Y2[i, j]) >=0 and int(X2[i, j]) < w  and  int(Y2[i, j]) < h:
            temp_1 = str(int(X1[i, j])) +' '+ str(int(Y1[i, j])) +' '+ str(int(X2[i, j])) +' '+ str(int(Y2[i, j]))
            if temp_1 in final_out:
                do = 0
            else:
                final_out.append(temp_1)
        print(count)
        count=count-1







thefile2 = open('hd_sin.constraints', 'w')
thefile2.write("%s\n" % str(len(final_out)))
for item in final_out:
    thefile2.write("%s\n" % item)
thefile2.close()

