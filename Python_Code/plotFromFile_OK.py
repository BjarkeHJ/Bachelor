import matplotlib.pyplot as plt
import csv
import numpy as np

def getData(fname): #import data from .txt file. Some datasets were recorded without temperature data and without timestamp
    with open(fname+'.txt', 'r') as f: #OBS on folder
        reader = csv.reader(f, delimiter='\t')
        lines = []
        for line in reader:
            lines.append(line)
        if len(lines[0]) == 5: #with time [s]
            xs, ys, zs, ts, Ts = np.zeros((len(lines), 1)), np.zeros((len(lines), 1)), np.zeros((len(lines), 1)), np.zeros((len(lines), 1)), np.zeros((len(lines), 1))
            for x in range(len(lines)):
                xs[x] = float(lines[x][0])
                ys[x] = float(lines[x][1])
                zs[x] = float(lines[x][2])
                ts[x] = float(lines[x][3])
                Ts[x] = float(lines[x][4])
            return np.hstack((xs,ys,zs,ts,Ts))
        if len(lines[0]) == 4: #With temperature data / without time
            xs, ys, zs, ts = np.zeros((len(lines), 1)), np.zeros((len(lines), 1)), np.zeros((len(lines), 1)), np.zeros((len(lines), 1))
            for x in range(len(lines)):
                xs[x] = float(lines[x][0])
                ys[x] = float(lines[x][1])
                zs[x] = float(lines[x][2])
                ts[x] = float(lines[x][3])
            return np.hstack((xs, ys, zs, ts))
        else: #Without temperature data / without time
            xs, ys, zs = np.zeros((len(lines), 1)), np.zeros((len(lines), 1)), np.zeros((len(lines), 1))
            for x in range(len(lines)):
                xs[x] = float(lines[x][0])
                ys[x] = float(lines[x][1])
                zs[x] = float(lines[x][2])
            return np.hstack((xs, ys, zs))

def magField(array): #Computes the magnitude of the vecorial output
    mag = [np.sqrt(array[i,0]**2.+array[i,1]**2.+array[i,2]**2.) for i in range(len(array))]
    return mag
    
def makeTimeSeries(lst): #For plotting samples
    return np.linspace(0,len(lst),len(lst))

def standardize(arr): #Z-score standardization 
    arr = np.array(arr)[:,np.newaxis]
    mean = np.mean(arr)
    std = np.std(arr)
    return (arr-mean)/std

def sphereFitCal(data):
    x, y, z = data[:,0], data[:,1], data[:,2]
    # x, y, z = [], [], []
    # for i in range(len(data)):
    #     x.append(data[i][0])
    #     y.append(data[i][1])
    #     z.append(data[i][2])
    # x = np.array(x)
    # y = np.array(y)
    # z = np.array(z)
    
    m = len(x) #Number of data points
    A = np.zeros((m,4))
    b = np.zeros((m,1))
    A[:,0], A[:,1], A[:,2], A[:,3] = x,y,z,1.
    for i in range(m):
        b[i] = x[i]**2. + y[i]**2. + z[i]**2.
        
    sol = np.linalg.solve(np.matmul(np.transpose(A),A), np.dot(np.transpose(A),b)) #Solution vector of the least squares problem [a, b, c, d]
    
    #Calculate center of sphere:
    center = np.zeros((3,1))
    for i in range(3):
        center[i] = sol[i]/2
        
    #Calculating the radius of the sphere
    r = np.sqrt(sol[3] + center[0]**2. + center[1]**2. + center[2]**2.)
    
    cal_x, cal_y, cal_z = x - center[0], y - center[1], z - center[2]
    arr = [[cal_x[i], cal_y[i], cal_z[i]] for i in range(len(cal_x))]
    return np.array(arr), r, center #Return the debiased data points, best-fit sphere radius and offset 

def ellipsoidFitCal(data):
    x, y, z = data[:,0], data[:,1], data[:,2]
    num = len(x)
    M = np.array([[x[i]**2., y[i]**2., z[i]**2., 2.*x[i]*y[i], 2.*x[i]*z[i], 2.*y[i]*z[i], 2.*x[i], 2.*y[i], 2.*z[i]] for i in range(num)]) #Quadric Surface Equation
    k = np.ones((num,1)) #RHS of LSQ
    v = np.linalg.inv(np.transpose(M)@M)@np.transpose(M)@k #LSQ solution: v = (M^T M)^-1 M^T k
    a,b,c,d,e,f,g,h,i,j = v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], np.array([-1.])
    Q = np.array([[a,d,e,g],
                  [d,b,f,h],
                  [e,f,c,i],
                  [g,h,i,j]]).reshape(4,4)
    U = Q[:3,:3]
    ghi = np.array([[Q[0][3]],[Q[1][3]],[Q[2][3]]])
    U_inv = np.linalg.inv(U)
    offset = -U_inv@ghi #Calculate offset
    xc, yc, zc = offset[0][0], offset[1][0], offset[2][0]
    
    eig, eig_v = np.linalg.eig(U)
    idx = abs(eig).argsort()
    eig = eig[idx] #Eigenvalues of U
    eig_v = eig_v[:,idx] #Eigenvectors of U
    
    #Diagonalize U = PDP^T
    P = eig_v
    D = np.diag(eig)
    radius = np.sqrt(1/eig[0])
    E = np.sqrt(D)*radius #Maybe replace factor with a true field measurement. 
    
    xt, yt, zt = [], [], []
    for i in range(num):
        vv = np.array([[x[i]-xc], [y[i]-yc], [z[i]-zc]]) #Subtract found centroid
        w = P.T@vv #Determine new coordinate bias according to the principal axis theorem
        e = E@w #Now e is a p\toint on the sphere with radius of the longest principal axis when v is a point on the ellipsoid.
        xt.append(e[0][0])
        yt.append(e[1][0])
        zt.append(e[2][0])
    arr = [[xt[i], yt[i], zt[i]] for i in range(len(xt))]
    return np.array(arr), radius

def rmseSphere(center, radius, data):
    xc, yc, zc = center
    r = radius
    x, y, z = data[:,0][:,np.newaxis], data[:,1][:,np.newaxis], data[:,2][:,np.newaxis]
    residuals = np.sqrt((x - xc)**2 + (y - yc)**2 + (z - zc)**2) - r
    rmse = np.sqrt(np.mean(np.square(residuals)))
    return rmse
#----------- Plotting Functions -----------------

def plotMagnitudes(*sets, timestamp = 0):
    fig, ax = plt.subplots()
    for set in sets:
        if timestamp == 0:
            timeseries = makeTimeSeries(set[:,0])
            ax.set_xlabel("Samples")
        else:
            timeseries = set[:,4]
            ax.set_xlabel("Time [s]")
        magnitude = magField(set)
        # filt = butter_lowpass_filter(magnitude, 0.1, 100, 1)
        ax.plot(timeseries, magnitude)
        # ax.plot(timeseries, filt)
    ax.set_title("Total Magnetic Field Strength")
    ax.set_ylabel("Field Strenght [nT]")
    plt.legend(("Set1", "Set2", "Set 3")) #Will only take as many as datasets
    plt.grid()
    plt.show()

def plotAxis(*sets, timestamp = 0):
    fig, axs = plt.subplots(3,1,sharex=True)
    for set in sets:
        if timestamp == 0:
            timeseries = makeTimeSeries(set[:,0])
        else: timeseries = set[:,4]
        x = set[:,0]
        axs[0].plot(timeseries,x)
        y = set[:,1]
        axs[1].plot(timeseries,y)
        z = set[:,2]
        axs[2].plot(timeseries,z)
    plt.show()
        
def plotDataTemp(data, timestamp = 0):
    try:
        fig, axs = plt.subplots(2,1,sharex=True)
        if timestamp == 0:
            timeseries = makeTimeSeries(data[:,0])
            plt.xlabel("Samples")
        else:
            timeseries = data[:,4]
            plt.xlabel("Time [s]")
        mag = magField(data)
        axs[0].plot(timeseries, mag)
        axs[0].set_title("Magnetic Field and Temperature")
        axs[0].set_ylabel("Field Strenght [nT]")
        axs[0].grid()
        axs[1].plot(timeseries, data[:,3])
        axs[1].set_ylabel("Temeperature [C]")
        axs[1].grid()
        plt.show()      
    except:
        print("Temperature values not included in dataset")
        return None
         
def plotTempVsData(data): #Plots data vs temperature if temperature datapoints included
    try:
        temp = data[:,3]
        mag = magField(data)
        fig, ax = plt.subplots()
        ax.scatter(temp, mag, marker='.')
        ax.grid()
        plt.xlabel("Temperature [C]")
        plt.ylabel("Field Strenght [nT]")
        plt.title("FGM output temperature dependency")
        plt.show() 
    except:
        print("Temperature values not included in dataset")
        return None
 
def plot3D(*args):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('x component')
    ax.set_ylabel('y component')
    ax.set_zlabel('z component')
    for set in args:
        ax.scatter(set[:,0], set[:,1], set[:, 2])
    plt.legend(("Set 1", "Set 2", "Set 3", "Set 4"))
    plt.show()

#------------------------------------------------------------------------------------------


if __name__ == '__main__':  
    arr1 = getData("Dataset_5/" + "ST_cal")
    
    #Calibrated data sets
    sph_cal, sph_r, sph_c = sphereFitCal(arr1)
    ell_cal, ell_r = ellipsoidFitCal(arr1)
    
    plotMagnitudes(arr1, sph_cal,ell_cal)
