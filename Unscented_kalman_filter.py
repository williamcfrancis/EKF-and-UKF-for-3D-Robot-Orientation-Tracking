import numpy as np
import matplotlib.pyplot as plt
from scipy import io
import os
import math


def estimate_rot(data_num=1):
    imu = io.loadmat('source/imu/imuRaw'+str(data_num)+'.mat')
    # imu = io.loadmat('imu/imuRaw'+str(data_num)+'.mat') 

    # vicon = io.loadmat('vicon/viconRot'+str(data_num)+'.mat')
    # viconrpy = np.zeros((np.shape(vicon['rots'])[0], np.shape(vicon['rots'])[2]))
    # vicon_ts = vicon['ts'].T
    # for i in range(0, np.shape(vicon['rots'])[2]):
    #     viconrpy[:, i] = euler_angles(from_rotm(vicon['rots'][:, :, i]))

    imu_vals = imu['vals']
    imu_ts = imu['ts']
    imu_ts = np.array(imu['ts']).T  
    T = np.shape(imu['ts'])[1]
    acc = np.array([-np.array(imu_vals[0]), -np.array(imu_vals[1]), np.array(imu_vals[2])]).T
    gyro_x = np.array(imu_vals[4]) # angular rates are out of order !
    gyro_y = np.array(imu_vals[5])
    gyro_z = np.array(imu_vals[3])
    gyro = np.array([gyro_x, gyro_y, gyro_z]).T

    # gyro_sens, gyro_bias, acc_sens, acc_bias = calibrate(acc, gyro, viconrpy, vicon_ts)

    vref = 3300
    acc_sens = 332.166 
    acc_bias = np.array([65035.0, 65035.0, 503.0]) 
    gyro_bias = np.array([374.5, 376, 370]) 
    gyro_sens =  193.55
    gyro = (gyro-gyro_bias)*(vref/(1023*gyro_sens))
    acc = (acc-acc_bias)*(vref/(1023*acc_sens))
    imu_vals = np.hstack((acc,gyro)) 

    P_rot = P_omega = 0.1 * np.identity(3)
    Q_rot = Q_omega = 2 * np.identity(3)
    R_rot = R_omega = 2 * np.identity(3)
    q_new = np.array([1, 0, 0, 0])
    omega_new = np.array([0.5, 0.5, 0.5])
    qk = q_new
    rot_cov = omega_cov = P_rot
    omega_l = omega_new

    
    for t in range(T):

        acc = imu_vals[t,:3]
        gyro = imu_vals[t,3:] 

        # =========== Calculating Xi ==============
        Xi_rot = calculate_Xi(qk, P_rot, Q_rot, 0) #12x7 -> 6x4
        Xi_omega = calculate_Xi(omega_new, P_omega, Q_omega, 1) # 6x3

        # ========== Calculating Y ================
        if t == T-1:
            dt = imu_ts[-1] - imu_ts[-2]
        else:
            dt = imu_ts[t+1] - imu_ts[t]

        rows = Xi_rot.shape[0] # 6
        Y_rot = np.zeros((rows, 4)) # 6x4
        q_delta = from_axis_angle(gyro*dt) # 4x1
        for i in range(rows):  
            Y_rot[i] = np.array(multiply(Xi_rot[i], q_delta)) # q_del * q_w * q_k

        xk_bar_rot, err = quat_average(Y_rot, qk) # 38,39 # xk_bar: 4x1, error: 12x3 
        xk_bar_omega = np.mean(Xi_omega, axis = 0)

        omega_w = Xi_omega - xk_bar_omega # 12x3
        Wi_prime = err # 12x6
        Pk_bar = np.zeros((3, 3)) # 6x6

        Pk_bar = Wi_prime.T @ Wi_prime
        Pk_bar /= 6

        Pk_bar_omega = omega_w.T @ omega_w
        Pk_bar_omega /= 6

        g = np.array([0, 0, 0, 1]) 
        Z = np.zeros((6, 3)) 
        for i in range(6):
            q = Y_rot[i] # 4x1
            Z[i] = multiply(multiply(inv(q), g), q)[1:] # Transformation of g from global coordinate to tracker coordinate (27)

        zk_bar = np.mean(Z, axis=0) #(6,)
        zk_bar = normalize(zk_bar)
        zk_bar_omega = normalize(xk_bar_omega)
        Wi_prime_omega = Xi_omega - zk_bar_omega

        Pzz_rot = Pzz_omega = np.zeros((3, 3))
        Pxz_rot = Pzz_omega = np.zeros((3, 3))
        Z_err_rot = Z - zk_bar # 12x7
    
        Pzz_rot = Z_err_rot.T @ Z_err_rot
        Pxz_rot = Wi_prime.T @ Z_err_rot
        Pzz_rot /= 6
        Pxz_rot /= 6
        Pzz_omega = Wi_prime_omega.T @ Wi_prime_omega
        Pxz_omega = omega_w.T @ Wi_prime_omega
        Pzz_omega /= 6
        Pxz_omega /= 6

        imu_norm = imu_vals[t,:3]/np.linalg.norm(imu_vals[t,:3])
        vk_rot =  imu_norm - zk_bar # 6x1 (44)
        vk_omega = imu_vals[t,3:] - zk_bar_omega
        Pvv_rot = Pzz_rot + R_rot     # 6x6 (45)
        Pvv_omega = Pzz_omega + R_omega

        K_rot = np.dot(Pxz_rot, np.linalg.inv(Pvv_rot)) # (72) 6x6
        K_omega = np.dot(Pxz_omega, np.linalg.inv(Pvv_omega))

        q_gain = from_axis_angle(K_rot.dot(vk_rot)[:3])  # (74)
        omega_new = K_omega.dot(vk_omega) + xk_bar_omega

        qk = multiply(q_gain, xk_bar_rot) 
        P_rot = Pk_bar - K_rot.dot(Pvv_rot).dot(K_rot.T) # (75)
        P_omega = Pk_bar_omega - K_omega.dot(Pvv_omega).dot(K_omega.T)
        q_new = np.vstack((q_new, qk))
        
        omega_l = np.vstack((omega_l, omega_new))
        rot_cov = np.vstack((rot_cov, np.diag(P_rot)))
        omega_cov = np.vstack((omega_cov, np.diag(P_omega)))

    roll = np.zeros(q_new.shape[0])
    pitch = np.zeros(q_new.shape[0])
    yaw = np.zeros(q_new.shape[0])

    for i in range(q_new.shape[0]):
        roll[i], pitch[i], yaw[i] = euler_angles(q_new[i])

    # plt.figure(1)
    # plt.subplot(311)
    # plt.plot(roll, 'b', label = "Estimated roll")
    # plt.plot(viconrpy[0, :], 'g--', label = "Ground truth roll")
    # plt.subplot(312)
    # plt.plot(pitch, 'b', label = "Estimated pitch")
    # plt.plot(viconrpy[1, :], 'g--', label = "Ground truth pitch")
    # plt.subplot(313)
    # plt.plot(yaw, 'b', label = "Estimated yaw")
    # plt.plot(viconrpy[2, :], 'g--', label = "Ground truth yaw")
    # plt.xlabel('Time step k')
    # plt.ylabel('Ground truth rotations')
    # plt.title('Estimated mean of quaternions')
    # plt.legend()
    # plt.savefig("ypr_meanquat_ukf.png")

    # plt.figure(2)
    # plt.plot(rot_cov[:,0], 'r', label = "Covariance roll")
    # plt.plot(rot_cov[:,1], 'g', label = "Covariance pitch") 
    # plt.plot(rot_cov[:,2], 'b', label = "Covariance yaw")
    # plt.xlabel('Time step k')
    # plt.ylabel('Covariance')
    # plt.title('Estimated covariance of quaternions')
    # plt.legend()
    # plt.savefig("covquat_ukf.png")

    # plt.figure(3)
    # plt.plot(omega_l[:,0], 'r', label = "Omega x") 
    # plt.plot(omega_l[:,1], 'g', label = "Omega y")
    # plt.plot(omega_l[:,2], 'b', label = "Omega z")
    # plt.xlabel('Time step k')
    # plt.ylabel('Angular velocity')
    # plt.title('Estimated mean of angular velocity')
    # plt.legend()
    # plt.savefig("meanomega_ukf.png")

    # plt.figure(4)
    # plt.plot(omega_cov[:,0], 'r', label = "Omega x")
    # plt.plot(omega_cov[:,1], 'g', label = "Omega y")
    # plt.plot(omega_cov[:,2], 'b', label = "Omega z")
    # plt.xlabel('Time step k')
    # plt.ylabel('Angular velocity')
    # plt.title('Estimated covariance of angular velocity')
    # plt.legend()
    # plt.savefig("covomega_ukf.png")

    return roll, pitch, yaw

def quat_average(q, qt): # q: 6x4 q0: 4x1
    r = q.shape[0] 
    maxiter = 1000
    tol = 0.0001
    for _ in range(maxiter):
        err = np.zeros((r,3))
        for i in range(r):
            qi_err = normalize(multiply(q[i, :], inv(qt))) # (52) ei = qi * qt.inv
            ev_err = axis_angle(qi_err) # 3x1
            if np.linalg.norm(ev_err) != 0: # rotate
                err[i,:] = (-np.pi + np.mod(np.linalg.norm(ev_err) + np.pi, 2 * np.pi)) / np.linalg.norm(ev_err) * ev_err
            else:
                err[i:] = np.zeros(3)
        meanerr = np.mean(err, axis=0) # (54) 
        qt = normalize(multiply(from_axis_angle(meanerr), qt)) # (55)
        if np.isclose(np.linalg.norm(meanerr), 0, atol = tol): # If achieving error below tolerance
            return qt, err

def multiply(q, r):
    t0 = q[0]*r[0] -q[1]*r[1] -q[2]*r[2] -q[3]*r[3]
    t1 = q[0]*r[1] +q[1]*r[0] +q[2]*r[3] -q[3]*r[2]
    t2 = q[0]*r[2] -q[1]*r[3] +q[2]*r[0] +q[3]*r[1]
    t3 = q[0]*r[3] +q[1]*r[2] -q[2]*r[1] +q[3]*r[0]
    return [t0, t1, t2, t3]

def inv(q): 
    t = np.array([q[0], -q[1], -q[2], -q[3]])
    return normalize(t)

def normalize(q): # checked
    return q/np.linalg.norm(q)

# euler angle refers to roll, pitch and yaw
def euler_angles(q):  # given transformation
    phi = math.atan2(2*(q[0]*q[1]+q[2]*q[3]), 1 - 2*(q[1]**2 + q[2]**2))
    theta = math.asin(2*(q[0]*q[2] - q[3]*q[1]))
    psi = math.atan2(2*(q[0]*q[3]+q[1]*q[2]), 1 - 2*(q[2]**2 + q[3]**2))
    return phi, theta, psi

def from_axis_angle(a):
    q = np.zeros(4)
    angle = np.linalg.norm(a)
    if angle != 0:
        axis = a/angle
    else:
        axis = np.array([1,0,0])
    q[0] = math.cos(angle/2)
    q[1:4] = axis*math.sin(angle/2)
    return q

def axis_angle(q):
    theta = 2*math.acos(q[0])
    vec = q[1:4]
    if (np.linalg.norm(vec) == 0):
        return np.zeros(3)
    vec = vec/np.linalg.norm(vec)
    return vec*theta

def from_rotm(R):
    q = np.zeros(4)
    theta = math.acos((np.trace(R)-1)/2)
    omega_hat = (R - np.transpose(R))/(2*math.sin(theta))
    omega = np.array([omega_hat[2,1], -omega_hat[2,0], omega_hat[1,0]])
    q[0] = math.cos(theta/2)
    q[1:4] = omega*math.sin(theta/2)
    return normalize(q)

def calibrate(acc, gyro, vicon, vicon_ts):
    vicon_ts = vicon_ts.reshape(-1)
    print(vicon.shape)
    dq = vicon[:,1:] - vicon[:,:-1]
    dt = vicon_ts[1:] - vicon_ts[:-1]
    ang_v = dq.T/dt

    return g_alpha, g_beta, a_alpha, a_beta

def calculate_Xi(qk, P, Q, action):
    if(action == 0):
        size = P.shape[0] # 3x3
        S = np.linalg.cholesky(P+Q) # 6x6
        Wi = np.hstack((S * np.sqrt(2*size), -S * np.sqrt(2*size))) # 6x12
        Xi = np.zeros((2*size, 4)) # 12x7

        for i in range(2*size): # 12
            qW = from_axis_angle(Wi[:,i]) # (4,)
            Xi[i,:] = multiply(qk, qW) 
        return Xi
    else:
        size = P.shape[0] # 3x3
        S = np.linalg.cholesky(P+Q) # 6x6
        Wi = np.hstack((S * np.sqrt(2*size), -S * np.sqrt(2*size))) # 6x12
        Xi = np.zeros((2*size, 3)) # 12x7

        for i in range(2*size): # 12
            qW = from_axis_angle(Wi[:,i]) # (4,)
            Xi[i,:] = Wi[:,i] + qk # 12x7
        return Xi

# a,b,c = estimate_rot(2)
