addpath('fly_trajectories');

start_row = 1; % second row
start_col = 0; % first column

data = csvread('9b97392ebb1611e2a7e46c626d3a008a_9.csv',start_row,start_col);

frameno_col = 1; % frame number of the tracking cameras, run with precisely 100Hz -> every 10ms one frame
ratio_col = 2; % value between 0..1 and represents how far along the infinity path the fly is (it can circle many times)
rotation_rate_col = 3; % angular velocity of the pattern
v_offset_rate = 4;
x_col = 5; % from Kalman filter
y_col = 6;
z_col = 7;
vx_col = 8; % finite difference quotient of position
vy_col = 9;
vz_col = 10;
vel_col = 11; % sqrt(vx^2 + vy^2)
ax_col = 12; % finite difference quotient of velocity
ay_col = 13;
az_col = 14;
theta_col = 15; % heading orientation, atan2(vy,vx)
dtheta_col = 16; % finite difference of theta
radius_col = 17; % sqrt(x^2 + y^2)
omega_col = 18;  % vel*cos(theta)/radius
rcurve_col = 19; % radius of curvature of the turn the fly might be making, a least squares minimisation fitting a circle every 4 points


Ts_cam = 0.01; % sample time of tracking cameras
Ts_contr = 1/80; % control loop runs with 80Hz

plot_start = 1;
plot_end = size(data,1);
%plot_end = size(data,1);

frameno_val = data(plot_start:plot_end,frameno_col);
ratio_val = data(plot_start:plot_end,ratio_col);
rotation_rate_val = data(plot_start:plot_end,rotation_rate_col);

x_val = data(plot_start:plot_end,x_col);
y_val = data(plot_start:plot_end,y_col);
z_val = data(plot_start:plot_end,z_col);

vx_val = data(plot_start:plot_end,vx_col);
vy_val = data(plot_start:plot_end,vy_col);
vz_val = data(plot_start:plot_end,vz_col);

ax_val = data(plot_start:plot_end,ax_col);
ay_val = data(plot_start:plot_end,ay_col);
az_val = data(plot_start:plot_end,az_col);

vel_val = data(plot_start:plot_end,vel_col);

theta_val = data(plot_start:plot_end,theta_col);
dtheta_val = data(plot_start:plot_end,dtheta_col);
radius_val = data(plot_start:plot_end,radius_col);

rcurve_val = data(plot_start:plot_end,rcurve_col); 

radius_val_repr = sqrt(x_val.^2 + y_val.^2); % that's the way how radius_val is calculated
vel_val_repr = sqrt(vx_val.^2 + vy_val.^2); % that's the way how vel_val is calculated

% Time-base
texp = (frameno_val-frameno_val(1))*Ts_cam;



