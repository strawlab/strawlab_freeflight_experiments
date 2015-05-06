close all;
clear all;
clc
%load all data
readData

%mechanical parameters of the fly
p.Ixx= 0.306*(10^-12);
p.Iyy= 0.5060*(10^-12);
p.Izz= 0.306*(10^-12); 

p.m_f = 0.9315*(10^-6);   % kg
p.g = 9.81;

%filter for smoothing the measurements & numeric differentiation
window_size = 25;

x_val_f = myfilter(x_val,window_size);
vx_val_f = diff(x_val_f)/Ts_cam;    vx_val_f(end+1) = vx_val_f(end);
ax_val_f = diff(vx_val_f)/Ts_cam;   ax_val_f(end+1) = ax_val_f(end);

y_val_f = myfilter(y_val,window_size);
vy_val_f = diff(y_val_f)/Ts_cam;    vy_val_f(end+1) = vy_val_f(end);
ay_val_f = diff(vy_val_f)/Ts_cam;   ay_val_f(end+1) = ay_val_f(end);

z_val_f = myfilter(z_val,window_size);
vz_val_f = diff(z_val_f)/Ts_cam;    vz_val_f(end+1) = vz_val_f(end);
az_val_f = diff(vz_val_f)/Ts_cam;   az_val_f(end+1) = az_val_f(end);

%yaw angle: fly faces in the direction it is moving
eta_val = atan2(vy_val,vx_val);
w_eta_val = (-vy_val_f .* ax_val_f + vx_val_f .* ay_val_f) ./ (vx_val_f .^ 2 + vy_val_f .^ 2);
a_eta_val = -2 .* (-vy_val_f .* ax_val_f + vx_val_f .* ay_val_f) .* (vx_val_f .* ax_val_f + vy_val_f .* ay_val_f) ./ (vx_val_f .^ 2 + vy_val_f .^ 2) .^ 2;

%get tau for full model
tau = get_tau(ax_val,ay_val,az_val,eta_val,w_eta_val,a_eta_val,p);

%get tau for the roll angle phi = 0 and constant pitch angle theta = 30�
size_meas = length(x_val_f);
theta = 30*pi/180*ones(size_meas,1);
w_theta_val = zeros(size_meas,1);
a_theta_val = zeros(size_meas,1);

tau_red = get_tau_red(ax_val,ay_val,az_val,theta_val,w_eta_val,w_theta_val,a_eta_val,a_theta_val,p);

%compare reduced and full model
figure;
subplot(611);plot(tau_red(:,1));hold on;plot(tau(:,1),'r--');
subplot(612);plot(tau_red(:,2));hold on;plot(tau(:,2),'r--');
subplot(613);plot(tau_red(:,3));hold on;plot(tau(:,3),'r--');
subplot(614);plot(tau_red(:,4));hold on;plot(tau(:,4),'r--');
subplot(615);plot(tau_red(:,5));hold on;plot(tau(:,5),'r--');
subplot(616);plot(tau_red(:,6));hold on;plot(tau(:,6),'r--');

%compare filtered and not
figure;
subplot(411);plot(x_val);hold on;plot(x_val_f,'r--');
subplot(412);plot(vx_val);hold on;plot(vx_val_f,'r--');
subplot(413);plot(ax_val);hold on;plot(ax_val_f,'r--');
subplot(414);plot(theta_val);hold on;plot(unwrap(eta_val),'r--');

