function S = Inverse_dyn(datafile, ts, window_size, calc_full_model, calc_reduced_model, save_models, plot_models)
if nargin<1
    datafile='fly_trajectories/9b97392ebb1611e2a7e46c626d3a008a_9.csv';
end
if nargin<2
    ts=1/100;
end
if nargin<3
    window_size = 25;
end
if nargin<4
    calc_full_model = 1;
end
if nargin<5
    calc_reduced_model = 1;
end
if nargin<6
    save_models = 1;
end
if nargin<7
    plot_models = 1;
end

if isdeployed
    %need to fix types of command line arguments... yay
    datafile
    ts = str2double(ts)
    window_size = str2double(window_size)
    calc_full_model = str2double(calc_full_model)
    calc_reduced_model = str2double(calc_reduced_model)
    save_models = str2double(save_models)
    plot_models = str2double(plot_models)
end

if plot_models
    calc_full_model = 1;
    calc_reduced_model = 1;
end

disp 'Loading'
tic
traj = read_traj_data(datafile, ts);
S.traj = traj;
toc

%mechanical parameters of the fly
p.Ixx= 0.306*(10^-12);
p.Iyy= 0.5060*(10^-12);
p.Izz= 0.306*(10^-12); 

p.m_f = 0.9315*(10^-6);   % kg
p.g = 9.81;

S.p = p;

disp 'Smoothing'
tic

%filter for smoothing the measurements & numeric differentiation
x_val_f = myfilter(traj.x,window_size);
vx_val_f = diff(x_val_f)/traj.Ts;    vx_val_f(end+1) = vx_val_f(end);
ax_val_f = diff(vx_val_f)/traj.Ts;   ax_val_f(end+1) = ax_val_f(end);

y_val_f = myfilter(traj.y,window_size);
vy_val_f = diff(y_val_f)/traj.Ts;    vy_val_f(end+1) = vy_val_f(end);
ay_val_f = diff(vy_val_f)/traj.Ts;   ay_val_f(end+1) = ay_val_f(end);

z_val_f = myfilter(traj.z,window_size);
vz_val_f = diff(z_val_f)/traj.Ts;    vz_val_f(end+1) = vz_val_f(end);
az_val_f = diff(vz_val_f)/traj.Ts;   az_val_f(end+1) = az_val_f(end);

toc

%yaw angle: fly faces in the direction it is moving
eta_val = atan2(traj.vy,traj.vx);
w_eta_val = (-vy_val_f .* ax_val_f + vx_val_f .* ay_val_f) ./ (vx_val_f .^ 2 + vy_val_f .^ 2);
a_eta_val = -2 .* (-vy_val_f .* ax_val_f + vx_val_f .* ay_val_f) .* (vx_val_f .* ax_val_f + vy_val_f .* ay_val_f) ./ (vx_val_f .^ 2 + vy_val_f .^ 2) .^ 2;

%get tau for full model
if calc_full_model
    disp 'Computing full model'
    tic

    tau = get_tau(traj.ax,traj.ay,traj.az,eta_val,w_eta_val,a_eta_val,p);
    S.tau = tau;

    toc

    if save_models
        save_tau(datafile,'full',tau);
    end

end


%get tau for the roll angle phi = 0 and constant pitch angle theta = 30�
if calc_reduced_model
    disp 'Computing reduced model'
    tic

    size_meas = length(x_val_f);
    theta = 30*pi/180*ones(size_meas,1);
    w_theta_val = zeros(size_meas,1);
    a_theta_val = zeros(size_meas,1);

    tau_red = get_tau_red(traj.ax,traj.ay,traj.az,traj.theta,w_eta_val,w_theta_val,a_eta_val,a_theta_val,p);
    S.tau_red = tau_red;

    toc

    if save_models
        save_tau(datafile,'reduced',tau_red);
    end

end

if plot_models

    %compare reduced and full model
    figure;
    subplot(611);plot(tau_red(:,1));hold on;plot(tau(:,1),'r--');ylabel('Fx');
    title('Reduced (blue) vs Full Model (red)')
    subplot(612);plot(tau_red(:,2));hold on;plot(tau(:,2),'r--');ylabel('Fy');
    subplot(613);plot(tau_red(:,3));hold on;plot(tau(:,3),'r--');ylabel('Fz');
    subplot(614);plot(tau_red(:,4));hold on;plot(tau(:,4),'r--');ylabel('T \phi');
    subplot(615);plot(tau_red(:,5));hold on;plot(tau(:,5),'r--');ylabel('T \theta');
    subplot(616);plot(tau_red(:,6));hold on;plot(tau(:,6),'r--');ylabel('T \eta');

    %compare filtered and not
    figure;
    subplot(411);plot(traj.x);hold on;plot(x_val_f,'r--');ylabel('x');
    title('Original (blue) vs smoothed (red)');
    subplot(412);plot(traj.vx);hold on;plot(vx_val_f,'r--');ylabel('vx');
    subplot(413);plot(traj.ax);hold on;plot(ax_val_f,'r--');ylabel('ax');
end

end

function save_tau(fn, slug, tau)
    %remove the extension and add the slug and .mat
    fn = [fn(1:end-4) '_dynamics_' slug '.mat'];

    %struct for saving
    S.Fx = tau(:,1);
    S.Fy = tau(:,2);
    S.Fz = tau(:,3);
    S.T_phi = tau(:,4);
    S.T_theta = tau(:,5);
    S.T_eta = tau(:,6);

    save(fn,'-struct','S');
    disp(['Wrote ' fn]);

end
