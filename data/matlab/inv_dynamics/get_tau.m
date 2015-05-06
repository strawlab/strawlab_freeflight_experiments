function tau = get_tau(ax_val,ay_val,az_val,eta,w_eta,a_eta,p)

F_val = zeros(size(ax_val)); phi_val = zeros(size(ax_val)); theta_val = zeros(size(ax_val));
opt = optimoptions('fsolve','Display','off','TolFun',1e-14);
delta = 60*pi/180;
p.delta = delta;
F_p_t = [0,0,0];
%calculate force vector F, roll and pitch angle phi and theta
for i = 1:length(ax_val)        %Could be parallelised
    F_p_t = fsolve(@calc_angles,F_p_t,opt,eta(i),ax_val(i),ay_val(i),az_val(i),p);
    F_val(i) = F_p_t(1); phi_val(i) = F_p_t(2); theta_val(i) = F_p_t(3);
end
%numeric differentation by finite differences
w_phi_val = diff(phi_val);  w_phi_val(end+1) = w_phi_val(end);
w_theta_val = diff(theta_val);  w_theta_val(end+1) = w_theta_val(end);

tau = zeros(length(ax_val),6);
m_f = p.m_f;
g = p.g;
%calculate the inverse dynamics
for i = 1:length(ax_val)        %Could be parallelised
    F = F_val(i); phi = phi_val(i); theta = theta_val(i);
    ax = ax_val(i); ay = ay_val(i); az = az_val(i);
    w_phi = w_phi_val(i);       a_phi = 0;
    w_theta = w_theta_val(i);   a_theta = 0;
    tau123 = [m_f * ax; m_f * ay; m_f * az + g * m_f;];
    tau456 = calc_tau(phi,theta,eta(i),w_phi,w_theta,w_eta(i),a_phi,a_theta,a_eta(i),p);
    tau(i,1:3) = tau123';
    tau(i,4:6) = tau456';
end

end

function res = calc_angles(F_p_t,eta,ax,ay,az,p)
m_f = p.m_f;
g = p.g;

F = F_p_t(1);
phi = F_p_t(2);
theta = F_p_t(3);
delta = p.delta;

res = [m_f * ax - F * (cos(phi) * cos(eta) * sin(theta) * cos(delta) + cos(eta) * cos(theta) * sin(delta) - sin(phi) * sin(eta) * cos(delta)); m_f * ay - F * (cos(phi) * sin(eta) * sin(theta) * cos(delta) + sin(eta) * cos(theta) * sin(delta) + cos(eta) * sin(phi) * cos(delta)); m_f * az + g * m_f - F * (cos(theta) * cos(phi) * cos(delta) - sin(theta) * sin(delta));];
end

function tau456 = calc_tau(phi,theta,eta,w_phi,w_theta,w_eta,a_phi,a_theta,a_eta,p)
I_xx = p.Ixx;
I_yy = p.Iyy;
I_zz = p.Izz;

tau_4 = I_yy * cos(theta) ^ 2 * cos(phi) * sin(phi) * w_eta ^ 2 - I_zz * cos(theta) ^ 2 * cos(phi) * sin(phi) * w_eta ^ 2 - 0.2e1 * w_theta * I_yy * cos(theta) * cos(phi) ^ 2 * w_eta + 0.2e1 * w_theta * I_zz * cos(theta) * cos(phi) ^ 2 * w_eta - I_yy * cos(phi) * sin(phi) * w_theta ^ 2 + I_zz * cos(phi) * sin(phi) * w_theta ^ 2 + cos(theta) * w_eta * w_theta * I_yy - I_zz * cos(theta) * w_eta * w_theta + sin(theta) * I_xx * a_eta + I_xx * a_phi;
tau_5 = I_yy * cos(theta) * cos(phi) ^ 2 * sin(theta) * w_eta ^ 2 - I_zz * cos(theta) * cos(phi) ^ 2 * sin(theta) * w_eta ^ 2 + w_eta * sin(theta) * cos(phi) * sin(phi) * w_theta * I_yy - w_eta * sin(theta) * cos(phi) * sin(phi) * w_theta * I_zz + I_xx * cos(theta) * sin(theta) * w_eta ^ 2 - cos(theta) * cos(phi) * sin(phi) * a_eta * I_yy - I_yy * cos(theta) * sin(theta) * w_eta ^ 2 + cos(theta) * cos(phi) * sin(phi) * a_eta * I_zz + w_phi * I_xx * cos(theta) * w_eta + I_yy * cos(phi) ^ 2 * a_theta - I_zz * cos(phi) ^ 2 * a_theta + I_zz * a_theta;
tau_6 = -I_yy * cos(theta) ^ 2 * cos(phi) ^ 2 * a_eta + I_zz * cos(theta) ^ 2 * cos(phi) ^ 2 * a_eta - cos(theta) * cos(phi) * sin(phi) * a_theta * I_yy + cos(theta) * cos(phi) * sin(phi) * a_theta * I_zz - I_xx * cos(theta) ^ 2 * a_eta + I_yy * cos(theta) ^ 2 * a_eta + sin(theta) * I_xx * a_phi + I_xx * a_eta;
tau456 = [tau_4,tau_5,tau_6];
end