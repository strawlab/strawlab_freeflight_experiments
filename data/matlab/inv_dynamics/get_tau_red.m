function tau = get_tau_red(ax_val,ay_val,az_val,theta_val,w_eta_val,w_theta_val,a_eta_val,a_theta_val,p)

I_xx = p.Ixx;
I_yy = p.Iyy;
I_zz = p.Izz;

m_f = p.m_f;
g = 9.81;

tau = zeros(length(ax_val),6);

for i=1:length(ax_val)
   ax = ax_val(i); ay = ay_val(i); az = az_val(i); 
   theta = theta_val(i);
   w_eta = w_eta_val(i); w_theta = w_theta_val(i);
   a_eta = a_eta_val(i); a_theta = a_theta_val(i);
   
   tau_1 = m_f * ax;
   tau_2 = m_f * ay;
   tau_3 = m_f * az + g * m_f;
   tau_4 = -cos(theta) * w_eta * w_theta * I_yy + I_zz * cos(theta) * w_eta * w_theta + sin(theta) * I_xx * a_eta;
   tau_5 = -I_zz * cos(theta) * sin(theta) * w_eta ^ 2 + I_xx * cos(theta) * sin(theta) * w_eta ^ 2 + I_yy * a_theta;
   tau_6 = I_zz * cos(theta) ^ 2 * a_eta - I_xx * cos(theta) ^ 2 * a_eta + I_xx * a_eta;
   tau(i,:) = [tau_1,tau_2,tau_3,tau_4,tau_5,tau_6];
end
end
