%% Quick code for Ronald's and mine work, Plotting

Det = 900; % in MHz
Rab_ref = 600/2/pi;
Gen_Rab_ref = sqrt(Det^2+Rab_ref^2);

Rab_arr = Rab_ref*2.^[-2:1];
Gen_Rab_arr = sqrt(Det^2+Rab_arr.^2);

hold all
plot(Rab_arr, Gen_Rab_arr, 'b+', 'MarkerSize', 8)
x = linspace(0,200,1000);
plot(x, sqrt(x.^2+Det^2), 'r-')
plot(x, 900*ones(size(x)), 'k--')
%xlim([])
ylim([890, 930])
legend('samples','expected dependence', 'detining')
title('Measurable vs Rabi Frequency')
xlabel('Rabi Freqency [MHz]/ Laser Power [a.u.]')
ylabel('Measureable Frequency [MHz]')
box on