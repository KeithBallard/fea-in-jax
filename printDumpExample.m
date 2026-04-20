
%microscale tests
numDoF                     = [45582             ,180426            ,800000]; 
numDoF_LARGE               = [45582             ,180426            ,800000,2864034];

load('testOutput.mat')


loglog(numDoF,f0,'-o','LineWidth',2);
hold on;
lgd = legend('','','','FontWeight', 'bold', 'FontSize', 14);
l = lgd.Location;
lgd.Location = 'northwest';
xlabel('# DoF','FontWeight', 'bold', 'FontSize', 16)
ylabel('Time (seconds)','FontWeight', 'bold', 'FontSize', 16)
set(gca, 'FontWeight', 'bold')
ax = gca;
ax.TickLength = [0.02 0.02]; 
ax.LineWidth = 1; 
ax.FontSize = 18;
hold off;
export_fig('exampleOutputPlot','-png','-transparent');
