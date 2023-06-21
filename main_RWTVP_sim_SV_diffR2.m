% Estimate a RW-TVP model:
% yt = xt' * bt + N(0,sig2t), 
% b_jt = b_{j,t-1} + N(0,d_jt), j = 1, ..., K
% d_jt = vj2 * I(z_jt > c_j),
% z_jt = (1-rho_j) * u_j + rho_j * z_{j,t-1} + N(0,sj),
%
% use GCK algorithm for zt (adpative MH， multivariate)
% draw static parameters by integrating out bt (adaptive MH, multivariate)
% use ASIS for extra boosting


clear;
dbstop if warning;
dbstop if error;
addpath(genpath('C:\Users\Zhongfang\Documents\My Research\Bayesian_TVP\2021Feb\Functions'));
addpath(genpath('C:\Users\Zhongfang\Documents\My Research\Bayesian_TVP\2021Apr\Functions'));
addpath(genpath('C:\Users\Zhongfang\Documents\My Research\Bayesian_TVP\2021Jul\Functions'));
addpath(genpath('C:\Users\Zhongfang\Documents\My Research\Bayesian_TVP\2021Sep\Functions'));
% rng(123);% existing
% rng(123456); %final
% rng(123456789); %verify
% rng(123211);
% rng(98765432);
rng(3721);
% rng(222);


%% Read data
mdl = {'CV','KHS','LG','LTVP','GHS','NWest'};
n_sheet = 2; 

tic;
for mdlj = 4:4 %model
    disp(mdl{mdlj});
    for sheet_i = 1:1
        disp(['work sheet ', num2str(sheet_i)]);

        read_file = 'Simulated_Data_SV_diffR2.xlsx';
        read_sheet = ['D',num2str(sheet_i)];
        data = readmatrix(read_file, 'Sheet', read_sheet, 'Range', 'A2:N301');
        y = data(:,1);
        sigtrue = data(:,2);
        x = data(:,3:8);
        btrue = data(:,9:14);


        % Set up
        ndraws = 5000;%5000*3;
        burnin = 2000*2.5;

%         ind_SV = 1;
        ind_SV = 0;
        ind_forecast = 0;
        ind_sparse = 0;
        switch mdlj
            case 1 %const variance
                draws = RWTVP(y, x, burnin, ndraws, ind_SV, ind_sparse, ind_forecast);               
            case 2 %dynamic horseshoe
                draws = RWTVP_KHS3_scale(y, x, burnin, ndraws, ind_SV, ind_forecast);
            case 3 %logistic MI
                draws = RWTVP_LS_LG(y, x, burnin, ndraws, ind_SV, ind_sparse, ind_forecast);   
            case 4 %truncation
                draws = RWTVP_LinkConst_AA_same(y, x, burnin, ndraws, ind_SV, ind_sparse, ind_forecast); 
            case 5 %GHS
                draws = RWTVP_DHS_robust2(y, x, burnin, ndraws, ind_SV, ind_sparse, ind_forecast);                 
            case 6 %NWest
                draws = RWTVP_NWest6(y, x, burnin, ndraws, ind_SV);
            otherwise
                error('Wrong model');
        end
        disp([mdl{mdlj},', Sheet ', num2str(sheet_i), ' is completed!']);
        toc;
%             if sheet_i == 1
%                 save(['Est_',mdl{mdlj},'_SV.mat'], 'draws');
%             end


        % RMSE
%         if sheet_i == 1
            rmse = compute_rmse_tvp_beta(draws.beta, btrue);
%                 write_col = {'A','B','C','D','E','F','G'};
%                 K = size(x,2);
%                 for j = 1:K
%                     write_sheet = ['Para',num2str(j)];
%                     writematrix(rmse(:,j),read_file,'Sheet',write_sheet,'Range',[write_col{mdlj}, '2']);
%                 end
%         end

    end %loop of sheet
end %loop of model


%% Plot
% coef_name = {'RW','CP','Const'};
% n = length(y);
% px = (1:n)';
% for j = 1:3  
%     subplot(3,3,3*j-2);
%     plot_shade(px, prctile(draws.beta{j},[5 50 95])');
%     hold on;
%     plot(px, btrue(:,j),'r--');
%     hold off;
%     title([coef_name{j},': CV']);
%     
%     subplot(3,3,3*j-1);
%     plot_shade(px, prctile(draws_MI.beta{j},[5 50 95])');
%     hold on;
%     plot(px, btrue(:,j),'r--');
%     hold off;
%     title([coef_name{j},': MI']); 
%     
%     subplot(3,3,3*j);
%     plot_shade(px, prctile(draws_DP.beta{j},[5 50 95])');
%     hold on;
%     plot(px, btrue(:,j),'r--');
%     hold off;
%     title([coef_name{j},': DP']); 
% end
% 
% 










    





