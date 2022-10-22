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
rng(222);


%% Read data
mdl = {'LTVP'};
n_sheet = 1;

tic;
for mdlj = 1:1 %model
    disp(mdl{mdlj});
    for sheet_i = 1:n_sheet
        disp(['work sheet ', num2str(sheet_i)]);

        read_file = 'Simulated_Data_SV.xlsx';
        read_sheet = 'Data';
        data = readmatrix(read_file, 'Sheet', read_sheet, 'Range', 'A2:N301');
        y = data(:,1);
        sigtrue = data(:,2);
        x = data(:,3:8);
        btrue = data(:,9:14);


        % Set up
        ndraws = 5000*3;
        burnin = 2000*2.5;

        ind_SV = 1;
        ind_forecast = 0;
        ind_sparse = 0;
        switch mdlj
            case 1 %LTVP
                draws = RWTVP_LinkConst_AA_same(y, x, burnin, ndraws, ind_SV, ind_sparse, ind_forecast); 
            otherwise
                error('Wrong model');
        end
        disp([mdl{mdlj},', Sheet ', num2str(sheet_i), ' is completed!']);
        toc;
        if sheet_i == 1
            save(['Est_',mdl{mdlj},'_SV.mat'], 'draws');
        end
    end %loop of sheet
end %loop of model


