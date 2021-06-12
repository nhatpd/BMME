clear; clc;
clear; clc;
addpath('tools');
dataset = {'movielens1m'};
load([dataset{1},'.mat']);

para.data = 'movielens1m';

rng('default');
rng(30); 

[row, col, val] = find(data);

[m, n] = size(data);


clear user item;
% clear data;

val = val - mean(val);
val = val/std(val);
theta = 5;

para.maxR = 5;
para.maxtime = 20;

para.regType = 4;
para.maxIter = 20000;
lambda = 0.5;

para.reg = 'exponential regularization';


para.tol = 1e-9;
trials = 20;
for ii = 1:trials
    idx = randperm(length(val));

    train_Idx = idx(1:floor(length(val)*0.7));
    test_Idx = idx(ceil(length(val)*0.3): end);

    clear idx;

    train_Data = sparse(row(train_Idx), col(train_Idx), val(train_Idx));
    train_Data(size(data,1), size(data,2)) = 0;

    para.test.row  = row(test_Idx);
    para.test.col  = col(test_Idx);
    para.test.data = val(test_Idx);
    para.test.m = m;
    para.test.n = n;

    % clear m n;




    [~, n_t] = size(train_Data);


    R = randn(n_t, para.maxR);
    para.R = R;
    clear n_t;
    U0 = powerMethod( train_Data, R, para.maxR, 1e-6);
    [~, ~, V0] = svd(U0'*train_Data, 'econ');
    para.U0 = U0;
    para.V0 = V0;




    fprintf('runing Cocain \n');
        method = 1;
        [out{method}{ii}] = CoCaIn( train_Data, lambda, theta, para ); 
    
    fprintf('runing BMME backtracking \n');
        method = 2;
        [out{method}{ii}] = BMME_Backtracking( train_Data, lambda, theta, para );
    
    fprintf('runing BMME \n');
        method = 3;
        [out{method}{ii}] = BMME( train_Data, lambda, theta, para );


end

timeStamp = strcat(datestr(clock,'yyyy-mm-dd_HH-MM-ss'));
% save(['results/',dataset{1}, strcat(timeStamp,'.mat')], 'out');

for i = 1:3
    min_iter = 1e10;
    for j = 1:trials
        [iter ~] = size(out{i}{j}.obj);
    min_iter = min(min_iter,iter);
    
    end
    
    out_obj{i} = out{i}{1}.obj(1:min_iter);
    
    
    out_Time{i} = out{i}{1}.Time(1:min_iter);
    
    
    out_RMSE{i} = out{i}{1}.RMSE(1:min_iter);
    
    
    for j = 2:trials
        out_obj{i} = [out_obj{i},out{i}{j}.obj(1:min_iter)];
        
        
        out_Time{i} = [out_Time{i},out{i}{j}.Time(1:min_iter)];
        
    
        out_RMSE{i} = [out_RMSE{i},out{i}{j}.RMSE(1:min_iter)];
    end
    
    
end

max_time = 1e6;
max_ind = 0;
for i = 1:3
    if max_time > max(mean(out_Time{i},2))
        max_ind = i;
        max_time = max(mean(out_Time{i},2));
        
    end
end
figure;
% subplot(1, 2, 1);
hold on;

plot(mean(out_Time{1},2), mean(out_RMSE{1},2), 'b-.','LineWidth',2);
hold on;

plot(mean(out_Time{2},2), mean(out_RMSE{2},2), 'r','LineWidth',2);
hold on;

plot(mean(out_Time{3},2), mean(out_RMSE{3},2), 'g','LineWidth',2);
hold on;

legend('CoCaIn','BMME-backtracking', 'BMME');

xlabel('Time (s)');
ylabel('RMSE');
xlim([0 max_time])
title(dataset{1})
saveas(gcf,[dataset{1},'rmse.fig'])
%---------------------------------




figure;
% subplot(1, 2, 1);
hold on;

plot(mean(out_Time{1},2), mean(out_obj{1},2), 'b-.','LineWidth',2);
hold on;

plot(mean(out_Time{2},2), mean(out_obj{2},2), 'r','LineWidth',2);
hold on;

plot(mean(out_Time{3},2), mean(out_obj{3},2), 'g','LineWidth',2);
hold on;


legend('CoCaIn','BMME-backtracking','BMME');

xlabel('Time (s)');
ylabel('Objective value');
xlim([0 max_time])
title(dataset{1})
saveas(gcf,[dataset{1},'obj.fig'])
