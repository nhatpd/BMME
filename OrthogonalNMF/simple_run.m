clear all; close all; clc;
options.maxiter =inf;
options.timemax =10;
options.display=0;
seed = rng;
m = 500;
n = 500;
r = 10;
options.lambda = 1000;

 U = rand(m,r);
 V = zeros(r,n); 
 for i = 1 : n
    V(randi(r),i) = rand(1); 
 end
 for i = 1 : r
     V(i,:) = V(i,:)/norm(V(i,:));
 end
% 
%V=rand(r,n);
X = U*V; % no noise
R = rand(m,n);
nX=norm(X,'fro');
X = X + 0.05*R/norm(R,'fro')*nX; % add noise

% % 
% initial  
optionsSPA.display = 0; 
K = SPA(X,r,optionsSPA); 
U0 = X(:,K); 
V0 = orthNNLS(X,U0); 
options.init.U =U0;
options.init.V = V0;
fprintf('\n running BMM\n ');
% run BMM
[~,~,BMM_e,BMM_eorth,BMM_obj,BMM_t] = BMM(X,r,options);

%run BMME
fprintf('\n running BMME\n ');
[~,~,BMME_e,BMME_eorth,BMME_obj,BMME_t] = BMME(X,r,options);

% run aBPALM
fprintf('\n running BPALM\n ');
[~,~,BPALM_e,BPALM_eorth,BPALM_obj,BPALM_t] = aBPALM(X,r,options);


% run BIBPA
fprintf('\n running BIBPA\n ');
[~,~,BIBPA_e,BIBPA_eorth,BIBPA_obj,BIBPA_t] = BIBPA(X,r,options);


% emin=min([e1,e2,e3,e4]); 
% e_orthmin=min([e_orth1,e_orth2,e_orth3,e_orth4]); 
% objmin=min([obj_1,obj_2,obj_3,obj_4]);

objmin=min([BMM_obj,BMME_obj,BPALM_obj,BIBPA_obj]);



figure;
set(0, 'DefaultAxesFontSize', 18);
set(0, 'DefaultLineLineWidth', 2);

semilogy(BMM_t,BMM_obj-objmin,'b--','LineWidth',1.5);hold on; % adaptive
semilogy(BMME_t,BMME_obj-objmin,'m-..','LineWidth',3);hold on; % adaptive
semilogy(BPALM_t,BPALM_obj-objmin ,'r-.','LineWidth',2);hold on; 
semilogy(BIBPA_t,BIBPA_obj-objmin ,'k--','LineWidth',3);hold on; 

ylabel('Obj - Obj_{min} ');
xlabel('Iteration'); 
legend('BMM','BMME','A-BPALM', 'BIBPA');


