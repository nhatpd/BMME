%% 
% Let us consider ONMF problem: Given X and r, solve min 1/2 ||X-UV||^2 s.t U>=0, V>=0, VV'=I_r
% BMME solves the penalized ONMF problem.
% Reference: DN Phan, LTK Hien, N Gillis, "Block Alternating Bregman Majorization Minimization 
% with Extrapolation"
%
% Input:   X: the input data set
%          r: rank
%    options is a structure including
%           init.U and init.V: initialization, default = SVD based.
%                     maxiter: max number of iterations, default= 
%                     timemax: max running time
%                       lambda: penalty parameter, default = 1000
%
%
% Output: U, V: solution
%          e: relative fitting error sequence
%     e_orth: orthogonal error sequence = norm(VV'-I)
%          t: time sequence
%        obj: the objective function 
%
% Written by LTK Hien
% Last update: June 2021
%%
function [U,V,e,e_orth,obj,t] = BMME(X,r,options) 
starttime=tic;
%% Parameters of NMF algorithm
if nargin < 3
    options = [];
end
if ~isfield(options,'display')
    options.display = 1; 
end
if ~isfield(options,'init')
    [U,V] = NNDSVD(X,r,0); % SVD-based initialization
else
    U = options.init.U; 
    V = options.init.V; 
end
if ~isfield(options,'maxiter')
    options.maxiter = inf; 
end
if ~isfield(options,'timemax')
    options.timemax = 10; 
end
if ~isfield(options,'lambda')
    options.lambda = 1000; % lambda: penalty parameter in lambda||I-VVt||^2
end

lambda=options.lambda;
% inner_iter=options.inner_iter; 

nX = norm(X,'fro'); 
i=1;
t(i)=toc(starttime);
Ir=eye(r);

time_err0=tic;
VVt=V*V';
UtU=U'*U;

XUV2= max(0,nX^2 - 2*sum(sum( (X*V').*U ) ) + sum( sum((VVt).*(UtU) ) ) );
e(1)=sqrt(XUV2) / nX; 
e_orth(1)=norm(VVt - Ir,'fro');

obj(1)=XUV2/2+lambda/2*e_orth(1)^2;
time_err=toc(time_err0);

if lambda==0 
    error('lambda must be positive');
end
paramU = 1;
paramV = 1;
U_old=U;
V_old=V;
C1 = 0.9999;
C2 = 0.9999;
L1 =  norm(VVt);
L1_prev = L1;
lambda2=2*lambda;
Lv = max(norm(UtU),lambda2);
lambda6=lambda*6;
gamma2=1; % stepsize for updating V
%% Main loop
while i <= options.maxiter && t(i) < options.timemax 
    
    % update U, see Section 2.2.1 for a detailed explanation
    XVt=X*V';
    VVt=V*V';
    L1_prev = L1;
    L1 = norm(VVt);
    gamma1=1/L1; %step size for U 
    
    
    % Update U
   
    param_prev = paramU;
    paramU = 0.5 * ( 1+sqrt( 1 + 4*param_prev^2 ) ); 
    ex_coeff = (param_prev-1)/paramU; 
    ex_coeff = min(ex_coeff,sqrt(C1*L1_prev/L1));
    U_ex=U+ex_coeff*(U-U_old);
    U_old=U;
    U=max(U_ex-gamma1*(U_ex*VVt-XVt),eps);

    % update V,
    UtU=U'*U; 
    normUtU=norm(UtU); % ||UU'||
    Lv = max(normUtU,lambda2);
    Lv_prev=max(norm(U_old'*U_old),lambda2); %to do the linesearch
    UtX=U'*X;
    Lv3=Lv^3;
    % 
    param_prev = paramV;
    paramV = 0.5 * ( 1+sqrt( 1 + 4*param_prev^2 ) ); 
    ex_coeff = (param_prev-1)/paramV;
        
    [~,V_ex,~,gradV_ex] = find_beta(V,V_old,C2,Lv,Lv_prev,ex_coeff,lambda6);

    V_old=V;
    VVt=V_ex*V_ex'; 
    g=gradV_ex - gamma2*((UtU)*V_ex-UtX+2*lambda*(VVt*V_ex-V_ex));
    g=max(eps,g); 
    normg2=norm(g,'fro')^2; %||\Pi_{+}(g) ||^2
    c=lambda6*normg2;
    Delta=c^2+4/27*c*Lv3; % find Delta
    sDelta=sqrt(Delta);
          
    % find rho
    rho=1/3*Lv+nthroot((c+sDelta)/2+ Lv3/27,3)+nthroot((c-sDelta)/2+ Lv3/27,3);
    V=g/rho;
       
    % compute relative fitting error and orthogonal error; the computed time is not counted. 
    i=i+1;
    
    time_err0=tic;
    VVt=V*V';
    XUV2=max(0,nX^2 - 2*sum(sum( (X*V').*U) ) + sum(sum( (UtU).*(VVt) ) ) );
    e(i)=sqrt(XUV2 ) / nX; 
    e_orth(i)=norm(VVt - Ir,'fro');
    obj(i)=XUV2/2+lambda/2*e_orth(i)^2;
    time_err=time_err+toc(time_err0);
    t(i)=toc(starttime)-time_err; 
    if options.display == 1
     if mod(i,100)==0
        fprintf('BMME, iteration %4d fitting error: %1.2e  orthogonal error: %1.2e\n',i,e(i),e_orth(i));     
     end
    end
end
end

function[beta,V_ex,normV_ex,gradV_ex] = find_beta(V,V_old,C2,Lv,Lv_prev,ex_coeff,lambda6)

kappa = C2*0.5; % # delta-epsilon chosen close to 1
deltaV = V-V_old;
%beta = min(ex_coeff,beta);
beta=ex_coeff;
V_ex = V+ beta*deltaV;
normV_ex=norm(V_ex,'fro')^2; 
normV = norm(V,'fro')^2; 
normV_old = norm(V_old,'fro')^2;

gradV = V*(lambda6*normV+Lv_prev);
gradV_ex = V_ex*(lambda6*normV_ex+Lv);

temp=h2(Lv,lambda6,normV);
temp2=h2(Lv_prev,lambda6,normV);
D_kernelV = h2(Lv_prev,lambda6,normV_old) -temp2 + sum(sum(deltaV.*gradV));

D_kernelV_exp = temp -  h2(Lv,lambda6,normV_ex) + beta*sum(sum(deltaV.*gradV_ex));

while kappa*D_kernelV - D_kernelV_exp <-1e-10 && beta>0
    beta = beta*0.9;
    V_ex = V+ beta*deltaV;
    normV_ex=norm(V_ex,'fro')^2;
    gradV_ex = V_ex*(lambda6*normV_ex+Lv);
    D_kernelV_exp = temp -  h2(Lv,lambda6,normV_ex)  + beta*sum(sum(deltaV.*gradV_ex));
    if beta <= 1e-4
        beta = 0;
    end
end
end
function h2V=h2(Lv,lambda6,normV)
h2V=lambda6/4*normV^2 + Lv/2*normV;
end

