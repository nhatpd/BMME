%%
% ONMF problem: Given X and r, solve min 1/2 ||X-UV||^2 s.t U>=0, V>=0, VV'=I_r
% vBPALM solves ONMF problem by using the following generating kernel
% h(U,V)=alpha/2 ||U||^2 ||V||^2 + beta/4 ||V||^4 + epsilon_1/2 ||U||^2 +
% epsilon_2/2 ||V||^2
% see AHOOKHOSH et al, "MULTI-BLOCK BREGMAN PROXIMAL ALTERNATING LINEARIZED
% MINIMIZATION AND ITS APPLICATION TO SPARSE ORTHOGONAL NONNEGATIVE MATRIX FACTORIZATION",
%     arxiv, 2019.
%
% This code implements BIBPA proposed in 
% M. Ahookhosh et al, "Multi-block Bregman proximal alternating linearized minimization 
% and its application to orthogonal nonnegative matrix factorization"
% to solve the penalized NMF problem
%
% Input: X, r, options
% options is a structure including
%           init.U and init.V: initialization, default = SVD based.
%                     maxiter: max number of iterations, default= 
%                     timemax: max running time
%                       lambda: penalty parameter
% Output: U, V: solution
%          e: relative fitting error sequence
%     e_orth: orthogonal error sequence = norm(VV'-I)
%          t: time sequence
%
%
% Written by LTK Hien
% Last update: March 2021
%%
function [U,V,e,e_orth,obj,t] = BIBPA(X,r,options) 

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
    options.lambda = 1; % lambda: penalty parameter in lambda||I-VVt||^2
end
if ~isfield(options,'alpha')
    % default value  (alpha,beta2) for h_2
    options.alpha = 1;  
end
if ~isfield(options,'beta')
   % default value (alpha,beta2) for h_2    
   options.beta=options.alpha;
end
lambda=options.lambda;


epsilon=1e-15;
alpha=options.alpha;
beta=options.beta;

L1=1/alpha;
L2=max(6*lambda/beta,1/alpha);

gamma1=1/L1;
gamma2=1/L2;

nX = norm(X,'fro'); 
i=1;
t(i)=toc(starttime);
Ir=eye(r);

time_err0=tic;
XUV2= max(0,nX^2 - 2*sum(sum( (X*V').*U ) ) + sum( sum((V*V').*(U'*U) ) ) );
e(1)=sqrt(XUV2) / nX; 
e_orth(1)=norm(V*V' - Ir,'fro');
obj(1)=XUV2/2+lambda/2*e_orth(1)^2;
time_err=toc(time_err0);
ex_coeffU=epsilon/2*eps/gamma1;
ex_coeffV=epsilon/2*eps/gamma2;
if lambda==0 
    error('lambda must be positive');
end
U_old=U;
V_old=V;
%% Main loop
while i <= options.maxiter && t(i) < options.timemax 
    % update U, 
    XVt=X*V';
    VVt=V*V';
    normV2=norm(V,'fro')^2;
    eta1=alpha*normV2 + epsilon; 
    
    U_ex=U-U_old; 
    U_old=U;
    U=max(U-gamma1/eta1*(U*VVt-XVt-ex_coeffU*U_ex),0);
 
    % update V, 
    UtU=U'*U; 
    UtX=U'*X; 

    V_ex=V-V_old; 
    V_old=V;
    eta2=epsilon + alpha*norm(U,'fro')^2;
    g=eta2*V + beta*normV2*V-gamma2*((UtU)*V-UtX+2*lambda*(VVt*V-V)-ex_coeffV*V_ex);
    % V+=max(g,0)/rho
    g=max(0,g); 
    normg2=norm(g,'fro')^2; %||\Pi_{+}(g) ||^2
    c=beta*normg2; 

    xip3=eta2^3;
    Delta=c^2+(4/27)*xip3*c; % find Delta
    sDelta=sqrt(Delta);
    
    % find tk
    tk=eta2/3+nthroot((c+sDelta)/2+xip3/27,3)+nthroot((c-sDelta)/2+xip3/27,3);
    V=g/tk;
   
    
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
        fprintf('BIBPA iteration %4d fitting error: %1.2e  orthogonal error: %1.2e\n',i,e(i),e_orth(i));     
     end
    end
end
end
