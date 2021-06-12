%% 
% Let us consider ONMF problem: Given X and r, solve min 1/2 ||X-UV||^2 s.t U>=0, V>=0, VV'=I_r
% aBPALM solves ONMF problem with adaptive stepsize and 
% h(U,V)=(beta1/2 ||U||_F^2+1)(alpha/4 ||V||_F^4+beta2/2 ||V||_F^2+1), 
% see AHOOKHOSH et al, "MULTI-BLOCK BREGMAN PROXIMAL ALTERNATING LINEARIZED
% MINIMIZATION AND ITS APPLICATION TO SPARSE ORTHOGONAL NONNEGATIVE MATRIX FACTORIZATION",
%     arxiv, 2019. 
%
% This aBPALM version use the latest value of relative smooth Lipschitz constant
% to be the initial Lipschitz constant for the back tracking step of the next update
%
% Input:   X: the input data set
%          r: rank
% coeff_Lbar: coefficient of Lbar to do back tracking: L1bar=coeff_Lbar*L1;
%                    and L2bar=coeff_Lbar*L2; 
% options is a structure including
%           init.U and init.V: initialization, default = SVD based.
%                     maxiter: max number of iterations, default=
%                     timemax: max running time
% alpha, lambda, beta1, beta2: parameters of the kernel function
%                     epsilon: expected accuracy of the output.
%                       lambda: penalty parameter
%
% Output: U, V: solution
%          e: relative fitting error sequence
%     e_orth: orthogonal error sequence = norm(VV'-I)
%          t: time sequence
%        obj: the objective function
%    accuracy: norm of subgradient
%
% Written by LTK Hien,
% Last update: March 2021
%%
function [U,V,e,e_orth,obj,t] = aBPALM(X,r,options,coeff_Lbar) 

starttime=tic;

%% Parameters of NMF algorithm
if nargin < 3
    options = [];
end
if nargin < 4
    coeff_Lbar=1e-6;
end
if ~isfield(options,'display')
    options.display = 1; 
end
if ~isfield(options,'init')
    [U0,V0] = NNDSVD(X',r,0); 
    U=V0';
    V=U0';% SVD-based initialization
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
if ~isfield(options,'epsilon')
    options.epsilon = 1e-6; % epsilon: the expected accuracy
end
if ~isfield(options,'lambda')
    options.lambda = 1e4; % lambda: penalty parameter in lambda||I-VVt||^2
end
if ~isfield(options,'alpha')
    % default value  (alpha,beta2) for h_2
    options.alpha = 1;  
end
if ~isfield(options,'beta')
   % default value (alpha,beta2) for h_2    
   options.beta=options.alpha;
end

% assign parameter for the multi-block kernel function
% h(U,V)=(beta1/2 ||U||_F^2+1)(alpha/4 ||V||_F^4+beta2/2 ||V||_F^2+1)
alpha=options.alpha;
beta=options.beta;
lambda=options.lambda;

if lambda==0 
    error('lambda must be positive');
end
epsilon=1e-15;
L1=1/alpha;
L2=max(6*lambda/beta,1/alpha);


nX = norm(X,'fro'); 
nX2=nX^2;
i=1;
t(i)=toc(starttime);

Ir=eye(r);

% find ||X-UV||^2, required to evaluate objective function in back tracking step
XUV2=max(nX2 - 2*sum(sum( (X*V').*U ) ) + sum( sum((V*V').*(U'*U) ) ),0); 
objU=XUV2/2;  

time_err0=tic;
e_orth(1)=norm(V*V' - Ir,'fro'); %first orthogonal error
e(1)=sqrt(XUV2) / nX;  % first fitting error
obj(1)=objU+lambda/2*e_orth(1)^2; % first objective function
% first accuracy is set to be inf
time_err=toc(time_err0);

% initial relative smooth Lipschitz constants for backtracking; 
L1bar=coeff_Lbar*L1; %\bar L_1
gamma1=1/L1bar-eps; 

L2bar=coeff_Lbar*L2;  %\bar L_2
gamma2=1/L2bar-eps;
nu1=2; % \nu_1>1 
%% Main loop
while i <= options.maxiter && t(i) < options.timemax 
    % update U 
    % compute all common expression to save time
    normV2=norm(V,'fro')^2; % find ||V||_F^2
    VVt=V*V';   
    XVt=X*V';
    eta1=alpha*normV2 + epsilon; 

    
    % start back tracking
    p=0;
    U_old=U; % fix current U
    % Find gradient with respect to Uold. This value is unchanged during
    % backtracking for U
    grad= U_old*(VVt)-XVt;
    % find 1/2*||Uold||^2_F, this is to calculate D_h
    hUold=norm(U_old,'fro')^2/2;
    % The objective with respect to the current U. It is used during the backtracking
    % Note that lambda/2 ||I-VV'||^2 is unchanged
    objU_old=objU; 
    
    % first U update to check descent inequality
    % L1bar=L1bar*nu^p=L1bar for the first update
    [U,objU,descent] = UpdateU(U_old,grad,hUold,objU_old,gamma1,eta1,VVt,XVt,nX2,L1bar);
    L1bar_0=L1bar; % the initial Lipschitz constant for current bactracking step
    gamma1_0=gamma1;
    while descent>0 && gamma1 > 1e-8 % avoid too small stepsize that causes numerical unstability
       p=p+1;
       nup=nu1^p; 
       L1bar=L1bar_0*nup; % updata L1bar
       gamma1=gamma1_0/nup; % update gamma1 
       % p-th update to check descent inequality
       [U,objU,descent] = UpdateU(U_old,grad, hUold,objU_old,gamma1,eta1,VVt,XVt,nX2,L1bar);  
    end  
    % safety procedure
     for k = 1 : r
      if U(:,k) == 0
        U(:,k) = 1e-16*max(U(:)); 
      end
     end
    
    % update V 
    % compute all common expression to save time
    normU2=norm(U,'fro')^2; % ||U||_F^2
    eta2=epsilon + alpha*normU2;
    UtU=U'*U;
    UtX=U'*X;
    V_old=V; % fix the current V
    VVt=V_old*V_old';
    normV2=norm(V_old,'fro')^2;
    
    
    % Find gradient with respect to V_old. This value is unchanged during
    % backtracking for V
    grad=(UtU)*V_old-UtX+2*lambda*(VVt*V_old-V_old); 
    % find h_2(Vold) and gradient of h_2 with respect to V_old. 
    % These values are used to calculate D_h to check the descent inequality. 
    % Note that the constant 1 can be omitted 
    hVold=beta/4* ( normV2^2) + eta2* normV2/2;
    gradh=(beta* normV2+eta2)*V_old; %
    
    % start back tracking
    p=0;
    % find 1/2 ||X-UV||^2+ lambda/2 ||I-VVT||_F^2
    IrVVt=norm(VVt - Ir,'fro')^2*lambda/2;
    objV_old=objU + IrVVt; % the current objective value, U here is the newest U
        
    % first V update to check descent inequality
    % L2bar=L2bar*nu^p=L2bar for the first update
    [V,descent]=updateV(Ir,grad,gradh,hVold,objV_old,V_old,X,U,UtU,normV2,beta,eta2,gamma2,lambda,nX2,L2bar);
    
    L2bar_0=L2bar; % the initial Lipschitz constant for current bactracking step
    gamma2_0=gamma2;
    while descent>0 && gamma2 > 1e-8 % avoid too small stepsize that causes numerical unstability
        p=p+1;
        nup=nu1^p; 
        L2bar=L2bar_0*nup; % update L2bar 
        gamma2=gamma2_0/nup; % update gamma2
        [V,descent]=updateV(Ir,grad,gradh,hVold,objV_old,V_old,X,U,UtU,normV2,beta,eta2,gamma2,lambda,nX2,L2bar);
    end
          
%  %   Break if converged
%     if gamma1 < 1e-8 && gamma2 < 1e-8
%       break; % stop if no more improvement
%     end
    
     % safety procedure 
    for k = 1 : r
      if V(k,:) == 0
        V(k,:) = 1e-16*max(V(:)); 
      end
    end
    % compute relative fitting error and orthogonal error; the computed time is not counted. 
    i=i+1;
    
    % ||X-UV||^2, required to evaluate objective function in back tracking step for U
    VV_t=V*V';
    U_tU=U'*U;
    XUV2= max(nX^2 - 2*sum(sum( (X*V').*U) ) + sum(sum( (U_tU).*(VV_t) ) ),0); %||X-UV||^2
    objU=XUV2/2;  
    
    time_err0=tic;
    e(i)=sqrt( XUV2 ) / nX; 
    e_orth(i)=norm(V*V' - Ir,'fro');
    
    obj(i)=objU+lambda/2*e_orth(i)^2; % objective function 
       
    time_err=time_err+toc(time_err0);
    t(i)=toc(starttime)-time_err; 
    if options.display == 1  
     if mod(i,100)==0
        fprintf('A-BPALM1, coeff_Lbar = %1.1e: iteration %4d fitting error: %1.2e  orthogonal error: %1.2e\n',coeff_Lbar,i,e(i),e_orth(i));     
     end
    end
end
end
function [U,obj,descent] = UpdateU(Uold,grad,hUold,obj_old,gamma1,eta1,VVt,XVt,nX2,L1bar)
% This function is to update U file fixing V and find the descent error to verify the descent
% inequality
 U=max(Uold-gamma1/eta1*(Uold*VVt-XVt),0);
 %U=max(Uold-(gamma1/(beta1xeta1))*(grad),0); % 
 UUold=(U-Uold);
 iprod=grad(:).'*UUold(:); %find trace
 obj=max((nX2 - 2*sum(sum( (XVt).*U ) ) + sum( sum((VVt).*(U'*U) ) ))/2,0); % 1/2*||X-UV||^2
 % Find D_h(U,Uold)
 gradh=Uold;
 hU=norm(U,'fro')^2/2; % multiply with the coefficient eta1 later
 DhUUold=max(eta1*(hU-hUold-gradh(:).'*UUold(:)),0); % using multi-block kernel
 
 descent=obj-obj_old-iprod-L1bar*DhUUold;
 
end 
function  [V,descent]=updateV(Ir,grad,gradh,hVold,objV_old,V_old,X,U,UtU,normV2,beta,eta2,gamma2,lambda,nX2,L2bar)
    
    g=eta2*V_old + beta*normV2*V_old-gamma2*grad;
        % V+=max(g,0)/rho
    g=max(0,g); 
    normg2=norm(g,'fro')^2; %||\Pi_{+}(g) ||^2
    c=beta*normg2; 

    xip3=eta2^3;
    Delta=c^2+(4/27)*xip3*c; % find Delta
    sDelta=sqrt(Delta);
    %xi^3
    % find rho
    tk=eta2/3+nthroot((c+sDelta)/2+xip3/27,3)+nthroot((c-sDelta)/2+xip3/27,3);
    V=g/tk;
    %V=max(g/rho,0); 
    VVt=V*V'; 
    IrVVt=norm(VVt - Ir,'fro')^2*lambda/2;
    objU=max(0,(nX2 - 2*sum(sum( (X*V').*U ) ) + sum( sum((VVt).*(UtU) ) ))/2); %
    objV=objU + IrVVt; % new objective function 
    
    VVold=V-V_old;
    iprod=grad(:).'*VVold(:); %find trace
    
    normV2new=norm(V,'fro')^2; 
    hV=beta/4* (normV2new^2) + eta2*normV2new/2; 
    DhVVold=max(0,(hV-hVold-gradh(:).'*VVold(:)));
    descent=objV-objV_old-iprod-L2bar*DhVVold;
end

