%% 
% Let us consider ONMF problem: Given X and r, solve min 1/2 ||X-UV||^2 s.t U>=0, V>=0, VV'=I_r
% BMM (that is, BMME without inertial) solves the penalized ONMF problem
% Reference: DN Phan, LTK Hien, N Gillis, "Block Alternating Bregman Majorization Minimization 
% with Extrapolation"
%
%  Input:   X: the input data set
%          r: rank
%
% options is a structure including
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
% Written by LTK Hien,
% Last update: March 2021
%%
function [U,V,e,e_orth,obj,t] = BMM(X,r,options) 
starttime=tic;
% Parameters of NMF algorithm
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
lambda=options.lambda;
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

if lambda==0 
    error('lambda must be positive');
end

%% Main loop
while i <= options.maxiter && t(i) < options.timemax 
    % update U, 
    XVt=X*V';
    VVt=V*V';
    gamma1=1/(norm(VVt)); %step size for U 
    % update U
 
     U=max(U-gamma1*(U*VVt-XVt),eps);
  
    % update V, 
    UtU=U'*U; 
    normUtU=norm(UtU); % ||UU'||
    normUtU3=normUtU^3;
    lambda6=lambda*6;
    UtX=U'*X;
    gamma2=1;
    

    normV2=norm(V,'fro')^2;
    VVt=V*V';


    g=(lambda6*normV2+normUtU)*V-gamma2*((UtU)*V-UtX+2*lambda*(VVt*V-V));
    g=max(eps,g); 
    normg2=norm(g,'fro')^2; %||\Pi_{+}(g) ||^2

    c=lambda6*normg2;
    Delta=c^2+4/27*c*normUtU3; % find Delta
    sDelta=sqrt(Delta);

    % find rho
    rho=1/3*normUtU+nthroot((c+sDelta)/2+ normUtU3/27,3)+nthroot((c-sDelta)/2+ normUtU3/27,3);
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
        fprintf('BMM, iteration %4d fitting error: %1.2e  orthogonal error: %1.2e\n',i,e(i),e_orth(i));     
    end
    end
end
end
