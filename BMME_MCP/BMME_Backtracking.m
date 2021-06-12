function [output ] = BMME_Backtracking( D, lam, theta, para )
output.method = 'BMME-backtracking';

objstep = 1;

maxIter = para.maxIter;
tol = para.tol*objstep;

[row, col, data] = find(D);

[m, n] = size(D);


R = para.R;
U0 = para.U0;
U1 = U0;

[~, ~, V0] = svd(U0'*D, 'econ');
V0 = V0';
V1 = V0;

spa = sparse(row, col, data, m, n); % data input == D

c_1 = 3;
c_2 = norm(data);

clear D;

obj = zeros(maxIter+1, 1);
RMSE = zeros(maxIter+1, 1);
trainRMSE = zeros(maxIter+1, 1);
Time = zeros(maxIter+1, 1);
Lls = zeros(maxIter+1, 1);
Ils = zeros(maxIter+1, 1);
nnzUV = zeros(maxIter+1, 2);
no_acceleration = zeros(maxIter+1, 1);

part0 = partXY(U0', V0, row, col, length(data));

part0 = data - part0';

ga = theta;

L = 0.01;
lL = 0.001;
uL = 0.01;

maxinneriter = 300;

Lls(1) = L;
if(isfield(para, 'test'))
    tempS = eye(size(U1,2), size(V1',2));
    if(para.test.m ~= m)
        RMSE(1) = MatCompRMSE(V1', U1, tempS, para.test.row, para.test.col, para.test.data);
        trainRMSE(1) = sqrt(sum(part0.^2)/length(data));
    else
        RMSE(1) = MatCompRMSE(U1, V1', tempS, para.test.row, para.test.col, para.test.data);
        trainRMSE(1) = sqrt(sum(part0.^2)/length(data));
    end
    fprintf('method: %s data: %s  RMSE %.2d \n', output.method, para.data, RMSE(1));
end

C = 0.99;
c = 1;
norm1 = norm(U1,'fro')^2 + norm(V1,'fro')^2;
obj_1 = c_1*0.25*(norm1^2) + c_2*0.5*norm1;
obj_0 = obj_1; 
x_obj = (1/2)*sum(part0.^2);
obj(1) = x_obj + lam*(sum(sum(1 - exp(-ga*abs(U0)))) + sum(sum(1 - exp(-ga*abs(V0)))));

for i = 1:maxIter
    
    tt = cputime;
    cp = c;
    c = (1 + sqrt(1+4*c^2))/2;
    bi = (cp - 1)/c;

    grad_h1_U = U1*norm1;
    grad_h1_V = V1*norm1;

    grad_h_U = c_1*grad_h1_U + c_2*U1;
    grad_h_V = c_1*grad_h1_V + c_2*V1;
        
    delta_U = U0 - U1;
    delta_V = V0 - V1;

    D_x = obj_0 - obj_1 - sum(sum(delta_U.*grad_h_U)) - sum(sum(delta_V.*grad_h_V));
    
    for il = 1:300
        kappa = C*uL/(lL+uL);
        for ibi = 1:300
            y_U = (1+bi)*U1 - bi*U0;
            y_V = (1+bi)*V1 - bi*V0;
            
            norm_y = norm(y_U,'fro')^2 + norm(y_V,'fro')^2;
            grad_h1_yU = y_U*norm_y;
            grad_h1_yV = y_V*norm_y;

            grad_h2_yU = y_U;
            grad_h2_yV = y_V;

            grad_h_yU = c_1*grad_h1_yU + c_2*grad_h2_yU;
            grad_h_yV = c_1*grad_h1_yV + c_2*grad_h2_yV;
            
            obj_y = c_1*0.25*(norm_y^2) + c_2*0.5*norm_y;
           
            D_y = obj_1 - obj_y - bi*sum(sum(delta_U.*grad_h_yU)) - bi*sum(sum(delta_V.*grad_h_yV));
            
            if D_y <= kappa*D_x + 1e-10
                break;
            else
                bi = 0.9*bi;
            end
    
        end
        
        part1 = sparse_inp(y_U', y_V, row, col);
        
        part0 = data-part1';
        
        y_obj = (1/2)*sum(part0.^2); 
        
        setSval(spa,part0,length(part0));
    
        grad_U = -spa*y_V';
        grad_V = -y_U'*spa;
        
        if x_obj + lL*D_y >= y_obj + bi*sum(sum(delta_U.*grad_U)) + bi*sum(sum(delta_V.*grad_V)) - 1e-10
            break;
        else
            lL = 1.1*lL;
        end
    end
    
    
    
    

    U0 = U1;
    V0 = V1;
    
    
    
    
    
% --------------------------
    w_U = lam*ga*(1-exp(-ga*abs(U1)));
    w_V = lam*ga*(1-exp(-ga*abs(V1)));
  
    
    
% --------------------------
    obj_0 = obj_1; 
    for inneriter = 1:maxinneriter
      
    % update U, V 
        [U1, V1] = make_update(grad_U,grad_V,grad_h_yU,grad_h_yV,c_1,c_2,uL,w_U,w_V, 4);

        norm1 = norm(U1,'fro')^2 + norm(V1,'fro')^2;

        obj_1 = c_1*0.25*(norm1^2) + c_2*0.5*norm1;

        part0 = sparse_inp(U1', V1, row, col);

        part0 = data - part0';

        x_obj = (1/2)*sum(part0.^2); 
        delta_U = U1 - y_U;
        delta_V = V1 - y_V;
        reg = obj_1 - obj_y - sum(sum(delta_U.*grad_h_yU)) - sum(sum(delta_V.*grad_h_yV));
        
        if(x_obj <= y_obj + sum(sum(delta_U.*grad_U)) + sum(sum(delta_V.*grad_V)) + uL*reg + 1e-10)
            break;
        else
            uL = 1.1*uL;
        end
    end

    Lls(i+1) = lL;
    Ils(i+1) = uL;
    
    
    % ----------------------

    if(i > 1)
        delta = (obj(i)- x_obj)/x_obj;
    else
        delta = inf;
    end
    
    Time(i+1) = cputime - tt;
    obj(i+1) = x_obj + lam*(sum(sum(1 - exp(-ga*abs(U1)))) + sum(sum(1 - exp(-ga*abs(V1)))));
    
    fprintf('iter: %d; obj: %.3d (dif: %.3d); rank %d; lambda: %.1f; L %d; time: %.3d;  nnz U:%0.3d; nnz V %0.3d \n', ...
        i, x_obj, delta, para.maxR, ibi, uL, Time(i+1), nnz(U1)/(size(U1,1)*size(U1,2)),nnz(V1)/(size(V1,1)*size(V1,2)));
    
    nnzUV(i+1,1) = nnz(U1)/(size(U1,1)*size(U1,2));
    nnzUV(i+1,2) = nnz(V1)/(size(V1,1)*size(V1,2));
    
    if(isfield(para, 'test'))
        tempS = eye(size(U1,2), size(V1',2));
        if(para.test.m ~= m)
            RMSE(i+1) = MatCompRMSE(V1', U1, tempS, para.test.row, para.test.col, para.test.data);
            trainRMSE(i+1) = sqrt(sum(part0.^2)/length(data));
        else
            RMSE(i+1) = MatCompRMSE(U1, V1', tempS, para.test.row, para.test.col, para.test.data);
            trainRMSE(i+1) = sqrt(sum(part0.^2)/length(data));
        end
        fprintf('method: %s data: %s  RMSE %.2d \n', output.method, para.data, RMSE(i));
    end
    
    if(i > 1 && abs(delta) < tol)
        break;
    end
    
    if(sum(Time) > para.maxtime)
        break;
    end
end

output.obj = obj(1:(i+1));
output.Rank = para.maxR;
output.RMSE = RMSE(1:(i+1));
output.trainRMSE = trainRMSE(1:(i+1));

Time = cumsum(Time);
output.Time = Time(1:(i+1));
output.U = U1;
output.V = V1;
output.data = para.data;
output.L = Lls(1:(i+1));
output.Ils = Ils(1:(i+1));
output.nnzUV = nnzUV(1:(i+1),:);
output.no_acceleration = no_acceleration(1:(i+1));
output.lambda = lam;
output.theta = ga;
output.reg = para.reg;


end


