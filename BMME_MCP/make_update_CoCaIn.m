function[U, V] = make_update_CoCaIn(U,V,grad_U,grad_V,grad_h_U,grad_h_V,c_1,c_2,L,lam,ga)
    
    tpk = grad_U/L - grad_h_U;
    tqk = grad_V/L - grad_h_V;
    
    for i = 1:300
        
        w_U = lam*ga*(1-exp(-ga*abs(U)));
        w_V = lam*ga*(1-exp(-ga*abs(V)));
        
        pk =  max(0,abs(tpk) - w_U/L).*sign(-tpk);
        qk =  max(0,abs(tqk) - w_V/L).*sign(-tqk);


        % solve cubic equation:
        coeff = [c_1*(norm(pk,'fro')^2 + norm(qk,'fro')^2), 0, c_2, -1];
        temp = roots(coeff);
    %     fprintf('root %.2d \n', temp);
        if(length(temp)==3)
           r = temp(3); 
        else
            r = temp;
        end
        U0 = U;
        V0 = V;
        U = r*pk;
        V = r*qk;
        norm1 = norm(U-U0,'fro') + norm(V-V0,'fro');
        if norm1 < 1e-6
            break;
        end
    end

end