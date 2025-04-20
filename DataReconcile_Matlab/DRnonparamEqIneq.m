function reconciled_with_Kernels = DRnonparamEqIneq(eqFunc, ineqFunc, msrd_data, model_params, prior_vars, bandwidths)
    % eqFunc - указатель на функцию f, которая описывает взаимосвязи между
    % совместно измеряемыми величинами
% msrd_data - вектор согласуемых результатов измерений вида (x_msdr1, x_msdr2, ..., x_msdrN ) 
% model_params - вектор параметров модели взаимосвязей, погрешности которых
% не заданы
% prior_vars - априорно заданные параметры неопределенности согласуемых
% величин
% bandwidths - вектор значений ширины 
    eqNum = length(eqFunc(mean(msrd_data,2), model_params));
    ineqNum = length(ineqFunc(mean(msrd_data,2), model_params));
    xNum = size(msrd_data, 1);
    n = size(msrd_data, 2);

    l = zeros([eqNum, 1]);
    % расширяем вектор l на число неравенств
    l = [ l; zeros([ineqNum, 1]) ];
    mu_and_l_start = mean(msrd_data,2);
    for i = 1 : length(l)
        mu_and_l_start(length(mu_and_l_start)+1) = l(i);
    end
    
    mu_and_l_est = fsolve(@(mu_and_l_start) KernelsSystemToSolve (mu_and_l_start, ...
              eqFunc, ineqFunc, msrd_data, model_params, bandwidths, prior_vars, xNum, n, eqNum, ineqNum), mu_and_l_start);
    reconciled_with_Kernels =  mu_and_l_est(1:(length(mu_and_l_est)-length(l)));
end

function K = Kernel(x)
    K = exp(-x.^2 ./ 2)./sqrt(2*pi);
end

function mustbezeros = KernelsSystemToSolve(mu_and_l, eqFunc, ineqFunc, ...
                                            msrd_data, params_, ...
                                            bandwidths, msrd_vars, xNum, n, eqNum, ineqNum)
    mustbezeros = zeros([xNum, 1]);
    if length(bandwidths) == 1
        bandwidths = zeros([1, xNum]) + bandwidths;
    end
    for j = 1 : xNum
        mean_msrd = mean(msrd_data(j,:));
        s_msrd = msrd_vars(j);
        res = zeros([1,n]);
        for i = 1:n
            denumerator = Kernel(( (msrd_data(j,:)-mean_msrd)./s_msrd - (msrd_data(j,i)-mu_and_l(j))./s_msrd )./bandwidths(j));
            denumerator = reshape(denumerator, 1, []); % делаем так, чтобы точно была строка!
            numerator = sum(msrd_data(j,:).*denumerator);
            if sum(denumerator) == 0
                res(i) = 0;
            else
                res(i) = numerator/sum(denumerator);
            end
        end
        mean_via_kerns = mean(res);
        imagx = zeros([xNum,1]); 
        imagx(j) = 1i*( mu_and_l(j)*10^(-100) + 10^(-101) );

        df_eq_dx(:) = imag(eqFunc(mu_and_l(1:xNum)+imagx,params_))/imag(imagx(j));
        df_eq_dx = reshape(df_eq_dx,1, []);
        df_ineq_dx(:) = imag(ineqFunc(mu_and_l(1:xNum)+imagx,params_))/imag(imagx(j));
        df_ineq_dx = reshape(df_ineq_dx, 1, []);
        
        l_ = mu_and_l((xNum+1):length(mu_and_l));
        % all ineqalities multipilers in [l] have to be >=0, so 
        l_( (eqNum+1):length(l_) ) = ( l_((eqNum+1):length(l_)) ).^2;

        df_dx = [df_eq_dx, df_ineq_dx];
        l_ = reshape(l_, 1, []);

        mustbezeros(j) = mean_msrd - mean_msrd - mean_via_kerns + mu_and_l(j) ...
                      + bandwidths(j)^2*(1/n)*s_msrd^2*sum(df_dx.*l_);
    end

    model_eq_res = eqFunc(mu_and_l(1:xNum),params_);
    for i = 1 : length(model_eq_res)
        mustbezeros(xNum+i) = model_eq_res(i);
    end
    model_ineq_res = ineqFunc(mu_and_l(1:xNum),params_);
    for i = 1 : length(model_ineq_res)
        mustbezeros(xNum+eqNum+i) = model_ineq_res(i);
    end
    model_ineq_res = ineqFunc(mu_and_l(1:xNum),params_);
    for i = 1 : length(model_ineq_res)
        mustbezeros(xNum+eqNum+ineqNum+i) = model_ineq_res(i)*l_(eqNum+i);
    end
end
