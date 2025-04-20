function reconciled = DRparamEqIneqRobust(eqFunc, ineqFunc, msrd_data, error_params, model_params)
    % eqFunc - указатель на функцию f, которая описывает взаимосвязи между
    % совместно измеряемыми величинами
    % в виде уравнения f(x1, x2, ..., a1, a2, ...) = 0
    % либо в виде системы уравнений F(X, A) = 0_m, где 
    % 0_m - вектор-столбец длиной m, заполненный нулями. 
    % ineqFunc - указатель на неравнества
    % msrd_data - вектор согласуемых результатов измерений
    % error_params - апприорно заданные дисперсии/пределы погрешностей (var1, ..., varN)
    % точно известные параметры модели зависимостей
    % model_params - параметры модели, погрешности которых не заданы,
    % причем как параметры равенств, так и неравенств- в порядке на
    % усмотрение пользователя, который составляет уравнения и неравенства
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
    
    for jj = 1 : length(error_params)
        MAD(jj) = mad(msrd_data(jj,:), 1);
    end
    error_params = MAD .* 1.483;
    mu_and_l_res = fsolve(@(mu_and_l_start) SystemToSolve (mu_and_l_start, ...
      eqFunc, ineqFunc, msrd_data, error_params, model_params, xNum, n, eqNum, ineqNum), mu_and_l_start);
    reconciled = mu_and_l_res(1:(length(mu_and_l_res)-length(l)));
end


function mustbezeros = SystemToSolve(mu_and_l, eqFunc, ineqFunc, ...
                         msrd_data, vars, params_, xNum, n, eqNum, ineqNum)
    mustbezeros = zeros([xNum, 1]);
    for j = 1 : xNum
        if vars(j) == 0
            fprintf("None of the error parameters should be 0. " + ...
                " Note that all constant (without error) dependency model parameters need to be in the true_params array of " + ...
                "DataReconcil().");
            return
        else
            imagx = zeros([xNum,1]); 
            imagx(j) = 1i*(mu_and_l(j)*10^(-100) + 10^(-101) );

            df_eq_dx = imag(eqFunc(mu_and_l(1:xNum)+imagx,params_))/imag(imagx(j));
            df_eq_dx = reshape(df_eq_dx,1, []);
            df_ineq_dx = imag(ineqFunc(mu_and_l(1:xNum)+imagx,params_))/imag(imagx(j));
            df_ineq_dx = reshape(df_ineq_dx, 1, []);
            
            l_ = mu_and_l((xNum+1):length(mu_and_l));
            % all ineqalities multipilers in [l] have to be >=0, so 
            l_( (eqNum+1):length(l_) ) = ( l_((eqNum+1):length(l_)) ).^2;

            df_dx = [df_eq_dx, df_ineq_dx];
            l_ = reshape(l_, 1, []);

            mustbezeros(j) = mu_and_l(j)/vars(j) - mean(msrd_data(j,:))/vars(j)...
            + sum(l_.*df_dx)/n;
        end
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
