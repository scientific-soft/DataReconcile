function reconciled = DRparamEq(Func, msrd_data, error_params, params_)
    % Func - указатель на функцию f, которая описывает взаимосвязи между
    % совместно измеряемыми величинами
    % в виде уравнения f(x1, x2, ..., a1, a2, ...) = 0
    % либо в виде системы уравнений F(X, A) = 0_m, где 
    % 0_m - вектор-столбец длиной m, заполненный нулями. 
    % msrd_data - вектор согласуемых результатов измерений
    % error_params - апприорно заданные дисперсии/пределы погрешностей (var1, ..., varN)
    % точно известные параметры модели зависимостей
    % params_ - параметры модели, погрешности которых не заданы
    l = zeros([length(Func(mean(msrd_data,2), params_)), 1]);
    mu_and_l_start = mean(msrd_data,2);
    for i = 1 : length(l)
        mu_and_l_start(length(mu_and_l_start)+1) = l(i);
    end
    mu_and_l_res = fsolve(@SystemToSolve, mu_and_l_start, [], ...
      Func, msrd_data, error_params, params_);
    reconciled = mu_and_l_res(1:(length(mu_and_l_res)-length(l)));
end
function mustbezeros = SystemToSolve(mu_and_l, DependFunc, msrd_data, error_params, params_)
    vars = error_params;
    xNum = size(msrd_data, 1);
    n = size(msrd_data, 2);
    mustbezeros = zeros([xNum, 1]);
    for j = 1 : xNum
        if vars(j) == 0
            fprintf("None of the error parameters should be 0. " + ...
                " Note that all constant (without error) dependency model parameters need to be in the true_params array of " + ...
                "GramChDataReconcil(Func, msrd_data, error_params, true_params).");
            return
        else
            imagx = zeros([xNum,1]); 
            imagx(j) = 1i*(mu_and_l(j)*10^(-100) + 10^(-101) );
            df_dx(:) = imag(DependFunc(mu_and_l(1:xNum)+imagx,params_))/imag(imagx(j));

            df_dx = reshape(df_dx,1, []);
            l_ = mu_and_l((xNum+1):length(mu_and_l));
            l_ = reshape(l_, 1, []);
            mustbezeros(j) = mu_and_l(j)/vars(j) - mean(msrd_data(j,:))/vars(j)...
            + sum(l_.*df_dx)/n;
        end
    end
    model_res = DependFunc(mu_and_l(1:xNum),params_);
    for i = 1 : length(model_res)
        mustbezeros(length(mustbezeros)+1) = model_res(i);
    end
end
