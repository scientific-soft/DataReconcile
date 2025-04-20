function reconciled_with_AlphaGramCh = DRsemiparamEqRobust(Func, msrd_data, error_params, alpha_params, true_params)
    % Func - указатель на функцию f, которая описывает взаимосвязи между
    % совместно измеряемыми величинами
    % в виде уравнения f(x1, x2, ..., a1, a2, ...) = 0
    % либо в виде системы уравнений F(X, A) = 0_m, где 
    % 0_m - вектор-столбец длиной m, заполненный нулями. 
    % msrd_data - вектор согласуемых результатов измерений
    % prior_params - заданные параметры погрешностей (var1, ..., varN) либо
    % аналоги
    % params_ - известные параметры модели зависимостей
    % alpha_params - массив параметров функции альфа (As, Ex),
    % Либо (As1, Ex1, As2, Ex2, ..., AsN, ExN), где N - число
    % согласуемых величин
    l = zeros([length(Func(msrd_data, true_params)), 1]);
    mu_and_l_start = mean(msrd_data,2);
    for i = 1 : length(l)
        mu_and_l_start(length(mu_and_l_start)+1) = l(i);
    end
    for jj = 1 : length(error_params)
        MAD(jj) = mad(msrd_data(jj,:), 1);
    end
    error_params = MAD .* 1.483;
    
    mu_and_l_res = fsolve(@(mu_and_l_start) AlphaSystemToSolve (mu_and_l_start, ...
      Func, msrd_data, error_params, alpha_params, true_params), mu_and_l_start);

    reconciled_with_AlphaGramCh = mu_and_l_res(1:(length(mu_and_l_res)-length(l)));
end
function mustbezeros = AlphaSystemToSolve(mu_and_l, DependFunc, msrd_data, ...
    error_params, alpha_params, true_params)
    vars = error_params;
    xNum = size(msrd_data, 1);
    n = size(msrd_data, 2);
    mustbezeros = zeros([xNum, 1]);
    for j = 1 : xNum
        if vars(j) == 0
            fprintf("None of the error parameters should be 0. " + ...
                " Note that all constant (without error) dependency model parameters need to be in the true_params array of " + ...
                "GramChDataReconcil(Func, msrd_data, error_params, alpha_params, true_params).");
            return
        else
            if length(alpha_params) == 2
                imagx = zeros([xNum,1]); 
                imagx(j) = 1i*(mu_and_l(j)*10^(-100) + 10^(-101) );
                
                df_dx = imag(DependFunc(mu_and_l(1:xNum)+imagx,true_params))/imag(imagx(j));
                df_dx = reshape(df_dx,1, []);
                l_ = mu_and_l((xNum+1):length(mu_and_l));
                l_ = reshape(l_, 1, []);

                mustbezeros(j) = mu_and_l(j)/vars(j) - mean(msrd_data(j,:))/vars(j)...
                - calculMeanAlpha(mu_and_l(j), msrd_data(j,:), alpha_params, vars(j))...
                + sum(l_.*df_dx)/n;
            else
                imagx = zeros([xNum,1]); 
                imagx(j) = 1i*(mu_and_l(j)*10^(-100) + 10^(-101) );

                df_dx = imag(DependFunc(mu_and_l(1:xNum)+imagx,true_params))/imag(imagx(j));
                df_dx = reshape(df_dx,1, []);
                l_ = mu_and_l((xNum+1):length(mu_and_l));
                l_ = reshape(l_, 1, []);

                mustbezeros(j) = mu_and_l(j)/vars(j) - mean(msrd_data(j,:))/vars(j)...
                - calculMeanAlpha(mu_and_l(j), msrd_data(j,:), alpha_params((j*2-1):(j*2)), vars(j))...
                + sum(l_.*df_dx)/n;
            end
        end
    end
    model_res = DependFunc(mu_and_l(1:xNum), true_params);
    for i = 1 : length(model_res)
        mustbezeros(length(mustbezeros)+1) = model_res(i);
    end
end
function alpha_mean = calculMeanAlpha(mj, msrd_data, alpha_params, sj)
   As = alpha_params(1);
   Ex = alpha_params(2);
   tji = (msrd_data - mj)./sj;
   d_alpha_d_mu = (Ex - 3).*(1/2.*tji - 1/6.*tji.^3) + As.*(1/2 - 1/2.*tji.^2);
   alpha_plus_one = (Ex - 3).*(1/24.*tji.^4 + 1/8 - 1/4.*tji.^2) + As.*(1/6.*tji.^3 - 1/2.*tji) + 1;
   y = d_alpha_d_mu./alpha_plus_one;
   alpha_mean = (1/sj)*mean(y);
end
