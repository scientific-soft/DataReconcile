function reconciled_with_Kernels = DRnonparamEqRobust(Func, msrd_data, model_params, prior_vars, bandwidths)
% Func - указатель на функцию, описывающую зависимости между согласуемыми
% величинами
% msrd_data - вектор согласуемых результатов измерений вида (x_msdr1, x_msdr2, ..., x_msdrN ) 
% model_params - вектор параметров модели взаимосвязей, погрешности которых
% не заданы
% prior_vars - априорно заданные параметры неопределенности согласуемых
% величин
% bandwidths - вектор значений ширины 
    l = zeros([length(Func(msrd_data, model_params)), 1]);
    mu_and_l_start = mean(msrd_data, 2);
    for i = 1 : length(l)
        mu_and_l_start(length(mu_and_l_start)+1) = l(i);
    end

    % --- Robustness
    MAD = zeros([length(mean(msrd_data, 2)),1]);
    for jj = 1 : length(mean(msrd_data, 2))
        MAD(jj) = mad(msrd_data(jj,:), 1);
    end
    error_params = MAD .* 1.483;
    spn1 = eps; % Machine zero in Matlab
    % If calculated std is zero, then we use the external std value   
    for jj = 1 : length(mean(msrd_data, 2))
        if abs(error_params(jj)) < spn1
            error_params(jj) = prior_vars(jj);
        end
    end
    % ---


    mu_and_l_est = fsolve(@KernelsSystemToSolve, mu_and_l_start, [], ...
              Func, msrd_data, model_params, bandwidths, error_params);
    reconciled_with_Kernels =  mu_and_l_est(1:(length(mu_and_l_est)-length(l)));
end

function K = Kernel(x)
    K = exp(-x.^2 ./ 2)./sqrt(2*pi);
end

function mustbezeros = KernelsSystemToSolve(mu_and_l, DependFunc, ...
                                            msrd_data, model_params, ...
                                            bandwidths, msrd_vars)
    xNum = size(msrd_data, 1);
    n = size(msrd_data, 2);
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

        df_dx = imag(DependFunc(mu_and_l(1:xNum)+imagx, model_params))/imag(imagx(j));
        df_dx = reshape(df_dx,1, []);
        l_ = mu_and_l((xNum+1):length(mu_and_l));
        l_ = reshape(l_, 1, []);

        mustbezeros(j) = mean_msrd - mean_msrd - mean_via_kerns + mu_and_l(j) ...
                      + bandwidths(j)^2*(1/n)*s_msrd^2*sum(df_dx.*l_);
    end
    model_res = DependFunc(mu_and_l(1:xNum), model_params);
    for i = 1 : length(model_res)
        mustbezeros(length(mustbezeros)+1) = model_res(i);
    end
end
