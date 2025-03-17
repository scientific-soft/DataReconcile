% Аналитическая модель решения задачи Data Reconciliation
clear all;
close all hidden;

% Пример использования: 
% x = [x1;x2]; результат измерения x = [1.0; 0.9; 0.9];
% линейная связь вида 2*x1+3*x2=5; 1*x1+1*x2+1*x3=3
% групповое ограничение: недопустимо R2 при отклонении от столбца [1.0;1.0;1.0], большее, чем 0.2.  

x = [1.0; 0.9; 0.9];
A = [2, 3, 0
     1, 1, 1];
b = [5; 3];
c = [1.0; 1.0; 1.0];
r2 = 0.2;
% уточненный результат Data Reconciliation
y = AnalyticalModel(x, A, b, c, r2);

dx = [0.1; 0.1; 0.1]; % пределы абсолютной погрешности (FuzzyInterval)
dy = GetUncertainty(x, dx, A, b, c, r2);
1;

function y = AnalyticalModel(x, A_y, b_y, c_y, r2_y)
    % Гауссова модель распределения погрешностей.
    
    % x - вектор уточняемых результатов измерений,
    % A_y - система линейных ограничений на уравнения связи для результатов уточнения, by - вектор правых частей этих уравнений: A_y*y=b_y;
    %     т.к. y=x+dx, то A_y*(x+dx)=b_y. Следовательно, A_y*dx=b_y-A_y*x. Значит, в модели Ефремова-Козлова А=A_y, b=b_y-A_y*x.
    % с_y и r2_y - совокупное ограничение неравенством (y-c_y)'*(y-c_y)<r2_y. Т.к. y=x+dx, то (x+dx-c_y)'*(x+dx-c_y)<r2_y. 
    % Следовательно, (dx-(c_y-x))'*(dx-(c_y-x))<r2_y. Значит, в модели Ефремова-Козлова c=c_y-x, r2=r2_y.
    % y - результат уточнения
    
    x = x(:);  % принудительное приведение к вектор-столбцу
    b_y = b_y(:);  % принудительное приведение к вектор-столбцу
    c_y = c_y(:);  % принудительное приведение к вектор-столбцу

    A = A_y;
    b = b_y-A_y*x;
    c = c_y-x;
    r2 = r2_y;

    % Проверки:
    n = length(x);
    if size(c,1) ~= n
        error('Размерность векторов х и с должны совпадать.');
    end
    if size(A,2) ~= n
        error('Число столбцов в матрице А должно совпадать с числом элементов в х.');
    end
    if size(A,1) ~= size(b,1)
        error('Число строк в матрице А должно совпадать с числом элементов в b.');
    end

    P = transpose(A)*inv(A*transpose(A));
    D = eye(n)-P*A;

    a = transpose(b)*(transpose(P)*P)*b-transpose(b)*transpose(P)*c-transpose(c)*P*b+transpose(c)*c-transpose(c)*D*c-r2;
    h = 2*a;
    d = a+transpose(c)*D*c;
    lbd = roots([a;h;d]);
    
%     if max(abs(imag(lbd))./real(lbd))<10^-3
%        lbd = real(lbd);
%     else
%        error('Ограничения противоречат друг другу.'); 
%     end
    dx = NaN(n,2);
    for i = 1:2
        if (lbd(i) == 0) || (lbd(i) == -1)
            dx(:,i) = P*b;
        else
            dx(:,i) = P*b+lbd(i)*D*c/(1+lbd(i));
        end 
    end
    [~,ind] = min(sum(dx.^2,1));
    y = x+dx(:,ind);
end

function dy = GetUncertainty(x, dx, A_y, b_y, c_y, r2_y)
    % Все обозначения - те же, что и для AnalyticalModel,
    % dx - массив-вектор cell неопределенностей, формализованных по одному из типов.
    n = length(dx);
    % оценка частных производных
    alpha = 10^-100;
    dy_dx = NaN(n,n);
    for i = 1 : n
        d = zeros(n,1);
        d(i) = 1i*alpha;
        dy_dx(:,i) = imag(AnalyticalModel(x+d, A_y, b_y, c_y, r2_y))/alpha;    
    end
   
    % цикл по операциям сложения и умножения по типу неопределенности
    for i = 1 : n
        dy(i) = abs(dy_dx(i,1)) * dx(1) + abs(dy_dx(i,2)) * dx(2);
        for j = 3 : n
            dy(i) = dy(i) + abs(dy_dx(i,j)) * dx(j);
        end
    end
end