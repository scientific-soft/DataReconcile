% Преобразование разных типов представления неточных данных к формату,
% унифицированному для дальнейшего применения Data Reconciliation.

% Класс для одномерной выборки
classdef Sample
    properties
        x           % значения в выборке
    end
    
    methods
        % конструктор
        function obj = Sample(x)
            % проверка размерностей входных данных
            if isempty(x)
                error('В выборке должно быть хотя бы одно значение.');
            end
            if any(isnan(x))
                error('Выборка не может содержать значения типа NaN.');
            end
            obj.x = x(:)';               % принудительный перевод в вектор-строку
        end
        
        % оператор сложения двух выборок (операнды: Z = A + B)
        function Z = plus(A, B)
            if ~isa(A, 'Sample') || ~isa(B, 'Sample')
                error('Оба операнда должны быть объектами типа Sample.');
            end
            % проверка размерности выборок
            n1 = length(A.x);
            n2 = length(B.x);
            if n1 == n2
               z_values = A.x+B.x;
            elseif n1<n2
               % дополняем значения из первой выборки бутстрепом
               z_values = B.x;
               z_values(1:n1) = z_values(1:n1) + A.x;
               z_values(n1+1:end) = z_values(n1+1:end) + A.x(randi(n1, [1,n2-n1]));
            else
               % дополняем значения из второй выборки бутстрепом
               z_values = A.x;
               z_values(1:n2) = z_values(1:n2) + B.x;
               z_values(n2+1:end) = z_values(n2+1:end) + B.x(randi(n2, [1,n1-n2]));
            end
            Z = Sample(z_values);
        end
        
        function Z = mtimes(k, A)
            if ~isa(A, 'Sample') && ~isa(k, 'double')
                error('Операция умножения задана только для перемножения с коэффициентом.');
            end
            Z = Sample(k*A.x);
        end

        function stdCint = getStd(obj, method)
            n = length(obj.x);
            if (nargin < 2) || ((nargin == 2) && strcmp(method,'approximate'))
               x = sort(obj.x);
               % согласно теореме Крейновича-Тао.
               q = [0.05, 0.95];    % искомые квантили.
               for i = 1 : 2
                  k1 = max(floor(n*q(i)-sqrt(n*q(i)*(1-q(i)))*norminv(0.975,0,1)), 1);
                  k2 = min(ceil(n*q(i)+sqrt(n*q(i)*(1-q(i)))*norminv(0.975,0,1)), n);
                  left(i) = x(k1);
                  right(i) = x(k2);
               end
               stdCint = 0.25*[left(2)-right(1), right(2)-left(1)];            
            elseif strcmp(method,'accurate')
               s = std(obj.x);
               if n < 10 % используем бутстреп
                    M = 100; % количество повторений 
                    buf = zeros(M,1);                    
                    for i = 1:M
                        % генерация бутстреп-выборки с заменой
                        sample = obj.x(randi(n,[1,n]));
                        % вычисление дисперсии для текущей выборки
                        buf(i) = var(sample); % 1 - смещенная дисперсия
                    end                    
                    % оценка доверительного интервала для значений дисперсии
                    stdCint = sqrt([quantile(buf, 0.05, 'Method','approximate'), quantile(buf, 0.95, 'Method','approximate')]);
               else
                   % асимптотический доверительный интервад
                   stdCint = s * sqrt(n-1) ./ sqrt([chi2inv(0.975,n-1), chi2inv(0.025,n-1)]);
               end
            else
                error('Аргумент method не задан. Допустимые значения: approximate, accurate.');
            end
        end

        % построение графика ecdf
        function plot(obj)
            figure; hold on;
            h = cdfplot(obj.x);
            set(h, 'Color', 'b', 'LineWidth', 1.5); 
            xlabel('x'); ylabel('CDF');
            legend('Эмпирическая функция распределения', 'Location', 'southeast');
            title('Выборочная функция распределения');
            grid on;
        end

        % преобразование в PBox
        function pbox = Sample2PBox(obj)
            [~, X, Flo, Fup] = ecdf(obj.x);
            pbox = PBox(X, Flo, Fup);
        end

        % преобразование в Hist
        function Histogram = Sample2Hist(obj, binsNum)
            n = length(obj.x);
            if nargin < 2
               binsNum = ceil(1+1.59*log(n));
            end
            [m, edges] = histcounts(obj.x, binsNum);
            [~,cints] = binofit(m,n,0.05);
            lowerPDF = cints(:,1)./diff(edges');
            upperPDF = cints(:,2)./diff(edges');
            Histogram = Hist(edges, lowerPDF, upperPDF);
        end

        % преобразование в DempsterShafer
        function ds = Sample2DempsterShafer(obj, numFocal)
            n = length(obj.x);
            if nargin < 2
               numFocal = ceil(1+1.59*log(n));
            end
            % кластеризация выборки в фокальные элементы
            [~, centers] = kmeans(obj.x(:), numFocal);
            centers = sort(centers);
            
            % определение интервалов вокруг центров кластеров
            ranges = diff(centers)/2;
            ranges = [ranges(1); ranges; ranges(end)];
            
            intervals = zeros(numFocal, 2);
            for i = 1:numFocal
                intervals(i,:) = [centers(i)-ranges(i), centers(i)+ranges(i+1)];
            end
            
            % вычисление масс по числу точек, попавших в фокальные элементы
            counts = histcounts(obj.x, [intervals(1,:),intervals(2:end,2)']);
            masses = counts/n;
            
            ds = DempsterShafer(intervals, masses/sum(masses));
        end

        % преобразование в Fuzzy
        function fv = Sample2Fuzzy(obj, numPoints)
            n = length(obj.x);
            if nargin < 2
               numPoints = ceil(1+1.59*log(n));
            end
            % оценка PDF через KDE
            [pdf_est, x_vals] = ksdensity(obj.x, 'NumPoints', numPoints);
            % преобразование PDF в функцию принадлежности
            membership = pdf_est / max(pdf_est);
            fv = Fuzzy(x_vals, membership);
        end 

        % преобразование в FuzzyInterval
        function fi = Sample2FuzzyInterval(obj, numAlpha)
            if nargin < 2
               numAlpha = 5;
            end
            alphaLevels = linspace(0, 1, numAlpha);
            alphaLevels = alphaLevels(end:-1:1);
            intervals = zeros(length(alphaLevels), 2);
            
            for i = 1:length(alphaLevels)
                alpha = alphaLevels(i);
                p = [(1-alpha)/2, (1+alpha)/2];
                intervals(i,:) = quantile(obj.x, p, 'Method', 'approximate');
            end
            
            fi = FuzzyInterval(alphaLevels, intervals(end:-1:1,:));
        end
    end
end
