% Преобразование разных типов представления неточных данных к формату,
% унифицированному для дальнейшего применения Data Reconciliation.

% Класс для Hist (дискретизированная гистограмма, pdf)
classdef Hist   % дискретизированная гистограмма
    properties
        x           % границы полос гистограммы (длина на единицу больше, чем в переменных для полос)
        lowerPDF    % нижние границы высоты полосы гистограммы (наименьшие возможные значения pdf)
        upperPDF    % верхние границы высоты полосы гистограммы (наибольшие возможные значения pdf)
    end
    
    properties (Constant)
        tolerance_threshold = 0.1;
    end

    methods
        % конструктор
        function obj = Hist(x, lowerPDF, upperPDF)
            % проверка размерностей входных данных
            if length(x) < 2
                error('В гистограмме должна быть хотя бы одна полоса.');
            end
            if length(x) ~= length(lowerPDF)+1 || length(x) ~= length(upperPDF)+1
                error('lowerPDF и upperPDF должны быть векторами одинакового размера, x иметь на одно значение больше.');
            end
            if any(lowerPDF > upperPDF)
                error('Нижние границы полос гистограммы не должны превышать верхние границы гистограммы для всех значений аргументов.');
            end
            if any(lowerPDF < 0) || any(upperPDF < 0)
                error('Значения высоты полос гистограммы не могут быть меньше 0.');
            end
            obj.x = x(:)';               % принудительный перевод в вектор-строку
            obj.lowerPDF = lowerPDF(:)'; % принудительный перевод в вектор-строку
            obj.upperPDF = upperPDF(:)'; % принудительный перевод в вектор-строку

            % сильная проверка условия нормировки: в предположении, что границы вероятности определены асимптотическим доверительным интервалом
            threshold = 10^-2;
            if abs(sum(mean([obj.lowerPDF; obj.upperPDF], 1).*diff(obj.x))-1) > threshold
                warning('Площадь гистограммы не равна 1.0 при условии симметричности границ значений PDF.');
            end
            % слабая проверка условия нормировки: хотя бы при каком-то сочетании значений PDF нормировка должна выполняться
            if (sum(obj.lowerPDF.*diff(obj.x))>1) || (sum(obj.upperPDF.*diff(obj.x))<1)
                warning('Площадь гистограммы не равна 1.0.');
            end
            % значения вероятностей не должны быть больше единицы
            if any(obj.upperPDF.*diff(obj.x)>1)
                warning('Площади полос гистограммы не могут быть больше 1.0.');
            end
        end
        
        % оператор сложения двух гистограмм (операнды: Z = A + B)
        function Z = plus(A, B)
            if ~isa(A, 'Hist') || ~isa(B, 'Hist')
                error('Оба операнда должны быть объектами типа Hist.');
            end
            
            % Генерация всех возможных значений суммы границ полос гистограмм
            [A_edges, B_edges] = meshgrid(A.x, B.x);
            all_edges = A_edges(:) + B_edges(:);
            
            % Во избежание комбинаторного взрыва, объединяем близкие границы
            sum_edges = uniquetol(sort(all_edges), A.tolerance_threshold);
            
            lower_pdf = zeros(1, length(sum_edges)-1);
            upper_pdf = zeros(1, length(sum_edges)-1);
            
            % Определение вероятностей, соответствующих попаданию в полосы
            A_lowerProb = A.lowerPDF .* diff(A.x);
            A_upperProb = A.upperPDF .* diff(A.x);
            B_lowerProb = B.lowerPDF .* diff(B.x);
            B_upperProb = B.upperPDF .* diff(B.x);
            
            % Вычисление пределов вероятности для каждой полосы итоговой гистограммы
            for k = 1:(length(sum_edges)-1)
                z_min = sum_edges(k);
                z_max = sum_edges(k+1);
                
                total_lower = 0;
                total_upper = 0;
                
                % Цикл по всем возможным парам полос гистограмм-операндов
                for i = 1:(length(A.x)-1)
                    for j = 1:(length(B.x)-1)
                        % Определение возможных сумм для границ операндов
                        sum_min = A.x(i) + B.x(j);
                        sum_max = A.x(i+1) + B.x(j+1);
                        
                        % Вычисление доли пересечения (в стандартном предположении равномерного распределения внутри полосы)
                        overlap_min = min(max(z_min, sum_min), z_max);
                        overlap_max = max(min(z_max, sum_max), z_min);
                        
                        if overlap_max > overlap_min
                            % Величина пересечения
                            fraction = (overlap_max-overlap_min) / (sum_max-sum_min);
                            
                            % Вклад в вероятность попадания в полосу
                            total_lower = total_lower + A_lowerProb(i) * B_lowerProb(j) * fraction;
                            total_upper = total_upper + A_upperProb(i) * B_upperProb(j) * fraction;                           
                        end
                    end
                end
                
                % Перевод в плотность вероятности
                bin_width = sum_edges(k+1) - sum_edges(k);
                lower_pdf(k) = total_lower / bin_width;
                upper_pdf(k) = total_upper / bin_width;
            end
            
            % Валидность границ
            if any(lower_pdf > upper_pdf)
               error('Серди нижних границ PDF есть значения, превышающие верхнии границы PDF.');
            end
            Z = Hist(sum_edges, lower_pdf, upper_pdf);        
        end
        
        function Z = mtimes(k, A)
            if ~isa(A, 'Hist') && ~isa(k, 'double')
                error('Операция умножения задана только для перемножения с коэффициентом.');
            end
            z_values = k * A.x;
            if k >= 0
               lowerZ = A.lowerPDF / k;
               upperZ = A.upperPDF / k;
            else
               lowerZ = -A.lowerPDF(end:-1:1) / k;
               upperZ = -A.upperPDF(end:-1:1) / k; 
            end
            Z = Hist(z_values, lowerZ, upperZ);
        end

        % построение графика гистограммы
        function plot(obj)
            figure; hold on;
            n = length(obj.x);
            for i = 2 : n   % цикл по полосам гистограммы 
                plot([obj.x(i-1), obj.x(i-1), obj.x(i), obj.x(i)], [0, obj.upperPDF(i-1), obj.upperPDF(i-1), 0], 'b', 'LineWidth', 1.5); 
                plot([obj.x(i-1), obj.x(i-1), obj.x(i), obj.x(i)], [0, obj.lowerPDF(i-1), obj.lowerPDF(i-1), 0], 'r', 'LineWidth', 1.5); 
            end
            xlabel('x'); ylabel('PDF');
            legend('верхняя граница Hist', 'нижняя граница Hist', 'Location', 'southeast');
            title('Гистограмма Берлинта');
            grid on;
        end

        % определение величины возможного среднеквадратического отклонения.
        function stdCint = getStd(obj, method)
            % характеристики полос гистограммы
            binEdges = obj.x;
            nBins = length(binEdges)-1;
            binWidths = diff(binEdges);
            binMids = 0.5*(binEdges(1:end-1)+binEdges(2:end));
            
            % границы возможной вероятности на полосу
            lowerProbs = obj.lowerPDF .* binWidths;
            upperProbs = obj.upperPDF .* binWidths;
            
            % валидация получившихся границ вероятности
            if sum(lowerProbs) > 1 || sum(upperProbs) < 1
                error('Границы высот полос гистограммы заданы с ошибками.');
            end

            if (nargin < 2) || ((nargin == 2) && strcmp(method,'approximate'))
                m1 = sum(mean([upperProbs, lowerProbs]) .* binMids);
    
                varCint = NaN(1,2);
                varCint(1) = sum(lowerProbs.*((binMids-m1).^2)) + (1/12)*sum(lowerProbs.*(binWidths.^2));
                varCint(2) = sum(upperProbs.*((binMids-m1).^2)) + (1/12)*sum(upperProbs.*(binWidths.^2));
                stdCint = sqrt(varCint); 
            
            elseif strcmp(method,'accurate')

                % задание условий оптимизации
                Aeq = ones(1, nBins);
                beq = 1;
                lb = lowerProbs;
                ub = upperProbs;
                
                % оптимизация
                options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp', 'MaxFunctionEvaluations', 10^4);
                p0 = (lowerProbs + upperProbs)/2;  % начальное приближение
    
                % 1. минимальное значение дисперсии
                objFun = @(p) (p*(binMids.^2)' - (p*binMids')^2);
                p_min = fmincon(objFun, p0, [], [], Aeq, beq, lb, ub, [], options);
                minVar = objFun(p_min);
                
                % 2. максимальное значение дисперсии
                objFun = @(p) -(p*(binMids.^2)' - (p*binMids')^2);
                p_max = fmincon(objFun, p0, [], [], Aeq, beq, lb, ub, [], options);
                maxVar = -objFun(p_max);
    
                stdCint = sqrt([minVar, maxVar]);

            else
                error('Аргумент method не задан. Допустимые значения: approximate, accurate.');
            end
        end
        
        % преобразование в PBox
        function pbox = Hist2PBox(obj)
            % вычисление границ CDF
            lowerProbs = obj.lowerPDF .* diff(obj.x);
            upperProbs = obj.upperPDF .* diff(obj.x);
            
            lowerCDF = [0, cumsum(lowerProbs)];
            upperCDF = [0, cumsum(upperProbs)];
            
            % нормализация для обеспечения верных границ для CDF
            lowerCDF(lowerCDF > 1) = 1.0;
            upperCDF(upperCDF > 1) = 1.0;
            
            if lowerCDF(end) < 1
               lowerCDF(end+1) = 1.0;
               upperCDF(end+1) = 1.0;
            end
            pbox = PBox([obj.x, obj.x(end)+eps], lowerCDF, upperCDF);
        end

        % преобразование в DempsterShafer
        function ds = Hist2DempsterShafer(obj)
            % создание фокальных элементов из полос гистограммы
            intervals = [obj.x(1:end-1)' obj.x(2:end)'];
            
            % вычисление масс
            lowerMasses = obj.lowerPDF .* diff(obj.x);
            upperMasses = obj.upperPDF .* diff(obj.x);
            
            % создание структуры Демпстера-Шафера
            masses = (lowerMasses + upperMasses) / 2;  % аппроксимация средним
            masses = masses / sum(masses);  % Normalize
            
            ds = DempsterShafer(intervals, masses);
        end

        % преобразование в Fuzzy
        function fv = Hist2Fuzzy(obj, numUniverse)
            if nargin < 2
                numUniverse = length(obj.x)-1;
            end
            bin_centers = 0.5 * (obj.x(1:end-1) + obj.x(2:end));
            % определение функция принадлежности по верхним границам PDF
            x_universe = linspace(min(obj.x), max(obj.x), numUniverse);
            membership = interp1(bin_centers, obj.upperPDF.*diff(obj.x), x_universe, 'nearest', 0);
            
            fv = Fuzzy(x_universe, membership);
        end

        % преобразование в FuzzyInterval
        function fi = Hist2FuzzyInterval(obj, numAlpha)
            if nargin < 2
                numAlpha = length(obj.x)-1;
            end
            alpha_levels = linspace(0, 1, numAlpha);
            alpha_levels = sort(alpha_levels, 'descend');
            
            intervals = zeros(numAlpha, 2);
            for i = 1:numAlpha
                alpha = alpha_levels(i);
                valid_bins = obj.upperPDF >= alpha;
                
                if any(valid_bins)
                    intervals(i,1) = obj.x(find(valid_bins, 1, 'first'));
                    intervals(i,2) = obj.x(find(valid_bins, 1, 'last')+1);
                else
                    intervals(i,:) = [NaN NaN];
                end
            end 
            fi = FuzzyInterval(alpha_levels, intervals);
        end
    end
end
