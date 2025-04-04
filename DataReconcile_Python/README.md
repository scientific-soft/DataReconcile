
Пошаговая инструкция по развертыванию библиотеки DataReconcile (Python)

1. Если ранее этого не было сделано, Скачайте и установите интерпретатор Python версии 3.0 или новее.

2. Скачайте (А) файлы библиотеки DataReconcile, либо клонируйте (Б) на вычислительную машину, где планируется сборка проекта, использующего функционал библиотеки.
2.А Как скачать содержимое репозитория DataReconcile:
o	Перейдите на главную страницу репозитория по ссылке https://github.com/scientific-soft/DataReconcile
o	Кликните на кнопку «<>», расположенную над списком содержащихся в корневой директории фалов и папок библиотеки.
 
o	Во всплывающем меню выберите вкладку «Local» (1), после чего кликнете на строку «Download ZIP» (2).
 
o	После выполнения данных операций содержимое библиотеки начнет скачиваться в директорию загрузки на локальном компьютере в виде файла с расширением «.zip».
o	После того, как скачивание завершено, скаченный архивный файл «DataReconcile-main.zip» необходимо разархивировать программой архиватором, поддерживающей формат «.zip».
 
o	Результат скачивания файлов библиотеки – папка с именем DataReconcile-main, содержимое которой приведено на скриншоте ниже:
 
2.Б Как клонировать содержимое репозитория DataReconcile:
o	Перейдите на главную страницу репозитория по ссылке https://github.com/scientific-soft/DataReconcile
o	Кликните на кнопку «<>», расположенную над списком содержащихся в корневой директории фалов и папок библиотеки.
 
o	Скопируйте URL репозитория в буфер обмена:
1.	Чтобы клонировать репозиторий с помощью HTTPS, нажмите кнопку копирования в разделе «HTTPS».
2.	Чтобы клонировать репозиторий с помощью SSH ключа, включая сертификат, выданный центром сертификации SSH вашей организации, нажмите кнопку копирования в разделе SSH.
3.	Чтобы клонировать репозиторий с помощью GitHub CLI, нажмите кнопку копирования в разделе GitHub CLI.
 		
o	Откройте консоль Git Bash.
o	Измените текущий рабочий каталог на место, куда вы хотите клонировать содержимое DataReconcile.
o	Введите git clone, а затем вставьте URL-адрес, скопированный ранее в формате 
git clone https://github.com/scientific-soft/DataReconcile.git
o	Нажмите Enter, чтобы создать локальный клон библиотеки.

3. Установите следующие библиотеки языка Python: numpy, scipy. Для этой цели можно воспользоваться модулем pip.

4. Далее выполните одну из двух операций:
4.1 Разместите необходимые вам модули библиотеки DataReconcile из папки DataReconcile_Python в директории вашего исполняемого файла .py.
Пример добавления функции из библиотеки, именуемой условным именем my_function и хранящейся в файле с тем же именем my_function приведен ниже: 
from my_function import *
4.2 В начало исполняемого скрипта .py, в которым вы желаете воспользоваться функцией/ями из библиотеки допишите путь к директории, где сохранены/куда клонированы модули из папки DataReconcile_Python.
Пример указания пути и добавления функции из библиотеки, именуемой условным именем my_function и хранящейся в файле с тем же именем my_function  приведен ниже:
import sys
sys.path.append('/path/to/directory/containing/my_function')
from my_function import *

5. Все доступные функции по названию совпадают с названием файла расширением «.py». Если возникает ошибка вызова функции из библиотеки, проверьте, действительно в директории по указанному пути присутствует файл с названием функции. 
