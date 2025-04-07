Пошаговая инструкция по развертыванию библиотеки DataReconcile (C++)

1. Если ранее этого не было сделано, скачайте и установите компилятор С++ версии 11 или новее.

1.А Установка компилятора C++ на Windows
Способ 1: установка Microsoft Visual Studio
Загрузить Visual Studio (Community Edition)
Перейдите по ссылке https://visualstudio.microsoft.com/downloads/
Загрузить "Visual Studio Community" (бесплатная версия).
Запустите установщик
Выберите рабочую нагрузку "Desktop development with C++".
Убедитесь, что отмечен "MSVC (Microsoft Visual C++)".
Нажмите "Install" и дождитесь завершения (~5-20 ГБ).
Проверка установки:
Откройте командную строку (Win + R → cmd).
Проверьте компилятор (sh):
cl /?
Если не распознается, перезагрузите ПК или вручную добавьте в PATH.

Способ 2: Установка MinGW (альтернативный)
Скачать MinGW
Перейти на https://sourceforge.net/projects/mingw/
Скачать mingw-get-setup.exe.
Запустить установщик
Выбрать "mingw32-gcc-g++" во время установки.
Перейти к установке → Применить изменения.
Добавить MinGW в PATH
Открыть переменные среды (Win + S → "Изменить переменные среды").
Добавить C:\MinGW\bin в PATH.
Проверить (sh)
g++ --version
Вывод должен отображать версию GCC (например, g++ (MinGW 8.1.0)).

Способ 3: 
Шаг 1: Загрузите MSYS2
Перейдите на официальный сайт MSYS2.
Загрузите установщик для своей системы:
64-разрядная Windows: msys2-x86_64-<version>.exe
32-разрядная Windows: msys2-i686-<version>.exe (требуется редко)
Шаг 2: Установите MSYS2
Запустите загруженный установщик.
Следуйте подсказкам, выбрав каталог установки по умолчанию (C:\msys64 для 64-разрядной версии).
Отметьте опцию «Запустить MSYS2 сейчас» в конце установки.
Шаг 3: Обновление пакетов MSYS2
В открытом терминале MSYS2 обновите базу данных пакетов (sh):
pacman -Syu
Если терминал попросит закрыться, перезапустите MSYS2 и запустите снова:
pacman -Su
Шаг 4: Установка компилятора C++ (GCC)
Установите набор инструментов MinGW-w64 (GCC для Windows):

Для 64-разрядной разработки (sh):
pacman -S --needed base-devel mingw-w64-x86_64-toolchain
Для 32-разрядной разработки (sh):
pacman -S --needed base-devel mingw-w64-i686-toolchain
Нажмите Enter, чтобы установить все пакеты по умолчанию.

Шаг 5: Добавьте MSYS2 в системный PATH
Откройте меню «Пуск» Windows → Найдите «Переменные среды» → Откройте «Изменить переменные среды системы».

Нажмите «Переменные среды» → В разделе «Системные переменные» выберите Путь → Нажмите «Изменить».

Добавьте эти пути (исправьте C:\msys64, если установлено в другом месте):
C:\msys64\mingw64\bin
C:\msys64\usr\bin
Нажмите OK → OK → OK.

Шаг 6: Проверка установки
Откройте новую командную строку (Win + R → cmd).
Проверьте версию GCC (sh):
g++ --version
Ожидаемый вывод (пример) (sh):
g++ (Rev10, Built by MSYS2 project) 13.2.0

1.Б Установка компилятора C++ в Linux
Debian/Ubuntu (APT)
Обновить пакеты (sh):
sudo apt update
Установить G++ (sh):
sudo apt install g++
Проверка (sh):
g++ --version
Вывод должен показывать GCC (например, g++ (Ubuntu 11.3.0)).

Fedora (DNF)
Установить G++ (sh):
sudo dnf install gcc-c++
Проверка (sh):
g++ --version
Arch Linux (Pacman)
Установить G++ (sh):
sudo pacman -S gcc
Проверка (sh):
g++ --version

1.В Установка компилятора C++ на macOS
Способ 1: Установка инструментов командной строки Xcode
Откройте терминал (⌘ + Пробел → Терминал).
Выполнить (sh):
xcode-select --install
Нажмите «Установить» во всплывающем окне.
Проверка (sh)
g++ --version
Вывод должен показывать Apple Clang (например, Apple clang 14.0.0).

Способ 2: Установка Homebrew GCC
Установка Homebrew (если не установлен) (sh):
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
Установка GCC (sh)
brew install gcc
Проверка (sh)
g++-13 --version # Замените на вашу версию GCC (например, g++-12)


2. Скачайте (А) файлы библиотеки DataReconcile, либо клонируйте (Б) на вычислительную машину, где планируется сборка проекта, использующего функционал библиотеки.
2.А Как скачать содержимое репозитория DataReconcile:
o	Перейдите на главную страницу репозитория по ссылке https://github.com/scientific-soft/DataReconcile
o	Кликните на кнопку «<>», расположенную над списком содержащихся в корневой директории фалов и папок библиотеки:
 
o	Во всплывающем меню выберите вкладку «Local» (1), после чего кликнете на строку «Download ZIP» (2).
 
o	После выполнения данных операций содержимое библиотеки начнет скачиваться в директорию загрузки на локальном компьютере в виде файла с расширением «.zip».
o	После того, как скачивание завершено, скаченный архивный файл «DataReconcile-main.zip» необходимо разархивировать программой архиватором, поддерживающей формат «.zip».
 
o	Результат скачивания файлов библиотеки – папка с именем DataReconcile-main, содержимое которой приведено на скриншоте ниже:

2.Б Как клонировать содержимое репозитория DataReconcile:
o	Перейдите на главную страницу репозитория по ссылке https://github.com/scientific-soft/DataReconcile
o	Кликните на кнопку «<>», расположенную над списком содержащихся в корневой директории фалов и папок библиотеки:
 
o	Скопируйте URL репозитория в буфер обмена:
1.	Чтобы клонировать репозиторий с помощью HTTPS, нажмите кнопку копирования в разделе «HTTPS».
2.	Чтобы клонировать репозиторий с помощью SSH ключа, включая сертификат, выданный центром сертификации SSH вашей организации, нажмите кнопку копирования в разделе SSH.
3.	Чтобы клонировать репозиторий с помощью GitHub CLI, нажмите кнопку копирования в разделе GitHub CLI.
 		
o	Откройте консоль Git Bash.
o	Измените текущий рабочий каталог на место, куда вы хотите клонировать содержимое DataReconcile.
o	Введите git clone, а затем вставьте URL-адрес, скопированный ранее в формате 
git clone https://github.com/scientific-soft/DataReconcile.git
o	Нажмите Enter, чтобы создать локальный клон библиотеки.

3. Установите зависимости - библиотеку Eigen одним из следующих способов (в зависимости 
от используемой операционной системы):

3.А Windows (MinGW)
(команды bash консоли)
pacman -Syu --noconfirm
pacman -S --noconfirm mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-eigen3 git

3.Б Linux (Debian/Ubuntu)
(команды bash консоли)
sudo apt update
sudo apt install -y g++ cmake libeigen3-dev git

3.С MacOS (Homebrew)
(команды bash консоли)
brew update
brew install gcc cmake eigen git

4. Рекомендуется установить среду разработки, поддерживающую компилятор С++

5. Интегрируйте библиотеку в свой проект
Вариант A: Прямое включение 
Скопируйте файлы .cpp и .h в свой проект.
Пример структуры:
your_project/
├── DataReconcileFiles/ # Скопированные файлы .cpp/.h
├── main.cpp # Ваш код
└── CMakeLists.txt # Конфигурация сборки
Примечание: DataReconcileFiles – условное обозначение файлов библиотеки, добавляемых в проект.
Включите в свой код (например, main.cpp):
#include "DataReconcile/DRnonparamEq.h"  // Example header
#include <Eigen/Dense>

int main() {
    Eigen::MatrixXd data(3, 10);
    // ... use DataReconcile functions ...
    return 0;
}
5.3	 Скомпилируйте вручную (задайте свой путь при необходимости):
(консоль bash, пример)
g++ -std=c++11 -I/path/to/eigen -I./DataReconcile main.cpp DataReconcile/*.cpp -o my_app
Вариант B: CMake
Создайте CMakeLists.txt (cmake):
cmake_minimum_required(VERSION 3.10)
project(MyProject)
# Find Eigen (replace path if Eigen is not in default locations)
find_package(Eigen3 REQUIRED)
# Add DataReconcile sources
file(GLOB RECONCILE_SOURCES " DataReconcileFile/*.cpp")
# Build your executable
add_executable(my_app main.cpp ${RECONCILE_SOURCES})
target_link_libraries(my_app Eigen3::Eigen)
target_include_directories(my_app PRIVATE "DataReconcile")
Примечание: DataReconcileFile – условное обозначение файла библиотеки, добавляемых в проект.

Сборка и запуск:
(консоль bash, пример)
mkdir build && cd build
cmake .. && make
./my_app

6. Все доступные функции для согласования данных, описанные в технической документации, по названию совпадают с названием файлов с расширением «.cpp». Если возникает ошибка вызова функции из библиотеки, проверьте, действительно в директории по указанному пути присутствует файл с названием функции. 

