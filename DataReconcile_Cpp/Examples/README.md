Пошаговая инструкция по запуску примеров применения модулей библиотеки C++ 
библиотеки DataReconcile.

0. Если это не было сделано ранее, установите компилятор версией не старше C++11.
(инструкцию по пошаговой установке можно найти в директории DataReconcile_Сpp, в файле README)

1. Клонируйте репозиторий (команды bash консоли):
git clone https://github.com/scientific-soft/DataReconcile.git
cd DataReconcile

2. Установите зависимости - библиотеку Eigen одним из следующих способов (в зависимости 
от используемой операционной системы):

2.А Windows (MinGW)
(команды bash консоли)
pacman -Syu --noconfirm
pacman -S --noconfirm mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-eigen3 git

2.Б Linux (Debian/Ubuntu)
(команды bash консоли)
sudo apt update
sudo apt install -y g++ cmake libeigen3-dev git

2.С MacOS (Homebrew)
(команды bash консоли)
brew update
brew install gcc cmake eigen git

3. Скомпилируйте интересующий вас пример. Это можно сделать несколькими способами:

3.А. Прямая компиляция
g++ -std=c++11 -I./include -I/usr/include/eigen3 Example_XX__XXX.cpp -o example
Примечание: если потребуется, замените -I/usr/include/eigen3 на фактический путь до библиотеки 
Eigen в вашей системе.

3.Б. Сборка CMake-ом
Создайте CMakeLists.txt со следующим содержимым (Example_XX_XXX замените на название примера,
который хотите запустить)

cmake_minimum_required(VERSION 3.10)
project(DRExample)
find_package(Eigen3 REQUIRED)
add_executable(example Example_XX__XXX.cpp)
target_link_libraries(example Eigen3::Eigen)

Выполните сборку:
mkdir build && cd build
cmake .. && make

4. Запустите исполняемый файл (результаты согласования будут выведены в окне с консолью).
