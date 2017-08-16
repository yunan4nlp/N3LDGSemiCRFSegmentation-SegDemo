##HOW TO COMPILE THIS PROJECT IN WINDOWS</br>
* Step 0: Open cmd, and change directory to project directory. Use this command </br> `cd /your/project/path/SemiCRFSegmentation`. </br>
* Step 1: Create a new directory in SemiCRFSegmentation.For example, use this command `mkdir build` </br>
* Step 2: Change your directory. Use this command `cd build`. </br>
* Step 3: Build project. Use this command </br> `cmake .. -DEIGEN3_DIR=/your/eigen/path -DN3LDG_DIR=/your/N3LDG/path`. </br>
* Step 4: Then you can double click "SemiCRFSegmentation.sln" to open this project. </br>
* Step 5: Now you can compile this project by Visual Studio. </br>
* Step 6: If you want to run this project.Please open project properties and add this argument. </br>
`-train /your/training/corpus -dev /your/development/corpus -test /your/test/corpus -option /your/option/file -l` </br>

##NOTE</br> 
Make sure you have eigen ,N3LDG, cmake and visual studio 2013 version (or newer). </br>
* Eigen:http://eigen.tuxfamily.org/index.php?title=Main_Page </br>
* N3LDG:https://github.com/zhangmeishan/N3LDG </br>
* cmake:https://cmake.org/</br>
* Visual Studio 2013: https://www.visualstudio.com/zh-hans/downloads/
