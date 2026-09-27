# CMake generated Testfile for 
# Source directory: C:/Users/Cutie Magic 500/projects/creative/screen-ocr
# Build directory: C:/Users/Cutie Magic 500/projects/creative/screen-ocr
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test(test_core "C:/Users/Cutie Magic 500/projects/creative/screen-ocr/Debug/test_core.exe")
  set_tests_properties(test_core PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/Cutie Magic 500/projects/creative/screen-ocr/CMakeLists.txt;23;add_test;C:/Users/Cutie Magic 500/projects/creative/screen-ocr/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test(test_core "C:/Users/Cutie Magic 500/projects/creative/screen-ocr/Release/test_core.exe")
  set_tests_properties(test_core PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/Cutie Magic 500/projects/creative/screen-ocr/CMakeLists.txt;23;add_test;C:/Users/Cutie Magic 500/projects/creative/screen-ocr/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test(test_core "C:/Users/Cutie Magic 500/projects/creative/screen-ocr/MinSizeRel/test_core.exe")
  set_tests_properties(test_core PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/Cutie Magic 500/projects/creative/screen-ocr/CMakeLists.txt;23;add_test;C:/Users/Cutie Magic 500/projects/creative/screen-ocr/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test(test_core "C:/Users/Cutie Magic 500/projects/creative/screen-ocr/RelWithDebInfo/test_core.exe")
  set_tests_properties(test_core PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/Cutie Magic 500/projects/creative/screen-ocr/CMakeLists.txt;23;add_test;C:/Users/Cutie Magic 500/projects/creative/screen-ocr/CMakeLists.txt;0;")
else()
  add_test(test_core NOT_AVAILABLE)
endif()
