@echo off

set app_name=AI_OS
set root=..\dist\%app_name%

echo Preserving Metadata...
if exist ..\dist\%app_name%\metadata\ (
    xcopy /Y /E ..\dist\%app_name%\metadata\ metadata\
) else (
    echo No metadata folder found. Skipping copy.
)

echo Cleaning old build...
rmdir /s /q ..\dist\
rmdir /s /q ..\build\
del /q AI_OS.spec

echo Building .exe with PyInstaller...
pyinstaller --onedir --name observe ..\src\observe.py --distpath %root% --workpath ..\build\
pyinstaller --onedir --name benchmark_cli ..\benchmark_cli.py --distpath %root% --workpath ..\build\ --add-data "..\src\Lib\site-packages\llama_cpp\lib;llama_cpp\lib"
pyinstaller --onedir --name install ..\install.py --distpath %root% --workpath ..\build\

echo Merging executables...
set apps=observe benchmark_cli install
for %%A in (%apps%) do (
    echo Merging %%A...
    robocopy "%root%\%%A" "%root%" /E /XC /XN /XO
    rmdir /s /q "%root%\%%A"
)

echo Copying Requirements...
xcopy /Y /E ..\requirements\ ..\dist\%app_name%\requirements\

echo Copying required DLLs...
xcopy /Y /E ..\bin\ ..\dist\%app_name%\bin\

echo Copying Metadata...
if exist metadata\ (
    xcopy /Y /E metadata\ ..\dist\%app_name%\metadata\
    rmdir /s /q metadata\
) else (
    echo No metadata folder found. Skipping copy.
)