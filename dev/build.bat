@echo off

set app_name=AI_OS
set root=..\dist\%app_name%

echo Preserving data...
if exist ..\dist\%app_name%\data\ (
    xcopy /Y /E ..\dist\%app_name%\data\ data\
) else (
    echo No data folder found. Skipping copy.
)

echo Cleaning old build...
rmdir /s /q ..\dist\
rmdir /s /q ..\build\
del /q AI_OS.spec

echo Building .exe with PyInstaller...
pyinstaller --onedir --name observe ..\src\observe.py --distpath %root% --workpath ..\build\
pyinstaller --onedir --name reflect ..\src\reflect.py --distpath %root% --workpath ..\build\
pyinstaller --onedir --name act ..\src\act.py --distpath %root% --workpath ..\build\ ^
    --hidden-import=sklearn ^
    --hidden-import=sklearn._cyutility ^
    --hidden-import=sklearn.pipeline ^
    --hidden-import=sklearn.feature_extraction.text
pyinstaller --onedir --name benchmark_cli ..\benchmark_cli.py --distpath %root% --workpath ..\build\ --add-data "..\src\Lib\site-packages\llama_cpp\lib;llama_cpp\lib"
pyinstaller --onedir --name install ..\install.py --distpath %root% --workpath ..\build\

echo Merging executables...
set apps=observe reflect act benchmark_cli install
for %%A in (%apps%) do (
    echo Merging %%A...
    robocopy "%root%\%%A" "%root%" /E /XC /XN /XO
    rmdir /s /q "%root%\%%A"
)

echo Copying Requirements...
xcopy /Y /E ..\requirements\ ..\dist\%app_name%\requirements\

echo Copying required Libraries...
xcopy /Y /E ..\bin\ ..\dist\%app_name%\bin\

echo Copying SQL...
xcopy /Y /E ..\sql\ ..\dist\%app_name%\sql\

echo Copying Data...
if exist data\ (
    xcopy /Y /E data\ ..\dist\%app_name%\data\
    rmdir /s /q data\
) else (
    echo No data folder found. Skipping copy.
)