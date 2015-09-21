robocopy %RECIPE_DIR%\.. . /E /NFL /NDL

"%PYTHON%" setup.py install --single-version-externally-managed --record=%TEMP%record.txt

if errorlevel 1 exit 1
