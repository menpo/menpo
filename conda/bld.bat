"%PYTHON%" setup.py install --single-version-externally-managed --record=%TEMP%record.txt
if errorlevel 1 exit 1

:: Add more build steps here, if they are necessary.

:: See
:: https://github.com/ContinuumIO/conda/blob/master/conda/builder/README.txt
:: for a list of environment variables that are set during the build process.
