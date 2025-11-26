@echo off
echo Kutuphaneler yukleniyor...
python -m pip install pandas
python -m pip install numpy
python -m pip install matplotlib
python -m pip install seaborn
python -m pip install scikit-learn
python -m pip install tensorflow
python -m pip install streamlit
echo.
echo Yukleme tamamlandi!
python check_imports.py
pause

