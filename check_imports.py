"""Kütüphane kontrol scripti"""
try:
    import pandas
    print("✓ pandas yüklü")
except ImportError:
    print("✗ pandas yüklü değil")

try:
    import numpy
    print("✓ numpy yüklü")
except ImportError:
    print("✗ numpy yüklü değil")

try:
    import matplotlib
    print("✓ matplotlib yüklü")
except ImportError:
    print("✗ matplotlib yüklü değil")

try:
    import seaborn
    print("✓ seaborn yüklü")
except ImportError:
    print("✗ seaborn yüklü değil")

try:
    import sklearn
    print("✓ scikit-learn yüklü")
except ImportError:
    print("✗ scikit-learn yüklü değil")

try:
    import tensorflow
    print("✓ tensorflow yüklü")
except ImportError:
    print("✗ tensorflow yüklü değil")

try:
    import streamlit
    print("✓ streamlit yüklü")
except ImportError:
    print("✗ streamlit yüklü değil")

