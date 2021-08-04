

nuget install OpenBLAS -Version 0.2.14.1 -OutputDirectory /cibw/openblas -Verbosity detailed

echo "[openblas]" | Out-File -FilePath ~/.numpy-site.cfg -Encoding utf8

echo "libraries = openblas" | Out-File -FilePath ~/.numpy-site.cfg -Encoding utf8 -Append
echo 'library_dirs = D:\cibw\openblas\OpenBLAS.0.2.14.1\lib\native\lib\x64' | Out-File -FilePath ~/.numpy-site.cfg -Encoding utf8 -Append
echo 'include_dir = D:\cibw\openblas\OpenBLAS.0.2.14.1\lib\native\include' | Out-File -FilePath ~/.numpy-site.cfg -Encoding utf8 -Append

TREE "D:\\cibw\\openblas" /F

Rename-Item "D:\\cibw\\openblas\\OpenBLAS.0.2.14.1\\lib\\native\\lib\\x64\\libopenblas.dll.a" "libopenblas.lib"
Rename-Item "D:\\cibw\\openblas\\OpenBLAS.0.2.14.1\\lib\\native\\lib\\win32\\libopenblas.dll.a" "libopenblas.lib"

TREE "D:\\cibw\\openblas" /F
