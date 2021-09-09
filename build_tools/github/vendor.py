# TODO: This file is adapted from scikit-learn

"""Embed vcomp140.dll, vcruntime140.dll and vcruntime140_1.dll.
Note that vcruntime140_1.dll is only required (and available)
for 64-bit architectures.
"""


import os
import os.path as op
import shutil
import sys
import textwrap


TARGET_FOLDER = op.join("gensvm", ".libs")
DISTRIBUTOR_INIT = op.join("gensvm", "_distributor_init.py")
VCOMP140_SRC_PATH = "C:\\Windows\\System32\\vcomp140.dll"
VCRUNTIME140_SRC_PATH = "C:\\Windows\\System32\\vcruntime140.dll"
VCRUNTIME140_1_SRC_PATH = "C:\\Windows\\System32\\vcruntime140_1.dll"

OPENBLAS_LIB_32_PATH = "D:\\cibw\\OpenBLAS\\OpenBLAS.0.2.14.1\\lib\\native\\bin\\win32\\libopenblas.dll"
OPENBLAS_LIB_64_PATH = "D:\\cibw\\OpenBLAS\\OpenBLAS.0.2.14.1\\lib\\native\\bin\\x64\\libopenblas.dll"

OPENBLAS_32_LIBGCC = "D:\\cibw\\OpenBLAS\\OpenBLAS.0.2.14.1\\lib\\native\\bin\\win32\\libgcc_s_sjlj-1.dll"
OPENBLAS_32_FORTRAN = "D:\\cibw\\OpenBLAS\\OpenBLAS.0.2.14.1\\lib\\native\\bin\\win32\\libgfortran-3.dll"
OPENBLAS_32_OPENBLAS = "D:\\cibw\\OpenBLAS\\OpenBLAS.0.2.14.1\\lib\\native\\bin\\win32\\libopenblas.dll"
OPENBLAS_32_QUADMATH = "D:\\cibw\\OpenBLAS\\OpenBLAS.0.2.14.1\\lib\\native\\bin\\win32\\libquadmath-0.dll"


def make_distributor_init_32_bits(
    distributor_init,
    vcomp140_dll_filename,
    vcruntime140_dll_filename,
    openblas_gcc_filename,
    openblas_fortran_filename,
    openblas_lib_filename,
    openblas_quadmath_filename,
):
    """Create a _distributor_init.py file for 32-bit architectures.
    This file is imported first when importing the sklearn package
    so as to pre-load the vendored vcomp140.dll and vcruntime140.dll.
    """
    with open(distributor_init, "wt") as f:
        f.write(
            textwrap.dedent(
                """
            '''Helper to preload vcomp140.dll and vcruntime140.dll to
            prevent "not found" errors.
            Once vcomp140.dll and vcruntime140.dll are preloaded, the
            namespace is made available to any subsequent vcomp140.dll
            and vcruntime140.dll. This is created as part of the scripts
            that build the wheel.
            '''
            import os
            import os.path as op
            from ctypes import WinDLL
            if os.name == "nt":
                # Load vcomp140.dll and vcruntime140.dll
                libs_path = op.join(op.dirname(__file__), ".libs")

                vcomp140_dll_filename = op.join(libs_path, "{0}")
                vcruntime140_dll_filename = op.join(libs_path, "{1}")
                openblas_gcc_filename = op.join(libs_path, "{2}")
                openblas_fortran_filename = op.join(libs_path, "{3}")
                openblas_lib_filename = op.join(libs_path, "{4}")
                openblas_quadmath_filename = op.join(libs_path, "{5}")

                print("vcomp140_dll_filename", vcomp140_dll_filename)
                print("vcruntime140_dll_filename", vcruntime140_dll_filename)
                print("openblas_gcc_filename", openblas_gcc_filename)
                print("openblas_fortran_filename", openblas_fortran_filename)
                print("openblas_lib_filename", openblas_lib_filename)
                print("openblas_quadmath_filename", openblas_quadmath_filename)

                print("Loading vcomp140")
                WinDLL(op.abspath(vcomp140_dll_filename))
                print("Loading vcruntime140")
                WinDLL(op.abspath(vcruntime140_dll_filename))
                print("Loading openblas_gcc")
                WinDLL(op.abspath(openblas_gcc_filename))
                print("Loading openblas_fortran")
                WinDLL(op.abspath(openblas_fortran_filename))
                print("Loading openblas_quadmath")
                WinDLL(op.abspath(openblas_quadmath_filename))
                print("Loading openblas_lib")
                WinDLL(op.abspath(openblas_lib_filename))

            """.format(
                    vcomp140_dll_filename,
                    vcruntime140_dll_filename,
                    openblas_gcc_filename,
                    openblas_fortran_filename,
                    openblas_lib_filename,
                    openblas_quadmath_filename,
                )
            )
        )


def make_distributor_init_64_bits(
    distributor_init,
    vcomp140_dll_filename,
    vcruntime140_dll_filename,
    vcruntime140_1_dll_filename,
    openblas_lib_filename,
):
    """Create a _distributor_init.py file for 64-bit architectures.
    This file is imported first when importing the sklearn package
    so as to pre-load the vendored vcomp140.dll, vcruntime140.dll
    and vcruntime140_1.dll.
    """
    with open(distributor_init, "wt") as f:
        f.write(
            textwrap.dedent(
                """
            '''Helper to preload vcomp140.dll, vcruntime140.dll and
            vcruntime140_1.dll to prevent "not found" errors.
            Once vcomp140.dll, vcruntime140.dll and vcruntime140_1.dll are
            preloaded, the namespace is made available to any subsequent
            vcomp140.dll, vcruntime140.dll and vcruntime140_1.dll. This is
            created as part of the scripts that build the wheel.
            '''
            import os
            import os.path as op
            from ctypes import WinDLL
            if os.name == "nt":
                # Load vcomp140.dll, vcruntime140.dll and vcruntime140_1.dll
                libs_path = op.join(op.dirname(__file__), ".libs")
                vcomp140_dll_filename = op.join(libs_path, "{0}")
                vcruntime140_dll_filename = op.join(libs_path, "{1}")
                vcruntime140_1_dll_filename = op.join(libs_path, "{2}")
                openblas_lib_filename = op.join(libs_path, "{3}")
                WinDLL(op.abspath(vcomp140_dll_filename))
                WinDLL(op.abspath(vcruntime140_dll_filename))
                WinDLL(op.abspath(vcruntime140_1_dll_filename))
                WinDLL(op.abspath(openblas_lib_filename))
            """.format(
                    vcomp140_dll_filename,
                    vcruntime140_dll_filename,
                    vcruntime140_1_dll_filename,
                    openblas_lib_filename,
                )
            )
        )


def main(wheel_dirname, bitness):
    """Embed vcomp140.dll, vcruntime140.dll and vcruntime140_1.dll."""
    if not op.exists(VCOMP140_SRC_PATH):
        raise ValueError(f"Could not find {VCOMP140_SRC_PATH}.")

    if not op.exists(VCRUNTIME140_SRC_PATH):
        raise ValueError(f"Could not find {VCRUNTIME140_SRC_PATH}.")

    if not op.exists(VCRUNTIME140_1_SRC_PATH) and bitness == "64":
        raise ValueError(f"Could not find {VCRUNTIME140_1_SRC_PATH}.")

    if not op.isdir(wheel_dirname):
        raise RuntimeError(f"Could not find {wheel_dirname} file.")

    vcomp140_dll_filename = op.basename(VCOMP140_SRC_PATH)
    vcruntime140_dll_filename = op.basename(VCRUNTIME140_SRC_PATH)
    vcruntime140_1_dll_filename = op.basename(VCRUNTIME140_1_SRC_PATH)

    target_folder = op.join(wheel_dirname, TARGET_FOLDER)
    distributor_init = op.join(wheel_dirname, DISTRIBUTOR_INIT)

    # Create the "gensvm/.libs" subfolder
    if not op.exists(target_folder):
        os.mkdir(target_folder)

    print(f"Copying {VCOMP140_SRC_PATH} to {target_folder}.")
    shutil.copy2(VCOMP140_SRC_PATH, target_folder)

    print(f"Copying {VCRUNTIME140_SRC_PATH} to {target_folder}.")
    shutil.copy2(VCRUNTIME140_SRC_PATH, target_folder)

    if bitness == "64":
        print(f"Copying {VCRUNTIME140_1_SRC_PATH} to {target_folder}")
        shutil.copy2(VCRUNTIME140_1_SRC_PATH, target_folder)

    if bitness == "32":
        print(f"Copying {OPENBLAS_32_LIBGCC} to {target_folder}")
        shutil.copy2(OPENBLAS_32_LIBGCC, target_folder)
        openblas_gcc_filename = op.basename(OPENBLAS_32_LIBGCC)

        print(f"Copying {OPENBLAS_32_FORTRAN} to {target_folder}")
        shutil.copy2(OPENBLAS_32_FORTRAN, target_folder)
        openblas_fortran_filename = op.basename(OPENBLAS_32_FORTRAN)

        print(f"Copying {OPENBLAS_32_OPENBLAS} to {target_folder}")
        shutil.copy2(OPENBLAS_32_OPENBLAS, target_folder)
        openblas_lib_filename = op.basename(OPENBLAS_32_OPENBLAS)

        print(f"Copying {OPENBLAS_32_QUADMATH} to {target_folder}")
        shutil.copy2(OPENBLAS_32_QUADMATH, target_folder)
        openblas_quadmath_filename = op.basename(OPENBLAS_32_QUADMATH)
    else:
        print(f"Copying {OPENBLAS_LIB_64_PATH} to {target_folder}")
        shutil.copy2(OPENBLAS_LIB_64_PATH, target_folder)
        openblas_lib_filename = op.basename(OPENBLAS_LIB_64_PATH)

    # Generate the _distributor_init file in the source tree
    print("Generating the '_distributor_init.py' file.")
    if bitness == "32":
        make_distributor_init_32_bits(
            distributor_init,
            vcomp140_dll_filename,
            vcruntime140_dll_filename,
            openblas_gcc_filename,
            openblas_fortran_filename,
            openblas_lib_filename,
            openblas_quadmath_filename,
        )
    else:
        make_distributor_init_64_bits(
            distributor_init,
            vcomp140_dll_filename,
            vcruntime140_dll_filename,
            vcruntime140_1_dll_filename,
            openblas_lib_filename,
        )


if __name__ == "__main__":
    _, wheel_file, bitness = sys.argv
    main(wheel_file, bitness)
