import sys
import platform
import pip
import pkg_resources


def get_installed_packages():
    installed_packages = sorted([f"{d.project_name}=={d.version}" for d in pkg_resources.working_set])
    return installed_packages


def main():
    print("Python Environment Details:")
    print("---------------------------")
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print(f"Platform: {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    print(f"Processor: {platform.processor()}")
    print("\nInstalled Packages:")
    print("------------------")
    for package in get_installed_packages():
        print(package)


if __name__ == "__main__":
    main()
