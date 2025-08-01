from setuptools import find_packages, setup

def get_requirements(file_path):
    """
    This function finds all the reqirements for the package.
    """

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()

        requirements = [
            req.strip()
            for req in requirements
            if req.strip() and not req.strip().startswith("-e .")
        ]
    
    return requirements

setup(
    name="ml_algorithms_from_scratch",
    version="0.0.1",
    author="Saiduzzaman Zishan",
    author_email="winchesterfelix007@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)