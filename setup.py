from setuptools import setup,find_packages
hypen="-e ."
def  get_requires(file_path):
    requirements=[]
    with open(file_path) as file:
        requirements=file.readlines()
        requirements=[i.replace('/n'," ") for i in requirements]
        if hypen in requirements:
            requirements.remove(hypen)
    return list(requirements)

setup(
    name='reliance_stock_price_forecast',
    author=['manoj','Rakesh','Akash','Harshitha'],
    author_email='manojthupakula06080@gmail.com',
    version='0.0.1',
    packages=find_packages(),
    install_requires=get_requires('requirements.txt')
)