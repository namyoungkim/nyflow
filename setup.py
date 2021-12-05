from setuptools import setup, find_packages
 
setup(
    name = 'nyflow',
    version = '0.0.7.6',
    description = 'nyfow is deeplearning framwork',
    author = 'namyoungKim',
    author_email = 'liniarq@gmail.com',
    url = 'https://github.com/namyoungkim/nyflow',
    download_url = 'https://github.com/namyoungkim/nyflow/archive/master.zip',
    install_requires =  [],
    packages = find_packages(exclude = []),
    keywords = ['nyflow'],
    python_requires = '>=3',
    package_data = {},
    zip_safe = False,
    classifiers = [
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)