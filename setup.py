from setuptools import setup
setup(
    name='aidp',
    version='0.0.1',
    entry_points={
        'console_scripts': [
            'aidp=aidp:main'
        ]
    }
)