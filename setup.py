from setuptools import setup, find_namespace_packages
from pathlib import Path

# Load the local files
here = Path(__file__).parent.resolve()
requirements = (here / "requirements.txt").read_text(encoding="utf8")
long_description = (here / 'README.md').read_text(encoding='utf-8')
version = (here / 'VERSION').read_text().rstrip("\n")

setup(name='ydata-quality',
      version=version,
      description='YData open-source tools for Data Quality.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='YData',
      author_email='community@ydata.ai',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Telecommunications Industry',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: Implementation',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      keywords='data science ydata',
      url='https://github.com/ydataai/ydata-quality',
      license="https://github.com/ydataai/ydata-quality/blob/master/LICENSE",
      python_requires=">3.9",
      packages=find_namespace_packages('src'),
      package_dir={'':'src'},
      include_package_data=True,
      options={"bdist_wheel": {"universal": True}},
      install_requires=requirements)
