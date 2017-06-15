from setuptools import setup


setup(name='sip_tpv',
      version="1.1",
      description='Conversion of distortion representations in FITS headers between SIP and TPV formats.',
      url='https://github.com/stargaser/sip_tpv',
      author='David Shupe',
      author_email='shupe@ipac.caltech.edu',
      license='BSD',
      packages=['sip_tpv'],
      install_requires=[
          'numpy',
          'astropy',
          'sympy',
      ],
      classifiers=['Intended Audience :: Science/Research',
                   'Topic :: Scientific/Engineering :: Astronomy',
                   'License :: OSI Approved :: BSD License',
                   'Programming Language :: Python',
                   ]
      )
